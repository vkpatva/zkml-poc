"""EZKL setup, prove, and verify helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import ezkl

from proving.ezkl_paths import (
    LOGIT_TOLERANCE,
    LOGROWS,
    Tier,
    Visibility,
    artifact_paths,
    assert_tier,
    assert_visibility,
    is_setup_complete,
    model_id,
    onnx_path,
    vk_hash,
    write_manifest,
)


def _run_args(visibility: Visibility) -> ezkl.PyRunArgs:
    args = ezkl.PyRunArgs()
    args.input_visibility = "private" if visibility == "private" else "public"
    args.output_visibility = "public"
    args.param_visibility = "fixed"
    return args


def setup_tier(tier: Tier, visibility: Visibility, *, force: bool = False) -> dict:
    """One-time EZKL compile + key generation for a tier and visibility profile."""
    paths = artifact_paths(tier, visibility)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    model = str(onnx_path(tier))
    if not Path(model).is_file():
        raise FileNotFoundError(f"ONNX not found: {model}. Run: python -m training.export_onnx")

    settings = str(paths["settings"])
    compiled = str(paths["compiled"])
    srs = str(paths["srs"])
    vk = str(paths["vk"])
    pk = str(paths["pk"])

    if is_setup_complete(tier, visibility) and not force:
        return write_manifest(tier, visibility)

    run_args = _run_args(visibility)
    ezkl.gen_settings(model, settings, py_run_args=run_args)

    calib = str(Path(model).parent / f"input_{tier}.json")
    ezkl.calibrate_settings(calib, model, settings, "resources")
    ezkl.compile_circuit(model, compiled, settings)

    if not paths["srs"].is_file() or force:
        ezkl.gen_srs(srs, LOGROWS)

    ezkl.setup(compiled, vk, pk, srs)
    return write_manifest(tier, visibility)


def write_input_json(pixels: list[float], path: Path) -> None:
    path.write_text(json.dumps({"input_data": [pixels]}) + "\n")


def witness_logits(witness_path: Path) -> list[float]:
    data = json.loads(witness_path.read_text())
    rows = data["pretty_elements"]["rescaled_outputs"][0]
    return [float(v) for v in rows]


def prove_pixels(
    pixels: list[float],
    tier: str = "premium",
    visibility: str = "public",
    *,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Generate a ZK proof for one 64-pixel input."""
    t = assert_tier(tier)
    v = assert_visibility(visibility)
    if not is_setup_complete(t, v):
        raise FileNotFoundError(
            f"EZKL not set up for tier={t} visibility={v}. "
            f"Run: python -m proving.setup_ezkl --tier {t} --visibility {v}"
        )

    paths = artifact_paths(t, v)
    if write_artifacts:
        input_path = paths["runtime_input"]
        witness = paths["witness"]
        proof = paths["proof"]
    else:
        input_path = Path(tempfile.mkstemp(suffix=".json")[1])
        witness = Path(tempfile.mkstemp(suffix=".json")[1])
        proof = Path(tempfile.mkstemp(suffix=".json")[1])

    write_input_json(pixels, input_path)

    compiled = str(paths["compiled"])
    srs = str(paths["srs"])
    pk = str(paths["pk"])
    settings = str(paths["settings"])

    ezkl.gen_witness(str(input_path), compiled, str(witness), srs_path=srs)
    ezkl.prove(str(witness), compiled, pk, str(proof), srs)
    verified = bool(ezkl.verify(str(proof), settings, str(paths["vk"]), srs))

    # Portable proof must match on-disk format (verify expects byte-array `proof` field).
    proof_data = json.loads(Path(proof).read_text())
    logits = witness_logits(witness)
    digit = max(range(len(logits)), key=lambda i: logits[i])
    manifest = json.loads(paths["manifest"].read_text())

    return {
        "digit": digit,
        "logits": logits,
        "tier": t,
        "visibility": v,
        "model_id": manifest["model_id"],
        "vk_hash": manifest["vk_hash"],
        "scenario": manifest["scenario"],
        "verified": verified,
        "proof": proof_data,
        "settings_path": settings,
        "vk_path": str(paths["vk"]),
        "srs_path": srs,
    }


def verify_proof_payload(
    proof: dict[str, Any],
    tier: str,
    visibility: str = "public",
) -> dict[str, Any]:
    """Verify a proof dict returned by prove_pixels."""
    t = assert_tier(tier)
    v = assert_visibility(visibility)
    paths = artifact_paths(t, v)
    if not paths["manifest"].is_file():
        raise FileNotFoundError(f"EZKL manifest missing for tier={t} visibility={v}")

    manifest = json.loads(paths["manifest"].read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(proof, tmp)
        proof_path = tmp.name

    ok = bool(
        ezkl.verify(
            proof_path,
            str(paths["settings"]),
            str(paths["vk"]),
            str(paths["srs"]),
        )
    )
    Path(proof_path).unlink(missing_ok=True)
    return {
        "verified": ok,
        "model_id": manifest["model_id"],
        "vk_hash": manifest["vk_hash"],
        "tier": t,
        "visibility": v,
        "scenario": manifest["scenario"],
    }


def compare_logits(pytorch_logits: list[float], ezkl_logits: list[float]) -> dict[str, float | int | bool]:
    digit_pt = max(range(len(pytorch_logits)), key=lambda i: pytorch_logits[i])
    digit_ez = max(range(len(ezkl_logits)), key=lambda i: ezkl_logits[i])
    max_diff = max(abs(a - b) for a, b in zip(pytorch_logits, ezkl_logits))
    return {
        "digit_pytorch": digit_pt,
        "digit_ezkl": digit_ez,
        "digits_match": digit_pt == digit_ez,
        "max_logit_diff": max_diff,
        "within_tolerance": max_diff <= LOGIT_TOLERANCE,
    }
