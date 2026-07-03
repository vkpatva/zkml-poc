"""Paths and IDs for EZKL artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from app.config import ARTIFACTS_DIR, PROJECT_ROOT, WEIGHTS_BY_TIER

Visibility = Literal["public", "private"]
Tier = Literal["free", "premium"]

EZKL_ROOT = PROJECT_ROOT / "ezkl"
VISIBILITIES: tuple[Visibility, ...] = ("public", "private")
TIERS: tuple[Tier, ...] = ("free", "premium")

LOGROWS = 20
LOGIT_TOLERANCE = 0.01


def onnx_path(tier: Tier) -> Path:
    return ARTIFACTS_DIR / f"digit_mlp_{tier}.onnx"


def ezkl_dir(tier: Tier, visibility: Visibility) -> Path:
    return EZKL_ROOT / tier / visibility


def artifact_paths(tier: Tier, visibility: Visibility) -> dict[str, Path]:
    base = ezkl_dir(tier, visibility)
    return {
        "dir": base,
        "settings": base / "settings.json",
        "compiled": base / "network.ezkl",
        "srs": base / "kzg.srs",
        "vk": base / "vk.key",
        "pk": base / "pk.key",
        "manifest": base / "manifest.json",
        "witness": base / "witness.json",
        "proof": base / "proof.json",
        "runtime_input": base / "runtime_input.json",
    }


def model_id(tier: Tier, visibility: Visibility) -> str:
    vis = "private_input" if visibility == "private" else "public_audit"
    return f"digit_mlp_{tier}_{vis}"


def vk_hash(vk_path: Path) -> str:
    return hashlib.sha256(vk_path.read_bytes()).hexdigest()


def write_manifest(tier: Tier, visibility: Visibility) -> dict:
    paths = artifact_paths(tier, visibility)
    mid = model_id(tier, visibility)
    manifest = {
        "model_id": mid,
        "tier": tier,
        "visibility": visibility,
        "vk_hash": vk_hash(paths["vk"]),
        "onnx": str(onnx_path(tier).relative_to(PROJECT_ROOT)),
        "scenario": scenario_name(visibility),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(tier: Tier, visibility: Visibility) -> dict:
    path = artifact_paths(tier, visibility)["manifest"]
    if not path.is_file():
        raise FileNotFoundError(
            f"EZKL manifest not found at {path}. "
            f"Run: python -m proving.setup_ezkl --tier {tier} --visibility {visibility}"
        )
    return json.loads(path.read_text())


def scenario_name(visibility: Visibility) -> str:
    return {
        "public": "correct_inference_audit",
        "private": "private_input_proof",
    }[visibility]


def is_setup_complete(tier: Tier, visibility: Visibility) -> bool:
    paths = artifact_paths(tier, visibility)
    required = ("settings", "compiled", "srs", "vk", "pk", "manifest")
    return all(paths[key].is_file() for key in required)


def assert_tier(tier: str) -> Tier:
    if tier not in WEIGHTS_BY_TIER:
        raise ValueError(f"Unknown tier: {tier}. Expected one of: {list(WEIGHTS_BY_TIER)}")
    return tier  # type: ignore[return-value]


def assert_visibility(visibility: str) -> Visibility:
    if visibility not in VISIBILITIES:
        raise ValueError(f"Unknown visibility: {visibility}. Expected: {list(VISIBILITIES)}")
    return visibility  # type: ignore[return-value]
