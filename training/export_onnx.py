"""Export DigitMLP weights to ONNX for EZKL / onnxruntime."""

import json
import sys

import numpy as np
import onnxruntime as ort
import torch

from app.config import ARTIFACTS_DIR, PROJECT_ROOT, WEIGHTS_BY_TIER
from app.inference import predict
from app.model import DigitMLP

FIXTURE_INPUT = PROJECT_ROOT / "tests" / "fixtures" / "test_input.json"
ONNX_BY_TIER = {tier: ARTIFACTS_DIR / f"digit_mlp_{tier}.onnx" for tier in WEIGHTS_BY_TIER}
INPUT_JSON_BY_TIER = {
    tier: ARTIFACTS_DIR / f"input_{tier}.json" for tier in WEIGHTS_BY_TIER
}


def export_tier(tier: str) -> None:
    weights_path = WEIGHTS_BY_TIER[tier]
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Weights not found for tier '{tier}' at {weights_path}. "
            "Run: python -m training.train"
        )

    model = DigitMLP()
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    dummy = torch.randn(1, 64, dtype=torch.float32)
    onnx_path = ONNX_BY_TIER[tier]
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"  exported → {onnx_path}")


def write_sample_inputs() -> list[float]:
    with open(FIXTURE_INPUT) as f:
        pixels = json.load(f)["input_data"][0]

    for tier in WEIGHTS_BY_TIER:
        out_path = INPUT_JSON_BY_TIER[tier]
        with open(out_path, "w") as f:
            json.dump({"input_data": [pixels]}, f)
        print(f"  sample input → {out_path}")

    return pixels


def verify_tier(tier: str, pixels: list[float]) -> bool:
    pt_digit, _, pt_logits = predict(pixels, tier=tier)

    sess = ort.InferenceSession(
        str(ONNX_BY_TIER[tier]), providers=["CPUExecutionProvider"]
    )
    onnx_logits = sess.run(
        None, {"input": np.array([pixels], dtype=np.float32)}
    )[0][0]
    onnx_digit = int(onnx_logits.argmax())

    max_diff = max(abs(a - b) for a, b in zip(pt_logits, onnx_logits.tolist()))
    ok = onnx_digit == pt_digit and max_diff < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"  [{tier}] {status}  digit={onnx_digit}  max_logit_diff={max_diff:.2e}")
    return ok


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ONNX export ===")
    for tier in WEIGHTS_BY_TIER:
        print(f"\n{tier}:")
        export_tier(tier)

    print("\n=== sample inputs (EZKL format) ===")
    pixels = write_sample_inputs()

    print("\n=== ONNX vs PyTorch (golden sample) ===")
    failed = any(not verify_tier(tier, pixels) for tier in WEIGHTS_BY_TIER)

    if failed:
        print("\nVerification failed.")
        return 1

    print("\nAll tiers exported and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
