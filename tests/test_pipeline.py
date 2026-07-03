"""Golden-sample sanity check for both model tiers."""

import json
import sys

from app.config import PROJECT_ROOT
from app.inference import predict

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def main() -> int:
    with open(FIXTURES / "test_input.json") as f:
        test_input = json.load(f)
    with open(FIXTURES / "expected_output.json") as f:
        expected = json.load(f)

    pixels = test_input["input_data"][0]
    true_label = expected["true_label"]
    failed = False

    for tier in ("free", "premium"):
        digit, confidence, logits = predict(pixels, tier=tier)
        ok = digit == true_label
        status = "PASS" if ok else "FAIL"
        print(f"[{tier}] {status}")
        print(f"  Predicted: {digit}  Confidence: {confidence:.4f}  True: {true_label}")
        print(f"  Logits: {[round(v, 4) for v in logits]}")
        print()
        if not ok:
            failed = True

    if failed:
        print("Some tiers failed — re-run: python -m training.train")
        return 1

    print("All tiers PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
