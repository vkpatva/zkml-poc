"""Generate and optionally verify an EZKL proof for one input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import PROJECT_ROOT
from app.inference import predict
from proving.ezkl_core import compare_logits, prove_pixels, verify_proof_payload
from proving.ezkl_paths import artifact_paths

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "test_input.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove DigitMLP inference with EZKL")
    parser.add_argument("--tier", default="premium", choices=["free", "premium"])
    parser.add_argument("--visibility", default="public", choices=["public", "private"])
    parser.add_argument("--input", type=Path, help="JSON file with input_data")
    parser.add_argument("--verify-only", action="store_true", help="Verify proof.json on disk")
    args = parser.parse_args()

    if args.verify_only:
        proof = json.loads(artifact_paths(args.tier, args.visibility)["proof"].read_text())
        result = verify_proof_payload(proof, args.tier, args.visibility)
        print(json.dumps(result, indent=2))
        return 0 if result["verified"] else 1

    input_path = args.input or FIXTURE
    with open(input_path) as f:
        pixels = json.load(f)["input_data"][0]

    print(f"=== prove ({args.tier}, {args.visibility}) ===")
    _, _, pt_logits = predict(pixels, tier=args.tier)
    result = prove_pixels(pixels, tier=args.tier, visibility=args.visibility)
    parity = compare_logits(pt_logits, result["logits"])

    print(f"  model_id   : {result['model_id']}")
    print(f"  vk_hash    : {result['vk_hash']}")
    print(f"  digit      : {result['digit']}")
    print(f"  verified   : {result['verified']}")
    print(f"  parity     : {parity}")

    if not result["verified"] or not parity["digits_match"]:
        print("\nFAIL")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
