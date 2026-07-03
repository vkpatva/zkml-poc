"""EZKL proof generation and verification (requires local setup)."""

import json
import sys
import unittest

from app.config import PROJECT_ROOT
from app.inference import predict
from proving.ezkl_core import compare_logits, prove_pixels, verify_proof_payload
from proving.ezkl_paths import is_setup_complete

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "test_input.json"


@unittest.skipUnless(
    is_setup_complete("premium", "public"),
    "EZKL not set up. Run: python -m proving.setup_ezkl --tier premium --visibility public",
)
class TestEzklPremiumPublic(unittest.TestCase):
    """Scenario 1 + 2: correct inference on a specific model (premium, public audit)."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.pixels = json.load(f)["input_data"][0]
        _, _, cls.pt_logits = predict(cls.pixels, tier="premium")

    def test_prove_and_verify(self):
        result = prove_pixels(self.pixels, tier="premium", visibility="public")
        self.assertTrue(result["verified"])
        self.assertEqual(result["scenario"], "correct_inference_audit")
        self.assertEqual(result["tier"], "premium")
        self.assertTrue(result["model_id"].startswith("digit_mlp_premium_"))

        parity = compare_logits(self.pt_logits, result["logits"])
        self.assertTrue(parity["digits_match"])
        self.assertTrue(parity["within_tolerance"])

        check = verify_proof_payload(result["proof"], "premium", "public")
        self.assertTrue(check["verified"])
        self.assertEqual(check["vk_hash"], result["vk_hash"])


@unittest.skipUnless(
    is_setup_complete("premium", "private"),
    "EZKL private profile not set up. Run: python -m proving.setup_ezkl --tier premium --visibility private",
)
class TestEzklPremiumPrivate(unittest.TestCase):
    """Scenario 3: private input — proof verifies without public pixels."""

    def test_private_input_proof(self):
        with open(FIXTURE) as f:
            pixels = json.load(f)["input_data"][0]
        result = prove_pixels(pixels, tier="premium", visibility="private")
        self.assertTrue(result["verified"])
        self.assertEqual(result["scenario"], "private_input_proof")
        check = verify_proof_payload(result["proof"], "premium", "private")
        self.assertTrue(check["verified"])


@unittest.skipUnless(
    is_setup_complete("free", "public") and is_setup_complete("premium", "public"),
    "Both tiers need setup for binding test. Run: python -m proving.setup_ezkl --all",
)
class TestEzklTierBinding(unittest.TestCase):
    """Scenario 2: free and premium produce different vk_hash values."""

    def test_tier_vk_hashes_differ(self):
        from proving.ezkl_paths import load_manifest

        free = load_manifest("free", "public")
        premium = load_manifest("premium", "public")
        self.assertNotEqual(free["vk_hash"], premium["vk_hash"])
        self.assertNotEqual(free["model_id"], premium["model_id"])


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
