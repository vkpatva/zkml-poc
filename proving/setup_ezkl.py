"""One-time EZKL setup per model tier and visibility profile."""

from __future__ import annotations

import argparse
import sys

from proving.ezkl_core import setup_tier
from proving.ezkl_paths import TIERS, VISIBILITIES


def main() -> int:
    parser = argparse.ArgumentParser(description="EZKL setup for DigitMLP tiers")
    parser.add_argument("--tier", choices=TIERS, default="premium")
    parser.add_argument("--visibility", choices=VISIBILITIES, default="public")
    parser.add_argument("--all", action="store_true", help="Setup all tier × visibility combinations")
    parser.add_argument("--force", action="store_true", help="Re-run setup even if artifacts exist")
    args = parser.parse_args()

    jobs = [(t, v) for t in TIERS for v in VISIBILITIES] if args.all else [(args.tier, args.visibility)]

    print("=== EZKL setup ===")
    for tier, visibility in jobs:
        print(f"\n{tier} / {visibility}:")
        manifest = setup_tier(tier, visibility, force=args.force)
        print(f"  model_id  → {manifest['model_id']}")
        print(f"  vk_hash   → {manifest['vk_hash']}")
        print(f"  scenario  → {manifest['scenario']}")

    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
