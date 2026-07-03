import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

WEIGHTS_BY_TIER: dict[str, Path] = {
    "free": ARTIFACTS_DIR / "digit_mlp_free.pth",
    "premium": ARTIFACTS_DIR / "digit_mlp_premium.pth",
}

# X-API-Key header value → tier. Leave unset for open dev mode (defaults to free).
TIER_KEYS: dict[str, str] = {}
if _free := os.environ.get("FREE_API_KEY", "").strip():
    TIER_KEYS[_free] = "free"
if _premium := os.environ.get("PREMIUM_API_KEY", "").strip():
    TIER_KEYS[_premium] = "premium"
