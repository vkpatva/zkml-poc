import torch

from app.config import WEIGHTS_BY_TIER
from app.model import DigitMLP

_models: dict[str, DigitMLP] = {}


def get_model(tier: str) -> DigitMLP:
    if tier not in WEIGHTS_BY_TIER:
        raise ValueError(f"Unknown tier: {tier}")

    if tier not in _models:
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
        _models[tier] = model

    return _models[tier]


def predict(pixels: list[float], tier: str = "free") -> tuple[int, float, list[float]]:
    """Run inference. Returns (digit, confidence, logits)."""
    x = torch.tensor([pixels], dtype=torch.float32)
    model = get_model(tier)
    with torch.no_grad():
        logits = model(x)
        digit = int(logits.argmax(dim=1).item())
        probs = torch.softmax(logits, dim=1).squeeze()
        confidence = float(probs[digit].item())
        logits_list = logits.squeeze().tolist()
    return digit, confidence, logits_list
