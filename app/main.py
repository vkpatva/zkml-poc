from fastapi import Depends, FastAPI, Header, HTTPException

from app.config import TIER_KEYS
from app.inference import predict
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Digit MLP API",
    description="Hosted inference with free and premium model tiers (Pattern A).",
    version="1.1.0",
)


def resolve_tier(x_api_key: str | None = Header(default=None)) -> str:
    if TIER_KEYS:
        if not x_api_key or x_api_key not in TIER_KEYS:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing X-API-Key. Use FREE_API_KEY or PREMIUM_API_KEY.",
            )
        return TIER_KEYS[x_api_key]
    return "free"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_digit(
    body: PredictRequest,
    tier: str = Depends(resolve_tier),
) -> PredictResponse:
    digit, confidence, logits = predict(body.pixels, tier=tier)
    return PredictResponse(digit=digit, confidence=confidence, logits=logits, tier=tier)
