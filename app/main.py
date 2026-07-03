from fastapi import Depends, FastAPI, Header, HTTPException

from app.config import TIER_KEYS
from app.inference import predict
from app.schemas import (
    PredictRequest,
    PredictResponse,
    ProveRequest,
    ProveResponse,
    VerifyRequest,
    VerifyResponse,
)
from proving.ezkl_core import prove_pixels, verify_proof_payload
from proving.ezkl_paths import is_setup_complete

app = FastAPI(
    title="Digit MLP API",
    description="Hosted inference with free/premium tiers and optional EZKL proofs.",
    version="1.2.0",
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


@app.post("/prove", response_model=ProveResponse)
def prove_digit(
    body: ProveRequest,
    tier: str = Depends(resolve_tier),
) -> ProveResponse:
    if not is_setup_complete(tier, body.visibility):
        raise HTTPException(
            status_code=503,
            detail=(
                f"EZKL not set up for tier={tier} visibility={body.visibility}. "
                f"Run: python -m proving.setup_ezkl --tier {tier} --visibility {body.visibility}"
            ),
        )
    try:
        result = prove_pixels(body.pixels, tier=tier, visibility=body.visibility)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Proving failed: {exc}") from exc

    return ProveResponse(
        digit=result["digit"],
        logits=result["logits"],
        tier=result["tier"],
        visibility=result["visibility"],
        model_id=result["model_id"],
        vk_hash=result["vk_hash"],
        scenario=result["scenario"],
        verified=result["verified"],
        proof=result["proof"],
    )


@app.post("/verify", response_model=VerifyResponse)
def verify_digit(body: VerifyRequest) -> VerifyResponse:
    if not is_setup_complete(body.tier, body.visibility):
        raise HTTPException(
            status_code=503,
            detail=(
                f"EZKL verifier artifacts missing for tier={body.tier} "
                f"visibility={body.visibility}."
            ),
        )
    try:
        result = verify_proof_payload(body.proof, body.tier, body.visibility)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Verification failed: {exc}") from exc

    return VerifyResponse(**result)
