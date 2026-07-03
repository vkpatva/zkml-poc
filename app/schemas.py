from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal

PIXEL_COUNT = 64
VisibilityMode = Literal["public", "private"]


class PredictRequest(BaseModel):
    """8×8 digit image as 64 normalized pixels (row-major, values in [0, 1])."""

    pixels: list[float] = Field(
        ...,
        min_length=PIXEL_COUNT,
        max_length=PIXEL_COUNT,
        description="64 floats: raw pixel / 16.0",
        examples=[[0.0, 0.125, 0.9375] + [0.0] * 61],
    )

    @field_validator("pixels")
    @classmethod
    def pixels_in_range(cls, values: list[float]) -> list[float]:
        for i, v in enumerate(values):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"pixel[{i}]={v} out of range [0, 1]")
        return values


class PredictResponse(BaseModel):
    digit: int = Field(..., ge=0, le=9, description="Predicted class (argmax of logits)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of predicted digit")
    logits: list[float] = Field(..., min_length=10, max_length=10)
    tier: str = Field(..., description="Model tier used: free or premium")


class ProveRequest(BaseModel):
    """Same pixels as /predict; visibility selects the EZKL scenario profile."""

    pixels: list[float] = Field(..., min_length=PIXEL_COUNT, max_length=PIXEL_COUNT)
    visibility: VisibilityMode = Field(
        default="public",
        description="public = audit trail (input visible); private = hide input pixels",
    )

    @field_validator("pixels")
    @classmethod
    def pixels_in_range(cls, values: list[float]) -> list[float]:
        for i, v in enumerate(values):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"pixel[{i}]={v} out of range [0, 1]")
        return values


class ProveResponse(BaseModel):
    digit: int = Field(..., ge=0, le=9)
    logits: list[float] = Field(..., min_length=10, max_length=10)
    tier: str
    visibility: VisibilityMode
    model_id: str = Field(..., description="Binds proof to a specific compiled model + profile")
    vk_hash: str = Field(..., description="SHA-256 of vk.key; pin this to trust model M")
    scenario: str = Field(..., description="correct_inference_audit | private_input_proof")
    verified: bool = Field(..., description="Server-side verify immediately after proving")
    proof: dict[str, Any] = Field(..., description="Portable ZK proof (share with verifiers)")


class VerifyRequest(BaseModel):
    proof: dict[str, Any]
    tier: str = Field(default="premium", description="Model tier the proof was generated with")
    visibility: VisibilityMode = Field(default="public")


class VerifyResponse(BaseModel):
    verified: bool
    model_id: str
    vk_hash: str
    tier: str
    visibility: VisibilityMode
    scenario: str
