from pydantic import BaseModel, Field, field_validator

PIXEL_COUNT = 64


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
