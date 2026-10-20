"""Stable application contract, independent of the inference engine."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "motion133-v1"
MODEL_REVISION = "bf0ed2f3bf43b8a1d9def85ca8ab51bef5a10b06"

class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    seed: int = Field(default=20260703, ge=0, le=2**31 - 1)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter a sentence.")
        if len(value.split()) > 60:
            raise ValueError("Use up to 60 words for this research demo.")
        return value

class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "loading", "generating", "preparing", "complete", "failed"]
    message: str
    result: dict | None = None

# SMPL-X body joints 12..21 after the first 36 parameters are omitted by SOKE.
# This is rotation-vector ordering, not COCO's unrelated 133-keypoint format.
BODY_JOINTS = ["neck", "left_collar", "right_collar", "head", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
HAND_JOINTS = [f"{finger}{joint}" for finger in ["index", "middle", "pinky", "ring", "thumb"] for joint in [1, 2, 3]]
