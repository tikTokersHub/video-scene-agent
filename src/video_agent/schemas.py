from typing import Literal

from pydantic import BaseModel, Field

class SceneEvidence(BaseModel):
    """A retrieved scene that supports an answer."""
    frame_idx: int

    timestamp_sec: float

    caption: str

    similarity_score: float | None = None

    frame_path: str | None = Field(
        default=None,
        description="Local path to the retrieved frame image.",
    )
    
    video_id: str | None = Field(
        default=None,
        description="Video identifier for the retrieved frame.",
    )


class AgentAnswer(BaseModel):
    answer: str = Field(
        ...,
        description="Natural language answer to the user's question.",
    )

    classification: Literal["normal", "anomalous", "uncertain"] = Field(
        default="uncertain",
        description="Whether the scene is normal, anomalous, or uncertain.",
    )

    evidence: list[SceneEvidence] = Field(
        default_factory=list,
        description="List of retrieved frames that support the answer.",
    )

    rules_consulted: list[str] = Field(
        default_factory=list,
        description="Normal or abnormal rules checked by the agent.",
    )

    reasoning: str = Field(
        ...,
        description="Brief explanation of how the answer was reached.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1.",
    )

    
