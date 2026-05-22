from typing import Literal

from pydantic import BaseModel, Field

class SceneEvidence(BaseModel):
    frame_idx: int = Field(
        ...,
        description="Index of the video frame used as evidence.",
    )

    timestamp_sec: float = Field(
        ...,
        description="Timestamp of the frame in seconds.",
    )

    caption: str = Field(
        ...,
        description="Caption or description of the retrieved frame.",
    )

    similarity_score: float | None = Field(
        default=None,
        description="Similarity score from retrieval. Higher usually means more relevant.",
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

    
