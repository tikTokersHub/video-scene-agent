from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from video_agent.agent import ask
from video_agent.schemas import AgentAnswer


app = FastAPI(
    title="Video Scene Agent",
    description="Conversational video scene QA agent with frame-level evidence.",
    version="0.0.1",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    video_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "video-scene-agent",
    }


@app.post("/ask", response_model=AgentAnswer)
def ask_endpoint(request: AskRequest) -> AgentAnswer:
    try:
        return ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e