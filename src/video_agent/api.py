import hashlib
import shutil
import threading
import uuid
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from video_agent.agent import ask
from video_agent.pipeline import prepare_uploaded_video
from video_agent.rules import parse_rule_text
from video_agent.schemas import AgentAnswer
from video_agent.tools import get_chroma_client, get_rule_encoder
from video_agent.ui import demo as gradio_demo


app = FastAPI(title="Video Scene Agent")


BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
UPLOAD_STAGING_DIR = BASE_DIR / "data" / "api_uploads"
DATA_UPLOADS_DIR = BASE_DIR / "data" / "uploads"
PREPARE_JOBS: dict[str, dict] = {}
PREPARE_JOBS_LOCK = threading.Lock()


class AskRequest(BaseModel):
    question: str
    video_id: str | None = None


class RuleEditRequest(BaseModel):
    normal_rules: list[str] = Field(default_factory=list)
    abnormal_rules: list[str] = Field(default_factory=list)


def safe_name(value: str, fallback: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._-")
    return cleaned or fallback


def clean_rule_inputs(rules: list[str]) -> list[str]:
    parsed = parse_rule_text("\n".join(str(rule) for rule in rules if rule))
    return list(dict.fromkeys(parsed))


def get_rules_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="rules",
        metadata={"hnsw:space": "cosine"},
    )


def read_video_rules(video_id: str) -> dict:
    collection = get_rules_collection()
    result = collection.get(
        where={"video_id": {"$eq": video_id}},
    )

    rows = []
    seen = set()

    for item_id, rule, metadata in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        metadata = metadata or {}
        rule_type = metadata.get("rule_type")

        if rule_type not in {"normal", "abnormal"} or not rule:
            continue

        source = metadata.get("source") or "unknown"
        dedupe_key = (rule_type, source, rule)

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        rows.append(
            {
                "id": item_id,
                "rule": rule,
                "rule_type": rule_type,
                "source": source,
                "video_id": metadata.get("video_id") or video_id,
            }
        )

    source_rank = {
        "user": 0,
        "vision_generated": 1,
        "caption_keyword_fallback": 2,
        "generated": 3,
        "unknown": 9,
    }
    rows.sort(
        key=lambda item: (
            source_rank.get(item["source"], 8),
            item["rule_type"],
            item["rule"].lower(),
        )
    )

    normal_rules = [item for item in rows if item["rule_type"] == "normal"]
    abnormal_rules = [item for item in rows if item["rule_type"] == "abnormal"]

    return {
        "video_id": video_id,
        "normal_rules": normal_rules,
        "abnormal_rules": abnormal_rules,
        "total_rules": len(rows),
    }


def append_video_rules(
    video_id: str,
    normal_rules: list[str],
    abnormal_rules: list[str],
) -> int:
    rules = (
        [(rule, "normal") for rule in normal_rules]
        + [(rule, "abnormal") for rule in abnormal_rules]
    )

    if not rules:
        return 0

    collection = get_rules_collection()
    encoder = get_rule_encoder()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for rule, rule_type in rules:
        digest = hashlib.sha1(
            f"{video_id}|{rule_type}|{rule}".encode("utf-8")
        ).hexdigest()[:14]
        item_id = f"rule_{video_id}_user_{rule_type}_{digest}"

        ids.append(item_id)
        embeddings.append(
            encoder.encode(
                rule,
                normalize_embeddings=True,
            ).tolist()
        )
        documents.append(rule)
        metadatas.append(
            {
                "rule_type": rule_type,
                "source": "user",
                "video_id": video_id,
            }
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


def delete_video_rule(video_id: str, rule_id: str) -> None:
    collection = get_rules_collection()
    result = collection.get(ids=[rule_id])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []

    if not ids:
        raise HTTPException(status_code=404, detail="Rule not found.")

    metadata = metadatas[0] if metadatas else {}

    if metadata.get("video_id") != video_id:
        raise HTTPException(status_code=404, detail="Rule not found for this video.")

    collection.delete(ids=[rule_id])


def update_prepare_job(job_id: str, **updates) -> None:
    with PREPARE_JOBS_LOCK:
        job = PREPARE_JOBS.setdefault(job_id, {})
        job.update(updates)


def get_prepare_job(job_id: str) -> dict | None:
    with PREPARE_JOBS_LOCK:
        job = PREPARE_JOBS.get(job_id)
        return dict(job) if job is not None else None


def run_prepare_job(
    job_id: str,
    staged_path: Path,
    video_id: str | None,
    sample_every_n: int,
    normal_rules: str | None,
    abnormal_rules: str | None,
    generate_video_rules: bool,
    rule_context_query: str | None,
) -> None:
    def progress_callback(payload: dict) -> None:
        update_prepare_job(
            job_id,
            status="running",
            stage=payload.get("stage", "prepare"),
            progress=round(float(payload.get("progress", 0)), 1),
            stage_progress=(
                round(float(payload["stage_progress"]), 1)
                if payload.get("stage_progress") is not None
                else None
            ),
            current=payload.get("current"),
            total=payload.get("total"),
            message=payload.get("message", "Preparing video"),
            metadata={
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "stage",
                    "progress",
                    "stage_progress",
                    "current",
                    "total",
                    "message",
                    "status",
                }
            },
        )

    try:
        update_prepare_job(
            job_id,
            status="running",
            stage="queued",
            progress=0,
            stage_progress=None,
            current=None,
            total=None,
            message="Starting preparation",
        )

        result = prepare_uploaded_video(
            video_path=staged_path,
            video_id=video_id,
            sample_every_n=max(1, int(sample_every_n)),
            normal_rules_text=normal_rules,
            abnormal_rules_text=abnormal_rules,
            generate_video_rules=generate_video_rules,
            rule_context_query=rule_context_query,
            progress_callback=progress_callback,
        )

        update_prepare_job(
            job_id,
            status="complete",
            stage="complete",
            progress=100,
            stage_progress=100,
            current=None,
            total=None,
            message="Video indexed",
            result=result,
        )

    except Exception as e:
        update_prepare_job(
            job_id,
            status="failed",
            stage="failed",
            progress=0,
            message=str(e),
            error=str(e),
        )


@app.get("/")
def landing_page():
    """Serve the static landing/front page."""
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "error": "static/index.html not found",
            "expected_path": str(index_file),
        }

    return FileResponse(index_file)


@app.get("/upload")
def upload_page():
    """Keep legacy upload links on the static experience."""
    return RedirectResponse(url="/#demo")


@app.post("/api/prepare/start")
def prepare_start_endpoint(
    video_file: UploadFile = File(...),
    video_id: str | None = Form(default=None),
    sample_every_n: int = Form(default=8),
    generate_video_rules: bool = Form(default=True),
    normal_rules: str | None = Form(default=None),
    abnormal_rules: str | None = Form(default=None),
    rule_context_query: str | None = Form(default=None),
):
    """
    Start an async preparation job for the static frontend.

    The upload request returns after the file is stored. The page then polls
    /api/prepare/{job_id}/progress for real extraction, captioning, indexing,
    and rules progress.
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="Upload a video file.")

    suffix = Path(video_file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(
            status_code=400,
            detail="Supported video formats: mp4, avi, mov, mkv.",
        )

    job_id = uuid.uuid4().hex
    UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged_name = f"{job_id}_{safe_name(video_file.filename, f'upload{suffix}')}"
    staged_path = UPLOAD_STAGING_DIR / staged_name

    update_prepare_job(
        job_id,
        status="uploading",
        stage="upload",
        progress=0,
        stage_progress=None,
        current=None,
        total=None,
        message="Uploading file",
        result=None,
        error=None,
    )

    try:
        with staged_path.open("wb") as destination:
            shutil.copyfileobj(video_file.file, destination)
    except Exception as e:
        update_prepare_job(
            job_id,
            status="failed",
            stage="upload",
            progress=0,
            message=str(e),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

    update_prepare_job(
        job_id,
        status="queued",
        stage="queued",
        progress=0,
        message="Upload complete. Waiting to prepare",
    )

    worker = threading.Thread(
        target=run_prepare_job,
        kwargs={
            "job_id": job_id,
            "staged_path": staged_path,
            "video_id": safe_name(video_id, "") if video_id else None,
            "sample_every_n": sample_every_n,
            "normal_rules": normal_rules,
            "abnormal_rules": abnormal_rules,
            "generate_video_rules": generate_video_rules,
            "rule_context_query": rule_context_query,
        },
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id}


@app.get("/api/prepare/{job_id}/progress")
def prepare_progress_endpoint(job_id: str):
    job = get_prepare_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Preparation job not found")

    return {"job_id": job_id, **job}


@app.post("/ask", response_model=AgentAnswer)
def ask_endpoint(request: AskRequest):
    """
    API endpoint used by the static frontend.

    Expected JSON:
    {
        "question": "Was anyone walking on the grass?",
        "video_id": "optional_video_id"
    }
    """
    try:
        return ask(
            question=request.question,
            video_id=request.video_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/videos/{video_id}/rules")
def video_rules_endpoint(video_id: str):
    safe_video_id = safe_name(video_id, "")

    if not safe_video_id:
        raise HTTPException(status_code=400, detail="A valid video_id is required.")

    return read_video_rules(safe_video_id)


@app.post("/api/videos/{video_id}/rules")
def add_video_rules_endpoint(video_id: str, request: RuleEditRequest):
    safe_video_id = safe_name(video_id, "")

    if not safe_video_id:
        raise HTTPException(status_code=400, detail="A valid video_id is required.")

    normal_rules = clean_rule_inputs(request.normal_rules)
    abnormal_rules = clean_rule_inputs(request.abnormal_rules)

    if not normal_rules and not abnormal_rules:
        raise HTTPException(
            status_code=400,
            detail="Add at least one normal or abnormal rule.",
        )

    inserted_count = append_video_rules(
        video_id=safe_video_id,
        normal_rules=normal_rules,
        abnormal_rules=abnormal_rules,
    )
    response = read_video_rules(safe_video_id)
    response["inserted_rules"] = inserted_count
    return response


@app.delete("/api/videos/{video_id}/rules/{rule_id}")
def delete_video_rule_endpoint(video_id: str, rule_id: str):
    safe_video_id = safe_name(video_id, "")
    safe_rule_id = safe_name(rule_id, "")

    if not safe_video_id or not safe_rule_id:
        raise HTTPException(status_code=400, detail="A valid rule id is required.")

    delete_video_rule(
        video_id=safe_video_id,
        rule_id=safe_rule_id,
    )
    response = read_video_rules(safe_video_id)
    response["deleted_rule_id"] = safe_rule_id
    return response


@app.get("/frames/{video_id}/{frame_name}")
def frame_file(video_id: str, frame_name: str):
    """Serve extracted evidence frames without exposing arbitrary file paths."""
    safe_video_id = safe_name(video_id, "")
    safe_frame_name = safe_name(frame_name, "")

    if not safe_video_id or safe_frame_name != frame_name:
        raise HTTPException(status_code=404, detail="Frame not found")

    frame_path = DATA_UPLOADS_DIR / safe_video_id / "frames" / safe_frame_name

    if not frame_path.exists() or frame_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(frame_path)


@app.get("/health")
def health():
    return {"status": "ok"}


# Static files used by index.html
if (STATIC_DIR / "css").exists():
    app.mount(
        "/css",
        StaticFiles(directory=STATIC_DIR / "css"),
        name="css",
    )

if (STATIC_DIR / "js").exists():
    app.mount(
        "/js",
        StaticFiles(directory=STATIC_DIR / "js"),
        name="js",
    )

if (STATIC_DIR / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=STATIC_DIR / "assets"),
        name="assets",
    )

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR, html=True),
        name="static",
    )


# Mount the existing Gradio app under /app
app = gr.mount_gradio_app(
    app,
    gradio_demo,
    path="/app",
)
