from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import cv2

from video_agent.captioner import CaptionBackend, caption_frames
from video_agent.frame_extractor import extract_frames
from video_agent.ingest import SceneIngester
from video_agent.rules import (
    build_rule_collection,
    generate_rules_from_captions,
    generate_rules_with_frames,
    parse_rule_text,
)


def make_video_id(video_path: Path) -> str:
    safe_stem = "".join(
        character if character.isalnum() else "_"
        for character in video_path.stem.lower()
    ).strip("_")
    return safe_stem or "uploaded_video"


def get_video_fps(video_path: Path, fallback: float = 24.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return fallback

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else fallback


def prepare_uploaded_video(
    video_path: Path,
    video_id: str | None = None,
    sample_every_n: int = 8,
    captioner_backend: CaptionBackend = "qwen2.5-vl",
    normal_rules_text: str | None = None,
    abnormal_rules_text: str | None = None,
    generate_video_rules: bool = True,
    rule_context_query: str | None = None,
    chroma_path: str = "./chroma_db",
    data_dir: Path = Path("data/uploads"),
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Extract, caption, ingest, and rule-index an uploaded video.

    Custom rules override defaults for this video. If no custom rules are
    provided, the thesis defaults are indexed. Generated rules can be added
    from the uploaded video's captions to make anomaly checks context-aware.
    """
    source_video = Path(video_path)
    if not source_video.exists():
        raise FileNotFoundError(f"Video not found: {source_video}")

    def emit_progress(
        stage: str,
        progress: float,
        message: str,
        **extra,
    ) -> None:
        if progress_callback is None:
            return

        progress_callback(
            {
                "stage": stage,
                "progress": progress,
                "message": message,
                **extra,
            }
        )

    resolved_video_id = video_id or make_video_id(source_video)
    work_dir = data_dir / resolved_video_id
    frames_dir = work_dir / "frames"
    captions_file = work_dir / "captions.jsonl"
    stored_video = work_dir / source_video.name

    work_dir.mkdir(parents=True, exist_ok=True)
    emit_progress("copy", 5, "Storing uploaded video")
    if source_video.resolve() != stored_video.resolve():
        shutil.copy2(source_video, stored_video)

    emit_progress("extract", 10, "Extracting frames")
    frames = extract_frames(
        video_path=stored_video,
        output_dir=frames_dir,
        sample_every_n=sample_every_n,
    )
    emit_progress(
        "extract",
        20,
        f"Extracted {len(frames)} frames",
        frames_extracted=len(frames),
    )

    if not captions_file.exists():
        def caption_progress(payload: dict) -> None:
            caption_stage_progress = float(payload.get("progress", 0))
            emit_progress(
                "caption",
                20 + (caption_stage_progress * 0.5),
                payload.get("message", "Captioning frames"),
                stage_progress=caption_stage_progress,
                current=payload.get("current"),
                total=payload.get("total"),
                status=payload.get("status", "running"),
            )

        caption_frames(
            frames_dir=frames_dir,
            output_file=captions_file,
            backend=captioner_backend,
            batch_size=4,
            progress_callback=caption_progress,
        )
    else:
        emit_progress(
            "caption",
            70,
            "Using existing captions",
            stage_progress=100,
            current=len(frames),
            total=len(frames),
            status="complete",
        )

    fps = get_video_fps(stored_video)
    emit_progress("ingest", 72, "Indexing captioned frames")
    ingester = SceneIngester(chroma_path=chroma_path)
    ingester.ingest(
        captions_file=captions_file,
        video_id=resolved_video_id,
        fps=fps,
        captioner_backend=captioner_backend,
    )
    emit_progress("ingest", 84, "Scene index ready")

    normal_rules = parse_rule_text(normal_rules_text)
    abnormal_rules = parse_rule_text(abnormal_rules_text)
    using_custom_rules = bool(normal_rules or abnormal_rules)

    generated_normal: list[str] = []
    generated_abnormal: list[str] = []
    rules_source = "none"

    if using_custom_rules:
        emit_progress("rules", 86, "Indexing custom rules")
        selected_normal_rules = normal_rules
        selected_abnormal_rules = abnormal_rules
        rules_source = "user"

    elif generate_video_rules:
        emit_progress("rules", 86, "Generating behavior rules")
        try:
            generated_normal, generated_abnormal = generate_rules_with_frames(
                captions_file=captions_file,
                context_query=rule_context_query,
            )
            rules_source = "vision_generated"

        except Exception as e:
            print(f"Vision rule generation failed, falling back to caption keywords: {e}")

            generated_normal, generated_abnormal = generate_rules_from_captions(
                captions_file=captions_file,
                context_query=rule_context_query,
            )
            rules_source = "caption_keyword_fallback"

        selected_normal_rules = generated_normal
        selected_abnormal_rules = generated_abnormal

    else:
        selected_normal_rules = []
        selected_abnormal_rules = []
        rules_source = "none"

    if selected_normal_rules or selected_abnormal_rules:
        build_rule_collection(
            chroma_path=chroma_path,
            normal_rules=selected_normal_rules,
            abnormal_rules=selected_abnormal_rules,
            video_id=resolved_video_id,
            source=rules_source,
        )    
    emit_progress("rules", 96, "Rules ready", rules_source=rules_source)

    return {
        "video_id": resolved_video_id,
        "stored_video": str(stored_video),
        "frames_dir": str(frames_dir),
        "captions_file": str(captions_file),
        "frames_extracted": len(frames),
        "rules_source": rules_source,
        "generated_normal_rules": generated_normal,
        "generated_abnormal_rules": generated_abnormal,
    }
