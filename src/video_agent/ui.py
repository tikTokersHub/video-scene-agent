import os

import gradio as gr

from video_agent.agent import ask
from video_agent.pipeline import prepare_uploaded_video


def resolve_frame_path(evidence_item):
    """
    Return a usable local frame path.

    Priority:
    1. Use frame_path from structured evidence.
    2. Fallback to old ShanghaiTech demo path.
    """
    frame_path = getattr(evidence_item, "frame_path", None)

    if frame_path and os.path.exists(frame_path):
        return frame_path

    frame_idx = getattr(evidence_item, "frame_idx", None)

    if frame_idx is not None:
        fallback_path = f"data/raw/shanghaitech/01_0014/{frame_idx:03d}.jpg"

        if os.path.exists(fallback_path):
            return fallback_path

    return None


def process_video_handler(
    video_file,
    video_id: str | None,
    normal_rules: str | None,
    abnormal_rules: str | None,
    generate_video_rules: bool,
    rule_context_query: str | None,
    sample_every_n: int,
):
    """Prepare an uploaded video for question answering."""
    if not video_file:
        return (
            "Upload a video before preparing it.",
            video_id or "",
            video_id or "",
        )

    try:
        # gr.File(type="filepath") returns a string path.
        # Older Gradio versions may return a dict.
        if isinstance(video_file, dict):
            video_path = video_file.get("path") or video_file.get("name")
        else:
            video_path = video_file

        if not video_path:
            return (
                "Could not resolve uploaded video path.",
                video_id or "",
                video_id or "",
            )

        result = prepare_uploaded_video(
            video_path=video_path,
            video_id=video_id or None,
            sample_every_n=int(sample_every_n),
            normal_rules_text=normal_rules,
            abnormal_rules_text=abnormal_rules,
            generate_video_rules=generate_video_rules,
            rule_context_query=rule_context_query,
        )

        generated_normal_md = "\n".join(
            f"- {rule}" for rule in result["generated_normal_rules"]
        )

        generated_abnormal_md = "\n".join(
            f"- {rule}" for rule in result["generated_abnormal_rules"]
        )

        if not generated_normal_md:
            generated_normal_md = "_No generated normal rules._"

        if not generated_abnormal_md:
            generated_abnormal_md = "_No generated abnormal rules._"

        status = f"""
## Prepared video `{result["video_id"]}`

- **Frames extracted:** {result["frames_extracted"]}
- **Captions file:** `{result["captions_file"]}`
- **Rule source:** `{result["rules_source"]}`

## Generated normal rules

{generated_normal_md}

## Generated abnormal rules

{generated_abnormal_md}
"""

        return (
            status,
            result["video_id"],
            result["video_id"],
        )

    except Exception as e:
        return (
            f"Error preparing video: {e}",
            video_id or "",
            video_id or "",
        )


def query_handler(question: str, video_id: str | None):
    """Handle a user question from the Gradio UI."""
    if not question or not question.strip():
        return (
            "Please enter a question.",
            "",
            "",
            "No evidence yet.",
            [],
        )

    try:
        answer = ask(question.strip(), video_id=video_id or None)

        evidence_lines = []
        gallery_items = []

        for e in answer.evidence[:6]:
            similarity = (
                f"{e.similarity_score:.3f}"
                if e.similarity_score is not None
                else "N/A"
            )

            evidence_lines.append(
                f"""
### Frame {e.frame_idx} — {e.timestamp_sec:.2f}s

{e.caption}

Similarity: `{similarity}`
"""
            )

            frame_path = resolve_frame_path(e)
            if frame_path is not None:
                gallery_items.append(
                    (
                        frame_path,
                        f"Frame {e.frame_idx} — {e.timestamp_sec:.2f}s",
                    )
                )

        evidence_md = "\n---\n".join(evidence_lines)

        if not evidence_md:
            evidence_md = "No frame-level evidence returned."

        rules_md = "\n".join(
            f"- {rule}" for rule in answer.rules_consulted
        )

        if not rules_md:
            rules_md = "No rules consulted."

        reasoning = answer.reasoning or "No reasoning returned."

        details_md = f"""
## Evidence

{evidence_md}

## Rules consulted

{rules_md}

## Reasoning

{reasoning}
"""

        return (
            answer.answer,
            answer.classification,
            f"{answer.confidence:.0%}",
            details_md,
            gallery_items,
        )

    except Exception as e:
        return (
            f"Error: {e}",
            "error",
            "",
            "",
            [],
        )


with gr.Blocks(
    title="Video Scene Agent",
    theme=gr.themes.Soft(),
) as demo:
    active_video_id = gr.State("")

    gr.Markdown(
        """
# Conversational Video Scene Agent

Upload a surveillance video, prepare it for question answering, and ask natural-language questions about what happened.

The system extracts frames, captions the video, generates behaviour rules from representative frames if no custom rules are provided, retrieves frame-level evidence, checks relevant rules, and returns a structured answer.
"""
    )

    with gr.Accordion("1. Upload video and prepare rules", open=True):
        with gr.Row():
            video_upload = gr.File(
                label="Upload video",
                file_types=[".mp4", ".avi", ".mov", ".mkv"],
                type="filepath",
                scale=3,
            )

            with gr.Column(scale=2):
                video_id_input = gr.Textbox(
                    label="Video ID",
                    placeholder="Optional. Leave blank to use the file name.",
                )

                sample_every_n = gr.Number(
                    label="Sample every N frames",
                    value=8,
                    precision=0,
                    minimum=1,
                )

                generate_rules = gr.Checkbox(
                    label="Auto-generate behaviour rules from uploaded video",
                    value=True,
                )

                active_video_display = gr.Textbox(
                    label="Active video ID",
                    value="",
                    interactive=False,
                )

        gr.Markdown(
            """
### Optional rule controls

Leave custom rules empty to let the system generate behaviour rules from representative frames and captions.

If you provide custom rules, they will override auto-generated rules for this uploaded video.
"""
        )

        with gr.Row():
            normal_rules = gr.Textbox(
                label="Custom normal rules",
                placeholder=(
                    "Optional. One rule per line.\n"
                    "Example:\n"
                    "People walking through the pathway\n"
                    "People sitting on benches"
                ),
                lines=5,
            )

            abnormal_rules = gr.Textbox(
                label="Custom abnormal rules",
                placeholder=(
                    "Optional. One rule per line.\n"
                    "Example:\n"
                    "Riding a bicycle in a pedestrian area\n"
                    "Fighting or pushing people"
                ),
                lines=5,
            )

        rule_context_query = gr.Textbox(
            label="Scene policy / anomaly context",
            placeholder=(
                "Optional, e.g. pedestrian walkway; bicycles are not allowed; "
                "running is suspicious; loitering near the entrance is abnormal"
            ),
            lines=2,
        )

        prepare_button = gr.Button(
            "Prepare Video",
            variant="primary",
        )

        prepare_status = gr.Markdown(
            "No uploaded video prepared in this session."
        )

    gr.Markdown("## 2. Ask questions about the active video")

    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="e.g. Was anyone riding a bicycle in the video?",
            lines=2,
            scale=5,
        )

        ask_button = gr.Button(
            "Ask",
            variant="primary",
            scale=1,
        )

    gr.Examples(
        examples=[
            "Was anyone riding a bicycle in the video?",
            "What happened around 7 seconds into the video?",
            "Were two people sitting near the pathway at the start of the video?",
            "Was anyone fighting or pushing another person?",
            "Was the scene normal before the bicycle appeared?",
        ],
        inputs=question,
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.Markdown("## Agent Answer")

            answer_out = gr.Textbox(
                label="Answer",
                lines=6,
                interactive=False,
            )

            with gr.Row():
                classification_out = gr.Textbox(
                    label="Classification",
                    interactive=False,
                )

                confidence_out = gr.Textbox(
                    label="Confidence",
                    interactive=False,
                )

            evidence_out = gr.Markdown(
                label="Evidence Details",
                value="Evidence will appear here after you ask a question.",
            )

        with gr.Column(scale=4):
            gr.Markdown("## Retrieved Evidence Frames")

            gallery_out = gr.Gallery(
                label="Frames",
                show_label=False,
                columns=2,
                height=520,
                object_fit="contain",
            )

    prepare_button.click(
        fn=process_video_handler,
        inputs=[
            video_upload,
            video_id_input,
            normal_rules,
            abnormal_rules,
            generate_rules,
            rule_context_query,
            sample_every_n,
        ],
        outputs=[
            prepare_status,
            active_video_id,
            active_video_display,
        ],
    )

    ask_button.click(
        fn=query_handler,
        inputs=[
            question,
            active_video_id,
        ],
        outputs=[
            answer_out,
            classification_out,
            confidence_out,
            evidence_out,
            gallery_out,
        ],
    )

    question.submit(
        fn=query_handler,
        inputs=[
            question,
            active_video_id,
        ],
        outputs=[
            answer_out,
            classification_out,
            confidence_out,
            evidence_out,
            gallery_out,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )