import json
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


CaptionBackend = Literal["qwen2.5-vl", "cogvlm"]


def load_qwen_captioner(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
):
    """
    Load Qwen2.5-VL captioning model.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def caption_batch_with_qwen(
    frame_paths: list[Path],
    model,
    processor,
    max_new_tokens: int = 40,
) -> list[str]:
    """
    Generate short surveillance-style captions for a batch of frames.
    """
    images = [Image.open(frame).convert("RGB") for frame in frame_paths]

    prompt_texts: list[str] = []

    for image in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Describe this surveillance camera frame in one short sentence. "
                            "Focus on what people are doing, how many people are visible, "
                            "and any notable objects such as bags, bikes, vehicles, or benches. "
                            "Do not speculate about emotions or intent."
                        ),
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_texts.append(text)

    inputs = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # Move tensors to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    captions: list[str] = []

    # Each sample may have a different prompt length because of padding
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for i, prompt_len in enumerate(prompt_lengths):
        generated_ids = output_ids[i][prompt_len:]
        caption = processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        captions.append(caption.strip())

    return captions


def chunk_list(items: list[Path], batch_size: int) -> list[list[Path]]:
    """
    Split a list into chunks of size batch_size.
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def caption_frames(
    frames_dir: Path,
    output_file: Path,
    backend: CaptionBackend = "qwen2.5-vl",
    limit: int | None = None,
    batch_size: int | None = 1,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, str]:
    """
    Caption all JPG frames in a directory and save results to JSONL.

    Args:
        frames_dir: Directory containing extracted .jpg frames.
        output_file: Path to output JSONL file.
        backend: Captioning backend.
        limit: Optional number of frames to caption for testing.
        batch_size: Number of images to caption at once.

    Returns:
        Dictionary mapping frame path string to caption.
    """
    frames = sorted(frames_dir.glob("*.jpg"))

    if limit is not None:
        frames = frames[:limit]

    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    total_frames = len(frames)

    def emit_progress(
        current: int,
        message: str,
        status: str = "running",
    ) -> None:
        if progress_callback is None:
            return

        progress_callback(
            {
                "stage": "caption",
                "status": status,
                "progress": (current / total_frames) * 100,
                "current": current,
                "total": total_frames,
                "message": message,
            }
        )

    emit_progress(0, f"Loading {backend} caption model")

    if backend == "qwen2.5-vl":
        model, processor = load_qwen_captioner()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    frame_batches = chunk_list(frames, batch_size)
    processed_frames = 0

    with output_file.open("w", encoding="utf-8") as f:
        for batch in tqdm(frame_batches, desc=f"Captioning with {backend}"):
            try:
                captions = caption_batch_with_qwen(
                    batch,
                    model,
                    processor,
                )

                for frame, caption in zip(batch, captions):
                    row = {
                        "frame": str(frame),
                        "caption": caption,
                        "backend": backend,
                    }

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    results[str(frame)] = caption

            except Exception as e:
                print(f"Failed to caption batch starting at {batch[0]}: {e}")

            processed_frames = min(processed_frames + len(batch), total_frames)
            emit_progress(
                processed_frames,
                f"Captioned {processed_frames} of {total_frames} frames",
            )

    print(f"Saved {len(results)} captions to {output_file}")
    emit_progress(
        total_frames,
        f"Captioned {total_frames} of {total_frames} frames",
        status="complete",
    )

    return results
