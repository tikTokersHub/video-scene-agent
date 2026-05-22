import json
from pathlib import Path
from typing import Literal

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

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def caption_with_qwen(
    frame_path: Path,
    model,
    processor,
    max_new_tokens: int = 80,
) -> str:
    """
    Generate one short surveillance-style caption for a single frame.
    """
    image = Image.open(frame_path).convert("RGB")

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

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[1] :]
    caption = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return caption.strip()



def caption_frames(
    frames_dir: Path,
    output_file: Path,
    backend: CaptionBackend = "qwen2.5-vl",
    limit: int | None = None,
) -> dict[str, str]:
    """
    Caption all JPG frames in a directory and save results to JSONL.

    Args:
        frames_dir: Directory containing extracted .jpg frames.
        output_file: Path to output JSONL file.
        backend: Captioning backend. Use qwen2.5-vl first.
        limit: Optional number of frames to caption for testing.

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

    if backend == "qwen2.5-vl":
        model, processor = load_qwen_captioner()

        def caption_fn(frame: Path) -> str:
            return caption_with_qwen(frame, model, processor)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    with output_file.open("w", encoding="utf-8") as f:
        for frame in tqdm(frames, desc=f"Captioning with {backend}"):
            try:
                caption = caption_fn(frame)

                row = {
                    "frame": str(frame),
                    "caption": caption,
                    "backend": backend,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                results[str(frame)] = caption

            except Exception as e:
                print(f"Failed to caption {frame}: {e}")

    print(f"Saved {len(results)} captions to {output_file}")

    return results