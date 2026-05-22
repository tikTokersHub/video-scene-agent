from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "google/siglip2-base-patch16-224"


def get_first_frame(frames_dir: Path) -> Path:
    frame_paths = sorted(
        list(frames_dir.glob("*.jpg"))
    )

    if not frame_paths:
        raise RuntimeError(f"No image frames found in: {frames_dir}")

    return frame_paths[0]


def main() -> None:
    frames_dir = Path("data/raw/shanghaitech/01_0014")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame folder not found: {frames_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    frame_path = get_first_frame(frames_dir)
    print(f"Testing frame: {frame_path}")

    image = Image.open(frame_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_outputs = model.get_image_features(**inputs)

    image_features = image_outputs.pooler_output

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    print(f"Embedding shape: {image_features.shape}")
    print(f"Embedding dtype: {image_features.dtype}")
    print(f"Embedding device: {image_features.device}")
    print(f"First 10 values: {image_features[0, :10].cpu().tolist()}")
    print("SigLIP 2 embedding works.")


if __name__ == "__main__":
    main()