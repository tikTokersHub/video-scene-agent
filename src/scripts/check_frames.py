from pathlib import Path
from PIL import Image

def main() -> None:
    frames_dir = Path("data/raw/shanghaitech/01_0014")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame folder not found: {frames_dir}")

    frame_paths = sorted(
        frames_dir.glob("*.jpg")
    )

    if not frame_paths:
        raise RuntimeError(f"No image frames found in: {frames_dir}")

    first_frame = frame_paths[0]
    image = Image.open(first_frame).convert("RGB")

    print(f"Frame folder: {frames_dir}")
    print(f"Number of frames: {len(frame_paths)}")
    print(f"First frame: {first_frame}")
    print(f"Image size: {image.size}")
    print("Frame loading works.")


if __name__ == "__main__":
    main()