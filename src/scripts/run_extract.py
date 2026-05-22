import argparse
from pathlib import Path

from video_agent.frame_extractor import extract_frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/test_video/",
        help="Directory to save extracted frames.",
    )

    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=8,
        help="Save one frame every N frames.",
    )

    args = parser.parse_args()

    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        sample_every_n=args.sample_every_n,
    )

if __name__ == "__main__":
    main()