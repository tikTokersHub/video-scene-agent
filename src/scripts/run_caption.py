import argparse
from pathlib import Path

from video_agent.captioner import caption_frames


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing extracted .jpg frames.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Where to save caption JSONL file.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen2.5-vl",
        choices=["qwen2.5-vl", "cogvlm"],
        help="Captioning backend.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of frames to caption for quick testing.",
    )

    args = parser.parse_args()

    caption_frames(
        frames_dir=args.frames_dir,
        output_file=args.output_file,
        backend=args.backend,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()