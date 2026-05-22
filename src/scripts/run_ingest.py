import argparse
from pathlib import Path

from video_agent.ingest import SceneIngester


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--captions",
        type=str,
        required=True,
        help="Path to captions JSONL file.",
    )

    parser.add_argument(
        "--video-id",
        type=str,
        required=True,
        help="Unique video ID, e.g. shanghai_01_0014.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Original video FPS.",
    )

    parser.add_argument(
        "--chroma-path",
        type=str,
        default="./chroma_db",
        help="Path to Chroma persistent DB.",
    )

    args = parser.parse_args()

    ingester = SceneIngester(chroma_path=args.chroma_path)

    ingester.ingest(
        captions_file=Path(args.captions),
        video_id=args.video_id,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()