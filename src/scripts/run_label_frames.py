import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


FRAMES_DIR = Path("data/raw/shanghaitech/")
OUTPUT_FILE = Path("data/labels/siglip_finetune.jsonl")

PROMPT = """You are labelling a surveillance camera frame.
In one short sentence (max 20 words), describe:
- What people are doing (walking, standing, sitting, talking, riding a bike, etc.)
- How many people
- Any notable objects (bags, bikes, vehicles)

Be specific and concrete. Do NOT speculate about intent or emotion."""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def label_frame(frame_path: Path) -> str:
    image_base64 = encode_image(frame_path)

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ]
            }
        ],
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAMES_DIR.glob("**/*.jpg"))

    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {FRAMES_DIR}")
    
    frames = frames[::5][:50]
    print(f"Found {len(frames)} frames to label.")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
         for frame_path in tqdm(frames):
             try:
                caption = label_frame(frame_path)

                row = {
                    "image": str(frame_path),
                    "text": caption,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

             except Exception as e:
                print(f"Failed on {frame_path}: {e}")

    print(f"Saved labels to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


