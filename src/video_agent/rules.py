from __future__ import annotations

from dotenv import load_dotenv
import json
import random
import re
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

class GeneratedRules(BaseModel):
    normal_rules: list[str] = Field(default_factory=list)
    abnormal_rules: list[str] = Field(default_factory=list)


def sample_3_spread_out_rows(rows: list[dict], min_gap: int = 5) -> list[dict]:
    n = len(rows)
    if n <= 3:
        return rows
    valid_indices = list(range(n))
    for _ in range(1000):
        sampled_indices = sorted(random.sample(valid_indices, 3))
        if (
            sampled_indices[1] - sampled_indices[0] >= min_gap
            and sampled_indices[2] - sampled_indices[1] >= min_gap
        ):
            return [rows[i] for i in sampled_indices]
    step = n // 3
    fallback_indices = [0, step, min(2 * step, n - 1)]
    return [rows[i] for i in fallback_indices]

def _normalise_rule(rule: str) -> str:
    return " ".join(rule.strip().strip("-*").split())


def parse_rule_text(rule_text: str | None) -> list[str]:
    """Parse newline, bullet, or semicolon separated rules from UI/API input."""
    if not rule_text:
        return []

    candidates = re.split(r"[\n;]+", rule_text)
    rules = [_normalise_rule(candidate) for candidate in candidates]
    return [rule for rule in rules if rule]


def load_caption_rows(captions_file: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in captions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cleaned_rows = []
    for row in rows:
        image_path = row.get("image") or row.get("frame") or row.get("frame_path")
        caption = row.get("text") or row.get("caption")
        if image_path and caption:
            cleaned_rows.append(
                {
                    "image": image_path,
                    "caption": caption,
                }
            )
    return cleaned_rows

def encode_image_base64(image_path: Path) -> str:
    """Encode an image file as base64 for OpenAI vision input."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")

def image_to_data_url(image_path: Path) -> str:
    encoded = encode_image_base64(image_path)
    return f"data:image/jpeg;base64,{encoded}"

def generate_rules_with_frames(
    captions_file: Path,
    context_query: str | None = None,
    n_frames: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Generate video-specific normal and abnormal rules using representative frames
    plus their captions.

    This is used when the user does not manually provide rules.
    """
    rows = load_caption_rows(captions_file)

    if not rows:
        return [], []

    sampled_rows = sample_3_spread_out_rows(rows, min_gap=10)

    # If later you want more than 3 frames, this keeps the function flexible.
    sampled_rows = sampled_rows[:n_frames]

    context = context_query or "No extra scene policy was provided."

    caption_text = "\n".join(
        f"- Frame image: {row['image']}\n  Caption: {row['caption']}"
        for row in sampled_rows
    )

    content = [
        {
            "type": "text",
            "text": f"""
You are generating behaviour rules for a surveillance video question-answering system.

The system needs two rule lists:

1. normal_rules:
Behaviours that appear normal or acceptable in this video context.

2. abnormal_rules:
Behaviours that should be treated as anomalous or suspicious.

Use BOTH the representative video frames and their captions.

Important rules:
- Do not just describe the images.
- Generate reusable behaviour rules.
- Do not invent events that are not visible.
- If the scene context says something is not allowed, include it as abnormal.
- If a visible behaviour looks unusual for a pedestrian surveillance scene, include it as abnormal.
- Rules should be short, concrete, and behaviour-focused.
- Return 3 to 8 normal rules.
- Return 1 to 6 abnormal rules.
- If no obvious abnormal behaviour is visible, include only general safety-related abnormal rules.

Scene context:
{context}

Representative frame captions:
{caption_text}
""",
        }
    ]

    for row in sampled_rows:
        image_path = Path(row["image"])

        if image_path.exists():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(image_path),
                    },
                }
            )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    ).with_structured_output(GeneratedRules)

    result = llm.invoke(
        [
            {
                "role": "user",
                "content": content,
            }
        ]
    )

    return result.normal_rules, result.abnormal_rules


def generate_rules_from_captions(
    captions_file: Path,
    context_query: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Fallback rule suggestion using caption keywords only.

    This is deterministic and does not call an LLM.
    """
    captions = [
        row["caption"]
        for row in load_caption_rows(captions_file)
    ]

    haystack = " ".join(captions).lower()
    query = (context_query or "").lower()
    search_text = f"{haystack} {query}"

    normal_patterns = [
        (r"\bwalking\b|\bwalk\b", "People walking through the scene"),
        (r"\bstanding\b|\bstand\b", "People standing still without disruptive behaviour"),
        (r"\bsitting\b|\bsit\b|\bbench\b|\bledg(e)?\b", "People sitting on benches or near the pathway"),
        (r"\bphone\b", "People looking at a phone"),
        (r"\bbag\b|\bbackpack\b|\bumbrella\b", "People carrying personal objects such as bags, backpacks, or umbrellas"),
        (r"\btrash bin\b|\bgarbage bin\b|\bbin\b", "People near normal scene objects such as trash bins"),
    ]

    abnormal_patterns = [
        (r"\brunning\b|\brun\b", "Running in the monitored area"),
        (r"\bbicycle\b|\bbike\b|\bcycling\b", "Riding a bicycle in a pedestrian area"),
        (r"\bskateboard\b|\bskateboarding\b", "Skateboarding in a pedestrian area"),
        (r"\bfight\b|\bfighting\b|\bpushing\b|\bpush\b", "Fighting or pushing people"),
        (r"\blying on the ground\b|\blying down\b", "Lying on the ground"),
        (r"\bvandal\b|\bvandalizing\b|\bdamaging\b", "Vandalizing objects"),
    ]

    generated_normal = [
        rule for pattern, rule in normal_patterns
        if re.search(pattern, haystack)
    ]

    generated_abnormal = [
        rule for pattern, rule in abnormal_patterns
        if re.search(pattern, search_text)
    ]

    return list(dict.fromkeys(generated_normal)), list(dict.fromkeys(generated_abnormal))

def upsert_rules(
    rules_collection,
    encoder: SentenceTransformer,
    rules: list[tuple[str, str, str]],
    video_id: str | None = None,
) -> int:
    """
    Insert or update behaviour rules in Chroma.

    rules format:
    [
        ("People walking through the scene", "normal", "vision_generated"),
        ("Riding a bicycle in a pedestrian area", "abnormal", "vision_generated"),
    ]
    """
    count = 0
    scope = video_id or "global"

    for i, (rule_text, rule_type, source) in enumerate(rules):
        clean_rule = _normalise_rule(rule_text)

        if not clean_rule:
            continue

        if rule_type not in {"normal", "abnormal"}:
            raise ValueError(f"Invalid rule_type: {rule_type}")

        embedding = encoder.encode(
            clean_rule,
            normalize_embeddings=True,
        ).tolist()

        # Make ID stable enough for repeated uploads of the same video.
        safe_source = re.sub(r"[^a-zA-Z0-9_]+", "_", source)
        safe_type = re.sub(r"[^a-zA-Z0-9_]+", "_", rule_type)
        item_id = f"rule_{scope}_{safe_source}_{safe_type}_{i:03d}"

        rules_collection.upsert(
            ids=[item_id],
            embeddings=[embedding],
            documents=[clean_rule],
            metadatas=[
                {
                    "rule_type": rule_type,
                    "source": source,
                    "video_id": scope,
                }
            ],
        )

        count += 1

    return count


def build_rule_collection(
    chroma_path: str = "./chroma_db",
    normal_rules: list[str] | None = None,
    abnormal_rules: list[str] | None = None,
    video_id: str | None = None,
    source: str = "generated",
) -> int:
    """
    Build or update the Chroma `rules` collection.

    This version does NOT depend on hardcoded NORMAL_RULES or ABNORMAL_RULES.

    It only indexes the rules passed in by:
    - user input
    - vision-generated rules
    - caption-keyword fallback rules

    Each rule is stored with metadata:
    - rule_type: normal / abnormal
    - source: user / vision_generated / caption_keyword_fallback
    - video_id: uploaded video id, or "global"
    """
    selected_normal_rules = normal_rules or []
    selected_abnormal_rules = abnormal_rules or []

    if not selected_normal_rules and not selected_abnormal_rules:
        print("No rules provided. Skipping rule collection update.")
        return 0

    client = chromadb.PersistentClient(path=chroma_path)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    rules_collection = client.get_or_create_collection(
        name="rules",
        metadata={"hnsw:space": "cosine"},
    )

    all_rules = (
        [(rule, "normal", source) for rule in selected_normal_rules]
        + [(rule, "abnormal", source) for rule in selected_abnormal_rules]
    )

    count = upsert_rules(
        rules_collection=rules_collection,
        encoder=encoder,
        rules=all_rules,
        video_id=video_id,
    )

    print(
        f"Indexed {count} rules into Chroma collection `rules` "
        f"for video_id={video_id or 'global'} using source={source}"
    )

    return count

if __name__ == "__main__":
    build_rule_collection()