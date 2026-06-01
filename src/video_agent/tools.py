import chromadb
import torch
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from video_agent.ingest import SceneIngester


CHROMA_PATH = "./chroma_db"

_ingester: SceneIngester | None = None
_rule_encoder: SentenceTransformer | None = None
_chroma_client = None

def get_ingester() -> SceneIngester:
    global _ingester
    if _ingester is None:
        _ingester = SceneIngester(chroma_path=CHROMA_PATH)
    return _ingester

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    return _chroma_client


def get_rule_encoder() -> SentenceTransformer:
    global _rule_encoder
    if _rule_encoder is None:
        _rule_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    return _rule_encoder

@torch.no_grad()
def embed_text_siglip(query: str) -> list[float]:
    ingester = get_ingester()

    inputs = ingester.processor(
        text=[query],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    ).to(ingester.device)

    features = ingester.model.get_text_features(**inputs)
    text_features = features.pooler_output
    features = text_features / text_features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().tolist()

def embed_rule_text(query: str) -> list[float]:
    encoder = get_rule_encoder()
    embedding = encoder.encode(query, normalize_embeddings=True)
    return embedding.tolist()

@tool
def search_scenes_by_text(
    query: str,
    n_results: int = 5,
    start_sec: float | None = None,
    end_sec: float | None = None,
    video_id: str | None = None,
) -> list[dict]:
    """
    Find video scenes matching a natural-language description.

    Optionally restrict the semantic search to a time range.

    Use start_sec/end_sec when the user asks about:
    - start / beginning / first few seconds
    - early part
    - middle
    - end / near the end
    - around X seconds
    - before / after / between timestamps

    Example calls:
    - search_scenes_by_text("person riding a bike")
    - search_scenes_by_text("two people sitting near pathway", start_sec=x, end_sec=y)
    - search_scenes_by_text("bicycle", start_sec=x, end_sec=y)
    """
    ingester = get_ingester()
    query_embedding = embed_text_siglip(query)

    conditions = []

    if start_sec is not None:
        conditions.append({"timestamp_sec": {"$gte": start_sec}})

    if end_sec is not None:
        conditions.append({"timestamp_sec": {"$lte": end_sec}})

    if video_id is not None:
        conditions.append({"video_id": {"$eq": video_id}})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }

    if where is not None:
        query_kwargs["where"] = where

    results = ingester.collection.query(**query_kwargs)

    rows = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        rows.append(
            {
                "video_id": meta.get("video_id"),
                "frame_idx": meta.get("frame_idx"),
                "timestamp_sec": meta.get("timestamp_sec"),
                "frame_path": meta.get("frame_path"),
                "caption": doc,
                "distance": dist,
                "similarity_score": 1 - dist,
            }
        )

    return sorted(rows, key=lambda x: x["timestamp_sec"])

@tool
def search_scenes_by_time_range(
    start_sec: float,
    end_sec: float,
    limit: int = 20,
    video_id: str | None = None,
) -> list[dict]:
    """
    Retrieve scenes between two timestamps.

    Useful for questions like:
    - "What happened between 30 and 45 seconds?"
    - "Was anyone standing there for a long time?"
    """
     
    ingester = get_ingester()

    conditions = [
        {"timestamp_sec": {"$gte": start_sec}},
        {"timestamp_sec": {"$lte": end_sec}},
    ]

    if video_id is not None:
        conditions.append({"video_id": {"$eq": video_id}})

    results = ingester.collection.get(
        where={"$and": conditions},
        limit=limit,
    )

    rows = []

    for doc, meta in zip(results["documents"], results["metadatas"]):
        rows.append(
            {
                "video_id": meta.get("video_id"),
                "frame_idx": meta.get("frame_idx"),
                "timestamp_sec": meta.get("timestamp_sec"),
                "frame_path": meta.get("frame_path"),
                "caption": doc,
            }
        )

    return sorted(rows, key=lambda x: x["timestamp_sec"])

@tool
def get_neighbouring_frames(
    frame_idx: int,
    window: int = 5,
    video_id: str | None = None,
) -> list[dict]:
    """
    Get frames before and after a target frame.

    Useful for temporal reasoning:
    - checking whether someone stayed in the same area
    - checking whether motion continued
    - checking whether an event persisted across nearby frames
    """
    ingester = get_ingester()

    start_idx = frame_idx - window
    end_idx = frame_idx + window

    conditions = [
        {"frame_idx": {"$gte": start_idx}},
        {"frame_idx": {"$lte": end_idx}},
    ]

    if video_id is not None:
        conditions.append({"video_id": {"$eq": video_id}})

    results = ingester.collection.get(where={"$and": conditions})

    rows = []

    for doc, meta in zip(results["documents"], results["metadatas"]):
        rows.append(
            {
                "video_id": meta.get("video_id"),
                "frame_idx": meta.get("frame_idx"),
                "timestamp_sec": meta.get("timestamp_sec"),
                "frame_path": meta.get("frame_path"),
                "caption": doc,
            }
        )

    return sorted(rows, key=lambda x: x["frame_idx"])

@tool
def check_against_rules(
    scene_description: str,
    video_id: str | None = None,
) -> dict:
    """
    Compare a scene description against behaviour rules and return a normal/anomalous verdict.

    IMPORTANT FOR THE AGENT:
    - If an active video_id exists, you MUST pass it to this tool.
    - Video-specific user rules override generated/global rules.
    - Do not let a generic normal rule override a user-provided abnormal rule.
    - If the user only provided abnormal rules for a video, use those abnormal rules directly.
    - Only use video_id=None when there is no active uploaded/indexed video.

    Args:
        scene_description:
            A short description of the scene to classify.
            Example: "A person is walking on the grass."

        video_id:
            The active uploaded/indexed video ID.
            Example: "ud", "test_vid_5_28", "shanghai_01_0014".

    Returns:
        A dictionary containing matched rules, distances, rule scope, and verdict.
    """
    client = get_chroma_client()
    rules = client.get_collection("rules")

    query_embedding = embed_rule_text(scene_description)

    def build_where(
        rule_type: str,
        scoped_video_id: str | None = None,
        source: str | None = None,
    ) -> dict:
        conditions = [{"rule_type": {"$eq": rule_type}}]

        if scoped_video_id is not None:
            conditions.append({"video_id": {"$eq": scoped_video_id}})

        if source is not None:
            conditions.append({"source": {"$eq": source}})

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def query_rules(
        rule_type: str,
        scoped_video_id: str | None = None,
        source: str | None = None,
        n_results: int = 3,
    ) -> list[dict]:
        result = rules.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=build_where(
                rule_type=rule_type,
                scoped_video_id=scoped_video_id,
                source=source,
            ),
        )

        if not result.get("documents") or not result["documents"][0]:
            return []

        rows = []

        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            rows.append(
                {
                    "rule": doc,
                    "rule_type": meta.get("rule_type"),
                    "source": meta.get("source"),
                    "video_id": meta.get("video_id"),
                    "distance": dist,
                    "similarity_score": 1 - dist,
                }
            )

        return rows

    normal_matches: list[dict] = []
    abnormal_matches: list[dict] = []
    rule_scope = "none"

    # 1. Highest priority: video-specific USER rules.
    if video_id is not None:
        user_normal = query_rules(
            rule_type="normal",
            scoped_video_id=video_id,
            source="user",
        )
        user_abnormal = query_rules(
            rule_type="abnormal",
            scoped_video_id=video_id,
            source="user",
        )

        if user_normal or user_abnormal:
            normal_matches = user_normal
            abnormal_matches = user_abnormal
            rule_scope = f"{video_id}:user"

    # 2. If no user rules exist, use any video-specific rules.
    if not normal_matches and not abnormal_matches and video_id is not None:
        normal_matches = query_rules(
            rule_type="normal",
            scoped_video_id=video_id,
        )
        abnormal_matches = query_rules(
            rule_type="abnormal",
            scoped_video_id=video_id,
        )

        if normal_matches or abnormal_matches:
            rule_scope = f"{video_id}:video_specific"

    # 3. If no video-specific rules exist, use global rules.
    if not normal_matches and not abnormal_matches:
        normal_matches = query_rules(
            rule_type="normal",
            scoped_video_id="global",
        )
        abnormal_matches = query_rules(
            rule_type="abnormal",
            scoped_video_id="global",
        )

        if normal_matches or abnormal_matches:
            rule_scope = "global"

    # 4. Final fallback: any rule in the collection.
    if not normal_matches and not abnormal_matches:
        normal_matches = query_rules(rule_type="normal")
        abnormal_matches = query_rules(rule_type="abnormal")

        if normal_matches or abnormal_matches:
            rule_scope = "all_rules"

    best_normal = normal_matches[0] if normal_matches else None
    best_abnormal = abnormal_matches[0] if abnormal_matches else None

    normal_distance = best_normal["distance"] if best_normal else None
    abnormal_distance = best_abnormal["distance"] if best_abnormal else None

    # Decision logic:
    # If there are user/video-specific abnormal rules and no matching normal rules,
    # classify as anomalous instead of falling back to generic normal rules.
    if best_abnormal and not best_normal:
        verdict = "anomalous"
    elif best_normal and not best_abnormal:
        verdict = "normal"
    elif best_abnormal and best_normal:
        verdict = "anomalous" if abnormal_distance < normal_distance else "normal"
    else:
        verdict = "uncertain"

    return {
        "scene_description": scene_description,
        "video_id_used": video_id,
        "rule_scope": rule_scope,
        "best_normal_rule": best_normal["rule"] if best_normal else None,
        "best_normal_source": best_normal["source"] if best_normal else None,
        "best_normal_distance": normal_distance,
        "best_abnormal_rule": best_abnormal["rule"] if best_abnormal else None,
        "best_abnormal_source": best_abnormal["source"] if best_abnormal else None,
        "best_abnormal_distance": abnormal_distance,
        "normal_matches": normal_matches,
        "abnormal_matches": abnormal_matches,
        "verdict": verdict,
    }

@tool
def compare_two_timestamps(timestamp_a: float, timestamp_b: float) -> dict:
    """
    Compare what happened around two timestamps.

    Useful for questions like:
    - "Did the same activity continue?"
    - "What changed between 20 seconds and 50 seconds?"
    """
    window = 2.0

    scenes_a = search_scenes_by_time_range.invoke(
        {
            "start_sec": timestamp_a - window,
            "end_sec": timestamp_a + window,
            "limit": 5,
        }
    )

    scenes_b = search_scenes_by_time_range.invoke(
        {
            "start_sec": timestamp_b - window,
            "end_sec": timestamp_b + window,
            "limit": 5,
        }
    )

    captions_a = [scene["caption"] for scene in scenes_a]
    captions_b = [scene["caption"] for scene in scenes_b]

    return {
        "timestamp_a": timestamp_a,
        "timestamp_b": timestamp_b,
        "scenes_near_a": scenes_a,
        "scenes_near_b": scenes_b,
        "summary": {
            "captions_a": captions_a,
            "captions_b": captions_b,
        },
    }
