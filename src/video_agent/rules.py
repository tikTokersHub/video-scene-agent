from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer


NORMAL_RULES = [
    "Walking alone or with others in various directions",
    "Standing still",
    "Looking at a phone",
    "Holding objects such as backpacks, bags, or umbrellas",
    "Sitting on benches",
    "Interacting with the trash bin",
]

ABNORMAL_RULES = [
    "Running",
    "Skateboarding",
    "Lying on the ground",
    "Fighting or pushing people",
    "Vandalizing objects",
    "Riding a bicycle in a pedestrian area",
]


def build_rule_collection(chroma_path: str = "./chroma_db") -> None:
    """
    Build a Chroma collection containing normal and abnormal behaviour rules.

    Each rule is embedded as text and stored with metadata:
    - rule_type: normal / abnormal
    - source: thesis
    """

    client = chromadb.PersistentClient(path=chroma_path)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    rules_collection = client.get_or_create_collection(
        name="rules",
        metadata={"hnsw:space": "cosine"},
    )

    all_rules = (
        [(rule, "normal") for rule in NORMAL_RULES]
        + [(rule, "abnormal") for rule in ABNORMAL_RULES]
    )

    for i, (rule_text, rule_type) in enumerate(all_rules):
        embedding = encoder.encode(rule_text).tolist()

        rules_collection.add(
            ids=[f"rule_{i:03d}"],
            embeddings=[embedding],
            documents=[rule_text],
            metadatas=[
                {
                    "rule_type": rule_type,
                    "source": "thesis",
                }
            ],
        )

    print(f"Indexed {len(all_rules)} rules into Chroma collection: rules")


if __name__ == "__main__":
    build_rule_collection()