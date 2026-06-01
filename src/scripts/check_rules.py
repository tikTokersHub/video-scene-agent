import argparse

import chromadb


def check_rules(
    video_id: str,
    chroma_path: str = "./chroma_db",
) -> None:
    client = chromadb.PersistentClient(path=chroma_path)
    rules = client.get_collection("rules")

    results = rules.get(
        where={"video_id": {"$eq": video_id}},
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    if not documents:
        print(f"No rules found for video_id={video_id!r}")
        return

    print(f"\nFound {len(documents)} rules for video_id={video_id!r}\n")

    normal_rules = []
    abnormal_rules = []

    for rule_id, document, metadata in zip(ids, documents, metadatas):
        row = {
            "id": rule_id,
            "rule": document,
            "source": metadata.get("source"),
            "rule_type": metadata.get("rule_type"),
            "video_id": metadata.get("video_id"),
        }

        if row["rule_type"] == "normal":
            normal_rules.append(row)
        elif row["rule_type"] == "abnormal":
            abnormal_rules.append(row)

    print("=== Normal Rules ===")
    if normal_rules:
        for i, row in enumerate(normal_rules, start=1):
            print(f"{i}. {row['rule']}")
            print(f"   id={row['id']}")
            print(f"   source={row['source']}")
    else:
        print("No normal rules.")

    print("\n=== Abnormal Rules ===")
    if abnormal_rules:
        for i, row in enumerate(abnormal_rules, start=1):
            print(f"{i}. {row['rule']}")
            print(f"   id={row['id']}")
            print(f"   source={row['source']}")
    else:
        print("No abnormal rules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-id",
        default="ud",
        help="Video ID to inspect, e.g. ud",
    )
    parser.add_argument(
        "--chroma-path",
        default="./chroma_db",
        help="Path to Chroma persistent DB",
    )

    args = parser.parse_args()

    check_rules(
        video_id=args.video_id,
        chroma_path=args.chroma_path,
    )