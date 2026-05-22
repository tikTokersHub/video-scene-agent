import chromadb
import torch
from transformers import AutoModel, AutoProcessor

SIGLIP_MODEL = "google/siglip2-base-patch16-224"


def embed_text(query: str) -> list[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device)
    model.eval()

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        text_features = features.pooler_output
        features = text_features / text_features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().tolist()


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("scenes")

print("Total scenes:", collection.count())

query = "person riding a bike"
query_embedding = embed_text(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print("-" * 50)
    print("Caption:", doc)
    print("Frame:", meta["frame_idx"])
    print("Time:", meta["timestamp_sec"])
    print("Frame path:", meta["frame_path"])
    print("Distance:", dist)