import json
import re
from pathlib import Path

import chromadb
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

SIGLIP_MODEL = "google/siglip2-base-patch16-224"

class SceneIngester:
    def __init__(
            self,
            chroma_path:str = './chroma_db',
            device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SigLIP 2 on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        self.model = AutoModel.from_pretrained(SIGLIP_MODEL).to(self.device)
        self.model.eval()

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="scenes",
            metadata={"hnsw:space": "cosine"},
        )


    @torch.no_grad()
    def embed_frame(self, frame_path: Path) -> list[float]:
        image = Image.open(frame_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        features = self.model.get_image_features(**inputs)
        image_features = features.pooler_output
        features = image_features / image_features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().tolist()
    
    def ingest(
            self,
            captions_file:Path,
            video_id:str,
            fps:float = 24.0,
            captioner_backend: str = "qwen2.5-vl",
    ):
        rows = [
            json.loads(line) for line in captions_file.read_text().splitlines() if line.strip()
        ]
        print(f"Found {len(rows)} caption rows.")

        for row in rows:
            frame_path = Path(row["image"])
            caption = row["text"]

            match = re.search(r"frame_(\d+)\.jpg", frame_path.name)
            if match is not None:
                frame_idx = int(match.group(1))
            else:
                match = re.search(r"(\d+)\.jpg", frame_path.name)
                if match is None:
                    raise ValueError(f"Could not parse frame index from {frame_path}")
                frame_idx = int(match.group(1))

            timestamp_sec = frame_idx / fps
            
            embedding = self.embed_frame(frame_path)
            item_id = f"{video_id}_{frame_idx:06d}_{captioner_backend}"

            self.collection.add(
                ids=[item_id],
                embeddings=[embedding],
                documents=[caption],
                metadatas=[
                    {
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "frame_path": str(frame_path),
                        "captioner_backend": captioner_backend,
                    }
                ],
            )

        print(f"Ingested {len(rows)} scenes into Chroma collection: scenes")  



