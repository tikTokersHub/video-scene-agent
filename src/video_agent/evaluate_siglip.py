import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-224"

def load_items(jsonl_path:Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f'No jsonl found on {jsonl_path}')
    
    items = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

    if len(items) == 0:
        raise ValueError(f'There is no item in {jsonl_path}')
    
    return items

def load_basemodel(device:str):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model, processor

def load_lora_model(adapter_path: str, device: str):
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter folder not found: {adapter_path}. "
            "Run fine_tune.py first."
        )
    
    processor = AutoProcessor.from_pretrained(adapter_path)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()

    return model, processor

@torch.no_grad()
def encode_images(model, processor, image_paths:list[Path], device:str) -> torch.Tensor:
    embeddings = []

    for image_path in tqdm(image_paths, desc='Encoding images'):
        if not image_path.exists():
            raise FileNotFoundError(f'Image not found in {image_path}')

        image = Image.open(image_path).convert('RGB')

        inputs = processor(
            images=image,
            return_tensors='pt',
        ).to(device) 

        features = model.get_image_features(**inputs)
        image_features = features.pooler_output
        features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings.append(features.squeeze(0).cpu())

    return torch.stack(embeddings)

@torch.no_grad()
def encode_texts(model, processor, texts: list[str], device: str) -> torch.Tensor:
    embeddings = []

    for text in tqdm(texts, desc="Encoding texts"):
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(device)

        features = model.get_text_features(**inputs)
        text_features = features.pooler_output
        features = text_features / text_features.norm(dim=-1, keepdim=True)

        embeddings.append(features.squeeze(0).cpu())

    return torch.stack(embeddings)

def recall_at_k(similarity: torch.Tensor, k: int) -> float:
    topk_indices = similarity.topk(k=k, dim=1).indices

    correct = 0
    for i in range(similarity.size(0)):
        if i in topk_indices[i].tolist():
            correct += 1

    return correct / similarity.size(0)


def evaluate_model(model, processor, items: list[dict], device: str) -> dict:
    image_paths = [Path(item["image"]) for item in items]
    texts = [item["text"] for item in items]

    image_embeddings = encode_images(model, processor, image_paths, device)
    text_embeddings = encode_texts(model, processor, texts, device)

    similarity = text_embeddings @ image_embeddings.T

    return {
        "recall_at_1": recall_at_k(similarity, k=1),
        "recall_at_5": recall_at_k(similarity, k=5),
    }

def save_report(
    vanilla_results: dict,
    lora_results: dict,
    eval_path: Path,
    n_items: int,
    device: str,
):
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "siglip_lora_eval.md"

    report = f"""# SigLIP 2 LoRA Evaluation

Text-to-image retrieval evaluation on held-out surveillance frame-caption pairs.

| Model | Recall@1 | Recall@5 |
|---|---:|---:|
| Vanilla SigLIP 2 | {vanilla_results["recall_at_1"]:.2%} | {vanilla_results["recall_at_5"]:.2%} |
| LoRA fine-tuned SigLIP 2 | {lora_results["recall_at_1"]:.2%} | {lora_results["recall_at_5"]:.2%} |

## Evaluation Setup

- Evaluation file: `{eval_path}`
- Number of evaluation pairs: `{n_items}`
- Device: `{device}`
- Base model: `{MODEL_NAME}`
- LoRA adapter: `models/siglip_lora`
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {report_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_path = Path("data/labels/siglip_finetune.jsonl")
    adapter_path = "models/siglip_lora"

    items = load_items(eval_path)

    print(f"Using device: {device}")
    print(f"Loaded {len(items)} evaluation pairs")

    print("\nEvaluating vanilla SigLIP 2...")
    vanilla_model, vanilla_processor = load_basemodel(device)
    vanilla_results = evaluate_model(
        vanilla_model,
        vanilla_processor,
        items,
        device,
    )

    print("\nEvaluating LoRA fine-tuned SigLIP 2...")
    lora_model, lora_processor = load_lora_model(adapter_path, device)
    lora_results = evaluate_model(
        lora_model,
        lora_processor,
        items,
        device,
    )

    print("\nResults")
    print("-" * 60)
    print(f"{'Model':35s} {'Recall@1':>10s} {'Recall@5':>10s}")
    print("-" * 60)
    print(
        f"{'Vanilla SigLIP 2':35s} "
        f"{vanilla_results['recall_at_1']:>10.2%} "
        f"{vanilla_results['recall_at_5']:>10.2%}"
    )
    print(
        f"{'LoRA fine-tuned SigLIP 2':35s} "
        f"{lora_results['recall_at_1']:>10.2%} "
        f"{lora_results['recall_at_5']:>10.2%}"
    )

    save_report(
        vanilla_results=vanilla_results,
        lora_results=lora_results,
        eval_path=eval_path,
        n_items=len(items),
        device=device,
    )

if __name__ == "__main__":
    main()