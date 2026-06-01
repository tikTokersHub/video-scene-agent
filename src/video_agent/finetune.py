import json
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-224"

class FrameTextDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor):
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Label files not found on {jsonl_path}")
        
        self.item = [
            json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
        ]
        self.processor = processor

        if len(self.item) == 0:
            raise ValueError(f"No items found in {jsonl_path}")
        
    def __len__(self):
        return len(self.item)

    def __getitem__(self, index):
        item = self.item[index]
        image_value = item.get("image") or item.get("frame")
        text = item.get("text") or item.get("caption")

        image_path = Path(image_value)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
def siglip_loss(image_features, text_features, logit_scale, logit_bias):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    logits = logits * logit_scale + logit_bias

    batch_size = logits.size(0)

    labels = 2 * torch.eye(batch_size, device=logits.device) - 1

    loss = -torch.nn.functional.logsigmoid(labels * logits).sum() / batch_size
    return loss

def train(
        jsonl_path: str = "data/labels/siglip_finetune.jsonl",
        output_dir: str = "models/siglip_lora",
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-4,
        lora_r: int = 16,
        lora_alpha: int = 32
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)

    dataset = FrameTextDataset(Path(jsonl_path), processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    mlflow.set_experiment("siglip2-lora-finetune")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "n_samples": len(dataset),
                "loss_type": "siglip_pairwise_sigmoid",
                "device": device,
            }
        )

        for epoch in range(epochs):
            model.train()
            losses = []

            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                
                loss = siglip_loss(
                    image_features=outputs.image_embeds,
                    text_features=outputs.text_embeds,
                    logit_scale=model.base_model.model.logit_scale.exp(),
                    logit_bias=model.base_model.model.logit_bias,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if step % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Step {step}/{len(loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            mean_loss = sum(losses) / len(losses)

            print(f"Epoch {epoch + 1} finished. Mean loss: {mean_loss:.4f}")
            mlflow.log_metric("train_loss", mean_loss, step=epoch)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_path)
        processor.save_pretrained(output_path)

        mlflow.log_artifacts(output_dir)

        print(f"Saved LoRA adapter and processor to: {output_path}")

if __name__ == "__main__":
    train()