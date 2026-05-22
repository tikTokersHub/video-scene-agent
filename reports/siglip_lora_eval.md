# SigLIP 2 LoRA Evaluation

Text-to-image retrieval evaluation on held-out surveillance frame-caption pairs.

| Model | Recall@1 | Recall@5 |
|---|---:|---:|
| Vanilla SigLIP 2 | 3.23% | 12.90% |
| LoRA fine-tuned SigLIP 2 | 12.90% | 38.71% |

## Evaluation Setup

- Evaluation file: `data/labels/siglip_finetune.jsonl`
- Number of evaluation pairs: `31`
- Device: `cuda`
- Base model: `google/siglip2-base-patch16-224`
- LoRA adapter: `models/siglip_lora`
