# RunPod & Cloud GPU Setup Guide

## Budget & GPU Recommendations (Budget: ~$12)

| GPU Option | VRAM | Spot Rate / hr | Secure Rate / hr | Recommended Use |
| --- | --- | --- | --- | --- |
| RTX 3090 / 4090 | 24 GB | ~$0.30 - $0.44 | ~$0.50 - $0.74 | ModernBERT & BERT fine-tuning, Qwen3-4B QLoRA |
| A4000 / A5000 | 16-24 GB | ~$0.20 - $0.35 | ~$0.35 - $0.50 | Cost-effective classifier training |
| A100 (80GB) | 80 GB | ~$1.60 - $2.10 | ~$2.50 | Not recommended under $12 budget |

**Target Training Runtime**: 2-3 hours on an RTX 3090 (~$1.20 - $1.50 total spend), well within the $12 budget limit.

## Step-by-Step RunPod Deployment
1. Select PyTorch 2.2 template on RunPod.
2. Clone repository: `git clone <repo-url>`
3. Set up environment:
   ```bash
   pip install -r ai-services/cora/requirements.txt
   pip install -r ai-services/qwen/requirements.txt
   ```
4. Perform fair benchmark:
   ```bash
   python ai-services/cora/scripts/evaluate.py
   ```
5. **CRITICAL**: Always terminate or stop the pod immediately when training completes to avoid draining the balance.
