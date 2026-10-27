import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bert_gpu import (
    TARGETS, ComplaintDataset, HierarchicalBERT,
    evaluate, seed_everything,
)

BASE = Path("ai-services/cora")
SOURCE = BASE / "outputs/experiments/bert_targetweighted_v1"
ENCODER_DIR = BASE / "outputs/experiments/bert_weighted_30epochs/encoder"
OUTPUT = BASE / "outputs/experiments/bert_independent_v1"
TOKENIZER_DIR = BASE / "outputs/experiments/bert_weighted_30epochs/tokenizer"

if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise RuntimeError(f"{OUTPUT} already contains files; refusing to overwrite.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable.")

seed_everything(42)
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")

train_df = pd.read_csv(BASE / "data/cora_train.csv")
val_df = pd.read_csv(BASE / "data/cora_validation.csv")
label_maps = json.loads((SOURCE / "label_mappings.json").read_text())
class_counts = [len(label_maps[t]) for t in TARGETS]

print("Loading existing best BERT checkpoint...", flush=True)
source_model = HierarchicalBERT(str(ENCODER_DIR), class_counts)
state = torch.load(SOURCE / "best_model.pt", map_location="cpu", weights_only=True)
source_model.load_state_dict(state, strict=True)

class IndependentBERT(nn.Module):
    def __init__(self, encoder, shared, counts):
        super().__init__()
        self.encoder = encoder
        self.shared = shared
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleList([
            nn.Linear(hidden, count) for count in counts
        ])

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = self.shared(output.last_hidden_state[:, 0, :])
        return [head(features) for head in self.heads]

model = IndependentBERT(
    source_model.encoder,
    source_model.shared,
    class_counts,
).to(device)
del source_model
torch.cuda.empty_cache()

tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
train_ds = ComplaintDataset(train_df, tokenizer, label_maps, 256)
val_ds = ComplaintDataset(val_df, tokenizer, label_maps, 256)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2,
                          pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2,
                        pin_memory=True)

# Moderate inverse-frequency weighting; cap extreme weights to reduce instability.
class_weights = []
for target in TARGETS:
    counts = train_df[target].astype(str).value_counts()
    weights = []
    for label, idx in sorted(label_maps[target].items(), key=lambda x: x[1]):
        weights.append(1.0 / max(int(counts.get(label, 0)), 1) ** 0.5)
    w = torch.tensor(weights, dtype=torch.float32)
    w = (w / w.mean()).clamp(0.5, 5.0)
    class_weights.append(w.to(device))

optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": list(model.shared.parameters()) + list(model.heads.parameters()),
     "lr": 5e-5},
], weight_decay=0.01)

epochs = 8
total_steps = epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.06 * total_steps),
    num_training_steps=total_steps
)
scaler = torch.amp.GradScaler("cuda")
best_score = -1.0
patience = 3
stale = 0

OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "label_mappings.json").write_text(json.dumps(label_maps, indent=2))
(OUTPUT / "experiment_config.json").write_text(json.dumps({
    "architecture": "independent_heads",
    "source_checkpoint": str(SOURCE / "best_model.pt"),
    "epochs": epochs, "batch_size": 16, "max_length": 256,
    "encoder_lr": 1e-5, "head_lr": 5e-5,
    "seed": 42
}, indent=2))

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Training rows:", len(train_ds), "| Validation rows:", len(val_ds), flush=True)

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, 1):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(ids, mask)
            losses = []
            for i, logits in enumerate(outputs):
                valid = labels[:, i] != -100
                if valid.any():
                    losses.append(nn.functional.cross_entropy(
                        logits[valid], labels[valid, i],
                        weight=class_weights[i]
                    ))
            loss = sum(losses) / max(len(losses), 1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch}/{epochs} step {step}/{len(train_loader)} "
                  f"loss={running_loss/step:.4f}", flush=True)

    metrics = evaluate(model, val_loader, device, label_maps)
    score = metrics["mean_macro_f1"]
    print(f"\nEPOCH {epoch} | train_loss={running_loss/len(train_loader):.4f}",
          flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    (OUTPUT / "latest_metrics.json").write_text(json.dumps(metrics, indent=2))
    if score > best_score:
        best_score = score
        stale = 0
        torch.save(model.state_dict(), OUTPUT / "best_model.pt")
        (OUTPUT / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"NEW BEST mean_macro_f1={best_score:.4%}", flush=True)
    else:
        stale += 1
        print(f"No improvement: {stale}/{patience}", flush=True)
        if stale >= patience:
            print("Early stopping.", flush=True)
            break

print("\nEXPERIMENT COMPLETE", flush=True)
print("Best metrics:", (OUTPUT / "best_metrics.json").read_text(), flush=True)
print("Saved checkpoint:", OUTPUT / "best_model.pt", flush=True)
print("Existing API and checkpoints were not changed.", flush=True)
