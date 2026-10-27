import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bert_gpu import TARGETS, ComplaintDataset, HierarchicalBERT

BASE = Path("ai-services/cora")
EXP = BASE / "outputs/experiments/bert_independent_v1"
ENCODER = BASE / "outputs/experiments/bert_weighted_30epochs/encoder"
TOKENIZER = BASE / "outputs/experiments/bert_weighted_30epochs/tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndependentBERT(nn.Module):
    def __init__(self, encoder, shared, counts):
        super().__init__()
        self.encoder = encoder
        self.shared = shared
        self.heads = nn.ModuleList([
            nn.Linear(self.encoder.config.hidden_size, n) for n in counts
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = self.shared(out.last_hidden_state[:, 0, :])
        return [head(features) for head in self.heads]

label_maps = json.loads((EXP / "label_mappings.json").read_text())
train = pd.read_csv(BASE / "data/cora_train.csv").fillna("")
val = pd.read_csv(BASE / "data/cora_validation.csv").fillna("")
counts = [len(label_maps[t]) for t in TARGETS]

source = HierarchicalBERT(str(ENCODER), counts)
model = IndependentBERT(source.encoder, source.shared, counts)
del source

state = torch.load(EXP / "best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
ds = ComplaintDataset(val, tokenizer, label_maps, 256)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

logits_all = [[] for _ in TARGETS]
labels_all = [[] for _ in TARGETS]

with torch.inference_mode():
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=device.type == "cuda"
        ):
            outputs = model(ids, mask)
        for i, out in enumerate(outputs):
            logits_all[i].append(out.float().cpu())
            labels_all[i].append(batch["labels"][:, i].cpu())

logits_all = [torch.cat(x).numpy() for x in logits_all]
labels_all = [torch.cat(x).numpy() for x in labels_all]

# Parent-child compatibility learned ONLY from training labels.
pairs = [(0, 1), (1, 2), (2, 3)]
compat = []
for parent_i, child_i in pairs:
    parent_col, child_col = TARGETS[parent_i], TARGETS[child_i]
    mapping = {}
    for child, group in train.groupby(child_col, dropna=False):
        child = str(child)
        if child not in label_maps[child_col]:
            continue
        parents = set(group[parent_col].astype(str))
        mapping[label_maps[child_col][child]] = {
            label_maps[parent_col][p]
            for p in parents if p in label_maps[parent_col]
        }
    compat.append(mapping)

def score(preds, name):
    f1s = []
    print(f"\n{name}")
    for i, target in enumerate(TARGETS):
        valid = labels_all[i] != -100
        y = labels_all[i][valid]
        p = preds[i][valid]
        acc = accuracy_score(y, p)
        f1 = f1_score(
            y, p, labels=list(range(counts[i])),
            average="macro", zero_division=0
        )
        f1s.append(f1)
        print(f"{target:12} accuracy={acc:.4f} macro_f1={f1:.4f} rows={len(y)}")
    print(f"mean_macro_f1={np.mean(f1s):.6f}")

original = [x.argmax(axis=1) for x in logits_all]
score(original, "ORIGINAL INDEPENDENT MODEL")

# Soft reranking: penalize a child label when its predicted parent
# has never co-occurred with that child in training. Try mild strengths.
for alpha in [0.25, 0.5, 1.0]:
    adjusted = [x.copy() for x in logits_all]
    for parent_i, child_i in pairs:
        parent_probs = torch.softmax(
            torch.tensor(logits_all[parent_i]), dim=1
        ).numpy()
        child_scores = adjusted[child_i]
        mapping = compat[pairs.index((parent_i, child_i))]
        for child_idx in range(child_scores.shape[1]):
            allowed = mapping.get(child_idx, set())
            if not allowed:
                continue
            allowed_mass = parent_probs[:, list(allowed)].sum(axis=1)
            child_scores[:, child_idx] += alpha * np.log(
                np.clip(allowed_mass, 1e-8, 1.0)
            )
    preds = [x.argmax(axis=1) for x in adjusted]
    score(preds, f"SOFT HIERARCHY RERANKING alpha={alpha}")

print("\nNo checkpoint or API files were changed.")
