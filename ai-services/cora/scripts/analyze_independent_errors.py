import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bert_gpu import TARGETS, ComplaintDataset, HierarchicalBERT

BASE = Path("ai-services/cora")
EXP = BASE / "outputs/experiments/bert_independent_v1"
ENC = BASE / "outputs/experiments/bert_weighted_30epochs/encoder"
TOK = BASE / "outputs/experiments/bert_weighted_30epochs/tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndependentBERT(nn.Module):
    def __init__(self, encoder, shared, counts):
        super().__init__()
        self.encoder = encoder
        self.shared = shared
        self.heads = nn.ModuleList([
            nn.Linear(encoder.config.hidden_size, n) for n in counts
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = self.shared(out.last_hidden_state[:, 0, :])
        return [h(x) for h in self.heads]

maps = json.loads((EXP / "label_mappings.json").read_text())
val = pd.read_csv(BASE / "data/cora_validation.csv").fillna("")
counts = [len(maps[t]) for t in TARGETS]

source = HierarchicalBERT(str(ENC), counts)
model = IndependentBERT(source.encoder, source.shared, counts)
del source
model.load_state_dict(torch.load(
    EXP / "best_model.pt", map_location="cpu", weights_only=True
), strict=True)
model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(str(TOK))
loader = DataLoader(
    ComplaintDataset(val, tokenizer, maps, 256),
    batch_size=32, shuffle=False, num_workers=2
)

predictions = [[] for _ in TARGETS]
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
            predictions[i].extend(out.argmax(1).cpu().tolist())

for i, target in enumerate(TARGETS):
    y = []
    p = []
    for j, actual in enumerate(val[target].astype(str)):
        idx = maps[target].get(actual)
        if idx is not None:
            y.append(idx)
            p.append(predictions[i][j])

    report = classification_report(
        y, p,
        labels=list(range(counts[i])),
        output_dict=True, zero_division=0
    )
    per_class = []
    for label, idx in maps[target].items():
        row = report[str(idx)]
        per_class.append({
            "label": label,
            "train_count": int(
                pd.read_csv(BASE / "data/cora_train.csv")[target]
                .astype(str).eq(label).sum()
            ),
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1-score"],
            "support": int(row["support"]),
        })

    result = pd.DataFrame(per_class)
    outdir = EXP / "error_analysis"
    outdir.mkdir(exist_ok=True)
    result.sort_values("f1").to_csv(
        outdir / f"{target.lower().replace('-', '_')}.csv", index=False
    )
    print(f"\n{target}:")
    print("Zero-F1 classes:", int((result.f1 == 0).sum()), "/", len(result))
    print("\nWorst 10 classes:")
    print(result.sort_values("f1")[
        ["label", "train_count", "support", "precision", "recall", "f1"]
    ].head(10).to_string(index=False))

print("\nSaved detailed reports to:", (EXP / "error_analysis").resolve())
