import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix

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
train = pd.read_csv(BASE / "data/cora_train.csv").fillna("")
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

preds = [[] for _ in TARGETS]
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
            preds[i].extend(out.argmax(1).cpu().tolist())

for target in ["Sub-product", "Issue", "Sub-issue"]:
    i = TARGETS.index(target)
    y = [
        maps[target][str(v)] if str(v) in maps[target] else -100
        for v in val[target]
    ]
    p = preds[i]
    rows = []

    for true_id in sorted(set(y) - {-100}):
        indices = [j for j, value in enumerate(y) if value == true_id]
        correct = sum(p[j] == true_id for j in indices)
        if correct == 0:
            alternatives = {}
            for j in indices:
                name = list(maps[target].keys())[p[j]]
                alternatives[name] = alternatives.get(name, 0) + 1
            for predicted, n in sorted(
                alternatives.items(), key=lambda x: -x[1]
            )[:3]:
                rows.append({
                    "target": target,
                    "true_label": list(maps[target].keys())[true_id],
                    "support": len(indices),
                    "predicted_label": predicted,
                    "wrong_predictions": n,
                })

    result = pd.DataFrame(rows)
    print(f"\n{target}: most common destinations for missed classes")
    if len(result):
        print(result.head(20).to_string(index=False))
    else:
        print("No completely missed validation classes.")

print("\nRead-only analysis; no checkpoint or API changes.")
