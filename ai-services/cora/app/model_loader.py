import json
from pathlib import Path
from typing import Optional, Any

import torch
from transformers import AutoTokenizer

from .config import config


class ModelLoader:
    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer = None
        self.label_maps = {}
        self.model_loaded = False
        self.loaded_checkpoint_path = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.load_model()

    def load_model(self):
        root = Path(__file__).resolve().parents[1]
        model_dir = (
            root / "outputs" / "experiments" / "bert_15epochs_v1"
        )
        checkpoint = model_dir / "best_model.pt"
        mappings_file = model_dir / "label_mappings.json"

        if not checkpoint.is_file() or not mappings_file.is_file():
            print(f"CORA BERT checkpoint not found: {model_dir}")
            return

        try:
            # Load the exact architecture used during training.
            import sys
            scripts_dir = str(root / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)

            from train_bert_gpu import HierarchicalBERT, TARGETS

            with mappings_file.open(encoding="utf-8") as f:
                self.label_maps = json.load(f)

            counts = [len(self.label_maps[t]) for t in TARGETS]
            self.model = HierarchicalBERT(
                str(model_dir / "encoder"), counts
            )

            state = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir / "tokenizer")
            )
            self.model_loaded = True
            self.loaded_checkpoint_path = str(checkpoint)
            print(f"CORA BERT loaded on {self.device}: {checkpoint}")

        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.label_maps = {}
            self.model_loaded = False
            self.loaded_checkpoint_path = None
            print(f"CORA BERT loading failed: {exc}")

    def predict(self, text: str):
        from train_bert_gpu import TARGETS

        tokens = self.tokenizer(
            text,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.inference_mode():
            logits = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        predictions = {}
        scores = {}

        for target, output in zip(TARGETS, logits):
            probabilities = torch.softmax(output.float(), dim=1)[0]
            idx = int(probabilities.argmax().item())
            inverse = {
                int(i): label
                for label, i in self.label_maps[target].items()
            }
            key = target.lower().replace("-", "_")
            predictions[key] = inverse.get(idx, "UNKNOWN")
            scores[key] = round(float(probabilities[idx].item()), 4)

        return predictions, scores


model_loader = ModelLoader()
