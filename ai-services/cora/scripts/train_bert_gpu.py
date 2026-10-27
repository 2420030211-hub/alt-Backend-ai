import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score


TARGETS = ["Product", "Sub-product", "Issue", "Sub-issue"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ComplaintDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_maps, max_length):
        self.texts = dataframe["text"].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = []

        for _, row in dataframe.iterrows():
            encoded = []
            for target in TARGETS:
                value = str(row[target]) if pd.notna(row[target]) else ""
                encoded.append(label_maps[target].get(value, -100))
            self.labels.append(encoded)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class HierarchicalBERT(nn.Module):
    def __init__(self, model_name, class_counts):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.head_product = nn.Linear(hidden, class_counts[0])
        self.head_subproduct = nn.Linear(hidden + class_counts[0], class_counts[1])
        self.head_issue = nn.Linear(
            hidden + class_counts[0] + class_counts[1], class_counts[2]
        )
        self.head_subissue = nn.Linear(
            hidden + class_counts[0] + class_counts[1] + class_counts[2],
            class_counts[3],
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        shared = self.shared(output.last_hidden_state[:, 0, :])

        product = self.head_product(shared)
        subproduct = self.head_subproduct(torch.cat([shared, product], dim=1))
        issue = self.head_issue(
            torch.cat([shared, product, subproduct], dim=1)
        )
        subissue = self.head_subissue(
            torch.cat([shared, product, subproduct, issue], dim=1)
        )

        return [product, subproduct, issue, subissue]


def evaluate(model, loader, device, label_maps):
    model.eval()
    true_labels = [[] for _ in TARGETS]
    predictions = [[] for _ in TARGETS]

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = model(ids, mask)

            for i, logits in enumerate(outputs):
                preds = logits.argmax(dim=1).cpu()
                actual = labels[:, i]
                valid = actual != -100
                true_labels[i].extend(actual[valid].tolist())
                predictions[i].extend(preds[valid].tolist())

    metrics = {}
    f1_values = []

    for i, target in enumerate(TARGETS):
        if not true_labels[i]:
            metrics[target] = {"accuracy": None, "macro_f1": None}
            continue

        acc = accuracy_score(true_labels[i], predictions[i])
        f1 = f1_score(
            true_labels[i],
            predictions[i],
            labels=list(range(len(label_maps[target]))),
            average="macro",
            zero_division=0,
        )
        metrics[target] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "evaluated_rows": len(true_labels[i]),
        }
        f1_values.append(f1)

    metrics["mean_macro_f1"] = (
        float(np.mean(f1_values)) if f1_values else 0.0
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="ai-services/cora/data/cora_train.csv",
    )
    parser.add_argument(
        "--validation",
        default="ai-services/cora/data/cora_validation.csv",
    )
    parser.add_argument(
        "--output",
        default="ai-services/cora/outputs/experiments/bert_base_gpu",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    seed_everything(42)
    device = torch.device("cuda")
    model_name = "google-bert/bert-base-uncased"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loading datasets...", flush=True)
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)

    required = ["text"] + TARGETS
    for name, df in [("training", train_df), ("validation", val_df)]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {missing}")

    label_maps = {}
    for target in TARGETS:
        classes = sorted(
            train_df[target].dropna().astype(str).unique().tolist()
        )
        label_maps[target] = {label: i for i, label in enumerate(classes)}
        print(f"{target}: {len(classes)} training classes", flush=True)

    # Report validation labels not represented in the training set.
    for target in TARGETS:
        train_classes = set(label_maps[target])
        val_classes = set(val_df[target].dropna().astype(str))
        unseen = val_classes - train_classes
        if unseen:
            print(
                f"WARNING: {target} has {len(unseen)} validation labels "
                "absent from training; those rows are excluded for that target.",
                flush=True,
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = ComplaintDataset(
        train_df, tokenizer, label_maps, args.max_length
    )
    val_ds = ComplaintDataset(
        val_df, tokenizer, label_maps, args.max_length
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2,
        pin_memory=True,
    )

    model = HierarchicalBERT(
        model_name, [len(label_maps[t]) for t in TARGETS]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_f1 = -1.0
    patience = 0
    history = []

    print(
        f"Training rows: {len(train_ds)} | Validation rows: {len(val_ds)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(ids, mask)
                losses = [
                    nn.functional.cross_entropy(
                        outputs[i], labels[:, i], ignore_index=-100
                    )
                    for i in range(len(TARGETS))
                ]
                loss = sum(losses) / len(losses)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step + 1}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}",
                    flush=True,
                )

        metrics = evaluate(model, val_loader, device, label_maps)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / max(len(train_loader), 1)
        history.append(metrics)

        print(json.dumps(metrics, indent=2), flush=True)

        if metrics["mean_macro_f1"] > best_f1:
            best_f1 = metrics["mean_macro_f1"]
            patience = 0
            model.encoder.save_pretrained(output_dir / "encoder")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            torch.save(
                model.state_dict(), output_dir / "best_model.pt"
            )
            with open(output_dir / "label_mappings.json", "w") as f:
                json.dump(label_maps, f, indent=2)
            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print("Saved new best checkpoint.", flush=True)
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping: validation score did not improve.", flush=True)
                break

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining finished.")
    print("Best macro-F1 mean:", round(best_f1, 4))
    print("Saved files:", output_dir.resolve())


if __name__ == "__main__":
    main()
