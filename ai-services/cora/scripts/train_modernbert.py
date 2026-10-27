import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from train_bert_gpu import (
    TARGETS,
    ComplaintDataset,
    seed_everything,
    evaluate,
)


MODEL_NAME = "answerdotai/ModernBERT-base"


class HierarchicalModernBERT(nn.Module):
    def __init__(self, model_name, class_counts):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.head_product = nn.Linear(hidden, class_counts[0])
        self.head_subproduct = nn.Linear(
            hidden + class_counts[0], class_counts[1]
        )
        self.head_issue = nn.Linear(
            hidden + class_counts[0] + class_counts[1],
            class_counts[2],
        )
        self.head_subissue = nn.Linear(
            hidden + class_counts[0] + class_counts[1] + class_counts[2],
            class_counts[3],
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # ModernBERT's first token representation.
        shared = self.shared(output.last_hidden_state[:, 0, :])

        product = self.head_product(shared)
        subproduct = self.head_subproduct(
            torch.cat([shared, product], dim=1)
        )
        issue = self.head_issue(
            torch.cat([shared, product, subproduct], dim=1)
        )
        subissue = self.head_subissue(
            torch.cat([shared, product, subproduct, issue], dim=1)
        )

        return [product, subproduct, issue, subissue]


def build_class_weights(df, target, label_map, device):
    counts = df[target].dropna().astype(str).value_counts()
    frequencies = np.array([
        max(int(counts.get(label, 0)), 1)
        for label, _ in sorted(
            label_map.items(), key=lambda item: item[1]
        )
    ], dtype=np.float64)

    weights = 1.0 / np.sqrt(frequencies)
    weights /= weights.mean()
    weights = np.clip(weights, 0.5, 3.0)
    weights /= weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--output",
        default="ai-services/cora/outputs/experiments/modernbert_v1",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    seed_everything(42)
    device = torch.device("cuda")

    base = Path("ai-services/cora")
    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}"
        )

    train_df = pd.read_csv(base / "data/cora_train.csv")
    val_df = pd.read_csv(base / "data/cora_validation.csv")

    # Build label mappings from training data only.
    label_maps = {}
    for target in TARGETS:
        labels = sorted(train_df[target].dropna().astype(str).unique())
        label_maps[target] = {
            label: idx for idx, label in enumerate(labels)
        }

    print("Label counts:")
    for target in TARGETS:
        print(f"  {target}: {len(label_maps[target])}")

    for target in TARGETS:
        unseen = (
            set(val_df[target].dropna().astype(str))
            - set(label_maps[target])
        )
        if unseen:
            print(
                f"WARNING: {target} has {len(unseen)} "
                "validation labels absent from training."
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = ComplaintDataset(
        train_df, tokenizer, label_maps, args.max_length
    )
    val_ds = ComplaintDataset(
        val_df, tokenizer, label_maps, args.max_length
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = HierarchicalModernBERT(
        MODEL_NAME, [len(label_maps[t]) for t in TARGETS]
    ).to(device)

    class_weights = {
        target: build_class_weights(
            train_df, target, label_maps[target], device
        )
        for target in TARGETS
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.06),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda")
    best_f1 = -1.0
    patience = 0
    history = []

    # Loss weighting emphasizes the more difficult hierarchy levels.
    target_weights = [1.0, 1.0, 1.5, 2.0]

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training rows:", len(train_ds))
    print("Validation rows:", len(val_ds))
    print("Output:", output_dir.resolve())

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                outputs = model(ids, mask)
                losses = []

                for i, target in enumerate(TARGETS):
                    losses.append(F.cross_entropy(
                        outputs[i],
                        labels[:, i],
                        weight=class_weights[target],
                        ignore_index=-100,
                    ))

                loss = sum(
                    w * item for w, item in zip(target_weights, losses)
                ) / sum(target_weights)

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
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Step {step + 1}/{len(train_loader)} "
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
            output_dir.mkdir(parents=True, exist_ok=True)

            model.encoder.save_pretrained(output_dir / "encoder")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            torch.save(model.state_dict(), output_dir / "best_model.pt")

            (output_dir / "label_mappings.json").write_text(
                json.dumps(label_maps, indent=2)
            )
            (output_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            print("Saved new best ModernBERT checkpoint.", flush=True)
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping.", flush=True)
                break

        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2)
        )

    print("Training finished.")
    print("Best mean macro-F1:", round(best_f1, 4))
    print("Saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
