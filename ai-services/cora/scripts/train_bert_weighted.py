import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from train_bert_gpu import (
    TARGETS,
    ComplaintDataset,
    HierarchicalBERT,
    seed_everything,
    evaluate,
)


def build_class_weights(df, target, label_map, device):
    counts = df[target].dropna().astype(str).value_counts()

    frequencies = np.array([
        max(int(counts.get(label, 0)), 1)
        for label, _ in sorted(
            label_map.items(), key=lambda item: item[1]
        )
    ], dtype=np.float64)

    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 3.0)
    weights = weights / weights.mean()

    return torch.tensor(
        weights, dtype=torch.float32, device=device
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--output",
        default="ai-services/cora/outputs/experiments/bert_weighted_v1",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    seed_everything(42)
    device = torch.device("cuda")
    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Path("ai-services/cora")
    checkpoint_dir = base / "outputs/experiments/bert_base_gpu"
    train_df = pd.read_csv(base / "data/cora_train.csv")
    val_df = pd.read_csv(base / "data/cora_validation.csv")

    label_maps = json.loads(
        (checkpoint_dir / "label_mappings.json").read_text()
    )

    for target in TARGETS:
        current = set(train_df[target].dropna().astype(str))
        saved = set(label_maps[target])
        if current != saved:
            raise RuntimeError(
                f"Label mapping mismatch for {target}; stopping safely."
            )

    print("Verified saved label mappings.", flush=True)

    for target in TARGETS:
        unseen = (
            set(val_df[target].dropna().astype(str))
            - set(label_maps[target])
        )
        if unseen:
            print(
                f"WARNING {target}: {len(unseen)} validation "
                "labels are unseen in training."
            )

    model_name = str(checkpoint_dir / "encoder")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir / "tokenizer"
    )

    train_ds = ComplaintDataset(
        train_df, tokenizer, label_maps, args.max_length
    )
    val_ds = ComplaintDataset(
        val_df, tokenizer, label_maps, args.max_length
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = HierarchicalBERT(
        model_name, [len(label_maps[t]) for t in TARGETS]
    ).to(device)

    state = torch.load(
        checkpoint_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    print("Loaded existing trained checkpoint.", flush=True)

    class_weights = {
        target: build_class_weights(
            train_df, target, label_maps[target], device
        )
        for target in TARGETS
    }

    for target, weights in class_weights.items():
        print(
            f"{target}: weight range "
            f"{weights.min().item():.3f}–{weights.max().item():.3f}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
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
                    losses.append(
                        F.cross_entropy(
                            outputs[i],
                            labels[:, i],
                            weight=class_weights[target],
                            ignore_index=-100,
                        )
                    )

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

            model.encoder.save_pretrained(output_dir / "encoder")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            torch.save(model.state_dict(), output_dir / "best_model.pt")

            with open(output_dir / "label_mappings.json", "w") as f:
                json.dump(label_maps, f, indent=2)

            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print("Saved new best weighted checkpoint.", flush=True)
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping.", flush=True)
                break

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("Training finished.")
    print("Best mean macro-F1:", round(best_f1, 4))
    print("Saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
