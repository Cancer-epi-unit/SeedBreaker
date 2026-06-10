#!/usr/bin/env python3
"""
train_lora.py
-------------
SeedBreaker Phase 5 — LoRA fine-tuning of Nucleotide Transformer v2 (500M)
for miRNA binding site disruption prediction.

Trains a binary classifier on top of NT-v2 using LoRA adapters.
Optimised for Apple Silicon (MPS backend, fp32, gradient accumulation).

Usage:
  python3 train_lora.py \
    --data  training_data/training_data.parquet \
    --out   models/seedbreaker_lora_v1 \
    --epochs 10 \
    --batch  8 \
    --lr     2e-4

Requirements:
  pip install transformers peft datasets torch pandas pyarrow
  pip install accelerate scikit-learn
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    sys.exit("pip install transformers peft accelerate")

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL  = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROPOUT= 0.1

# NT-v2 attention layer names to apply LoRA to
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]


# ── Dataset ────────────────────────────────────────────────────────────────────

class SeedDisruptionDataset(Dataset):
    """
    Dataset of (ref_window, alt_window, label) triples.
    Encodes REF and ALT as a sequence pair separated by [SEP].
    The model learns the delta between REF and ALT at the variant position.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # NT-v2 uses the ESM tokenizer which does not support sequence pairs.
        # Concatenate REF and ALT with a spacer so the model sees both windows
        # and can learn the delta between them.
        combined = row["ref_window"] + "NNN" + row["alt_window"]

        encoding = self.tokenizer(
            combined,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(int(row["label"]), dtype=torch.long),
            "confidence":     torch.tensor(float(row["label_confidence"]),
                                           dtype=torch.float),
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, prefix=""):
    """Compute AUROC, AUPRC, accuracy, and calibration error."""
    preds = (probs >= 0.5).astype(int)
    acc   = (preds == labels).mean()

    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except ValueError:
        auroc = auprc = float("nan")

    # Expected calibration error (10 bins)
    bins   = np.linspace(0, 1, 11)
    ece    = 0.0
    n      = len(labels)
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            bin_acc  = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() / n * abs(bin_acc - bin_conf)

    tag = f"{prefix}_" if prefix else ""
    return {
        f"{tag}auroc":    round(auroc, 4),
        f"{tag}auprc":    round(auprc, 4),
        f"{tag}accuracy": round(acc,   4),
        f"{tag}ece":      round(ece,   4),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    # ── Device ──────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU (will be slow)")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading training data: {args.data}")
    df = pd.read_parquet(args.data)
    print(f"  Total rows: {len(df):,}")
    print(f"  Label balance: {df['label'].mean()*100:.1f}% disruptive")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    # Subsample training set if requested
    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(args.max_train, random_state=42)
        print(f"  Training subsampled to {len(train_df):,}")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    train_ds = SeedDisruptionDataset(train_df, tokenizer, args.max_length)
    val_ds   = SeedDisruptionDataset(val_df,   tokenizer, args.max_length)
    test_ds  = SeedDisruptionDataset(test_df,  tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch * 2,
                               shuffle=False, num_workers=0)

    # ── Model + LoRA ────────────────────────────────────────────────────────
    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels = 2,
        ignore_mismatched_sizes = True,
        trust_remote_code = True,
    )

    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGET_MODULES,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.SEQ_CLS,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # ── Optimizer + scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = args.lr,
        weight_decay = 0.01,
    )

    total_steps   = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps  = int(total_steps * 0.06)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ── Class weights for imbalanced labels ─────────────────────────────────
    n_pos    = (train_df["label"] == 1).sum()
    n_neg    = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight = torch.tensor([1.0, pos_weight.item()]).to(device)
    )

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size:          {args.batch}")
    print(f"  Gradient accum:      {args.grad_accum} (effective batch: {args.batch * args.grad_accum})")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Warmup steps:        {warmup_steps}")
    print()

    best_val_auroc = 0.0
    history        = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss    = loss / args.grad_accum
            loss.backward()

            train_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_labels, val_probs = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs        = model(input_ids=input_ids,
                                       attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch["labels"].numpy())

        val_metrics = compute_metrics(
            np.array(val_labels), np.array(val_probs), prefix="val"
        )

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={avg_train_loss:.4f}  "
              f"val_auroc={val_metrics['val_auroc']:.4f}  "
              f"val_auprc={val_metrics['val_auprc']:.4f}  "
              f"val_acc={val_metrics['val_accuracy']:.4f}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss,
                         **val_metrics})

        # Save best model
        if val_metrics["val_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["val_auroc"]
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            print(f"  → New best model saved (val_auroc={best_val_auroc:.4f})")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\nEvaluating on held-out test set (chr8 + chr18)...")

    # Load best model
    best_model = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True, trust_remote_code=True
        ),
        out_dir / "best"
    ).to(device)
    best_model.eval()

    test_labels, test_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = best_model(input_ids=input_ids,
                                        attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(batch["labels"].numpy())

    test_metrics = compute_metrics(
        np.array(test_labels), np.array(test_probs), prefix="test"
    )

    print(f"\n{'='*50}")
    print("TEST SET RESULTS")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k:<20} {v:.4f}")
    print(f"{'='*50}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    results = {
        "base_model":    BASE_MODEL,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "epochs":        args.epochs,
        "batch_size":    args.batch,
        "lr":            args.lr,
        "window":        args.max_length,
        "best_val_auroc":best_val_auroc,
        "test_metrics":  test_metrics,
        "history":       history,
    }
    with open(out_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save final model (last epoch)
    model.save_pretrained(out_dir / "final")
    tokenizer.save_pretrained(out_dir / "final")

    print(f"\nArtifacts saved to: {out_dir}")
    print(f"  best/   — best validation checkpoint")
    print(f"  final/  — final epoch checkpoint")
    print(f"  training_results.json")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",       required=True,
                    help="Training parquet from build_training_data.py")
    ap.add_argument("--out",        default="models/seedbreaker_lora_v1",
                    help="Output directory for model artifacts")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch",      type=int,   default=8,
                    help="Per-device batch size (default: 8)")
    ap.add_argument("--grad-accum", type=int,   default=4,
                    help="Gradient accumulation steps (default: 4, effective batch=32)")
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--max-length", type=int,   default=256,
                    help="Max tokenized sequence length (default: 256)")
    ap.add_argument("--max-train",  type=int,   default=None,
                    help="Subsample training set (for testing, default: all)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Training data not found: {args.data}")

    train(args)


if __name__ == "__main__":
    main()
