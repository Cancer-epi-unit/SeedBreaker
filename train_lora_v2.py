#!/usr/bin/env python3
"""
train_lora_v2.py
----------------
SeedBreaker Phase 5 — LoRA fine-tuning for miRNA binding-site disruption.

Changes vs v1:
  - Auto-detects LoRA target modules (handles NT-v2 custom code + ESM-2)
  - All print() calls use flush=True → log file updates in real time
  - Step-level progress every 50 steps
  - MPS-safe: synchronize() after backward to surface GPU errors immediately
  - Saves checkpoint every epoch (not just best)
  - Resumes from last checkpoint if one exists
  - Tries NT-v2 first (already cached); falls back to ESM-2 with clean download

Usage:
  # Standard overnight run
  python3 train_lora_v2.py \
    --data  training_data/training_data.parquet \
    --out   models/seedbreaker_lora_v2 \
    --epochs 10 --batch 8 --lr 2e-4

  # Quick smoke test (5k examples)
  python3 train_lora_v2.py \
    --data  training_data/training_data.parquet \
    --out   models/seedbreaker_lora_v2 \
    --epochs 2 --batch 8 --max-train 5000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                             get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    sys.exit("Missing deps: pip install transformers peft accelerate scikit-learn")


# ── Model preference order ─────────────────────────────────────────────────────

MODEL_CANDIDATES = [
    # NT-v2 (nucleotide model, already cached, better DNA representations)
    ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", True),
    # ESM-2 (protein model, standard HuggingFace, robust peft compat)
    ("facebook/esm2_t33_650M_UR50D", False),
]

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1

# Ordered list of candidate target-module sets to try
LORA_MODULE_CANDIDATES = [
    ["query", "key", "value", "dense"],          # standard ESM/BERT
    ["query", "key", "value"],                   # ESM without output dense
    ["q_proj", "k_proj", "v_proj", "out_proj"],  # LLaMA-style
    ["q_proj", "k_proj", "v_proj"],              # LLaMA without out_proj
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg):
    """Timestamped print with immediate flush."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def detect_lora_targets(model, candidates=LORA_MODULE_CANDIDATES):
    """
    Walk the model and return the first candidate list where ≥2 modules match.
    Prints a summary of attention-related modules found to aid debugging.
    """
    all_names = set(name.split(".")[-1] for name, _ in model.named_modules())
    attn_names = [n for n in all_names
                  if any(k in n.lower() for k in ("query","key","value","proj","attn","dense"))]
    log(f"  Leaf module names (attention-related): {sorted(attn_names)}")

    for cand in candidates:
        hits = [m for m in cand if m in all_names]
        if len(hits) >= 2:
            log(f"  Auto-detected LoRA targets: {cand}  (matched: {hits})")
            return cand

    log("  WARNING: Could not auto-detect LoRA targets; falling back to ['query','key','value']")
    return ["query", "key", "value"]


def load_model_and_tokenizer(prefer_offline: bool = True):
    """
    Try MODEL_CANDIDATES in order. Returns (model, tokenizer, model_name).
    Tries offline cache first for each; if that fails tries download.
    """
    errors = []
    for model_name, trust_rc in MODEL_CANDIDATES:
        for offline in ([True, False] if prefer_offline else [False]):
            try:
                tag = "OFFLINE" if offline else "ONLINE"
                log(f"  Trying {model_name} [{tag}] ...")
                env_val = "1" if offline else "0"
                os.environ["TRANSFORMERS_OFFLINE"] = env_val

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_rc,
                )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    ignore_mismatched_sizes=True,
                    trust_remote_code=trust_rc,
                )
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                log(f"  Loaded: {model_name}")
                return base_model, tokenizer, model_name
            except Exception as e:
                errors.append(f"{model_name} [{tag}]: {e}")
                continue

    log("FATAL: No model loaded. Errors:")
    for e in errors:
        log(f"  {e}")
    sys.exit(1)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SeedDisruptionDataset(Dataset):
    """
    REF window + 'NNN' + ALT window, concatenated as single sequence.
    ESM-style tokenizers do not support sequence pairs.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        combined = row["ref_window"] + "NNN" + row["alt_window"]
        enc = self.tokenizer(
            combined,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, prefix=""):
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = (probs >= 0.5).astype(int)
    acc    = (preds == labels).mean()
    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except ValueError:
        auroc = auprc = float("nan")
    tag = f"{prefix}_" if prefix else ""
    return {
        f"{tag}auroc":    round(float(auroc), 4),
        f"{tag}auprc":    round(float(auprc), 4),
        f"{tag}accuracy": round(float(acc),   4),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    log("=" * 60)
    log("SeedBreaker LoRA Training v2")
    log("=" * 60)

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log(f"Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        log("Device: CPU (will be slow)")

    # ── Load data ────────────────────────────────────────────────────────────
    log(f"\nLoading training data: {args.data}")
    df = pd.read_parquet(args.data)
    log(f"  Total rows:    {len(df):,}")
    log(f"  Label balance: {df['label'].mean()*100:.1f}% disruptive")
    log(f"  Splits:        {df['split'].value_counts().to_dict()}")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(args.max_train, random_state=42)
        log(f"  Training subsampled to {len(train_df):,}")

    # ── Model + tokenizer ────────────────────────────────────────────────────
    log("\nLoading model and tokenizer...")
    base_model, tokenizer, model_name = load_model_and_tokenizer(prefer_offline=True)

    # ── Datasets + loaders ───────────────────────────────────────────────────
    train_ds = SeedDisruptionDataset(train_df, tokenizer, args.max_length)
    val_ds   = SeedDisruptionDataset(val_df,   tokenizer, args.max_length)
    test_ds  = SeedDisruptionDataset(test_df,  tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch * 2,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch * 2,
                              shuffle=False, num_workers=0)

    log(f"\nDataset sizes: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    log("\nApplying LoRA adapters...")
    target_modules = detect_lora_targets(base_model)

    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = target_modules,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.SEQ_CLS,
    )

    try:
        model = get_peft_model(base_model, lora_config)
    except Exception as e:
        log(f"FATAL: get_peft_model failed: {e}")
        log("  Try inspecting module names above and adjusting LORA_MODULE_CANDIDATES.")
        sys.exit(1)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    if trainable == 0:
        log("FATAL: No trainable parameters found — LoRA module names did not match.")
        log("  Detected module names:")
        for name, _ in model.named_modules():
            if any(k in name.lower() for k in ("query","key","value","proj","attn","dense")):
                log(f"    {name}")
        sys.exit(1)

    model = model.to(device)

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.06)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Class weights (handles 3:1 imbalance in training set)
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    cw    = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)

    # ── Output directory + checkpoint resume ──────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch    = 1
    best_val_auroc = 0.0
    history        = []

    resume_path = out_dir / "latest"
    if resume_path.exists():
        log(f"\nResuming from checkpoint: {resume_path}")
        try:
            results_path = out_dir / "training_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    prev = json.load(f)
                history        = prev.get("history", [])
                best_val_auroc = prev.get("best_val_auroc", 0.0)
                start_epoch    = len(history) + 1
                log(f"  Resuming from epoch {start_epoch}, best val AUROC={best_val_auroc:.4f}")
        except Exception as e:
            log(f"  Could not parse checkpoint metadata ({e}); starting fresh.")

    # ── Training config summary ───────────────────────────────────────────────
    log(f"\nTraining config:")
    log(f"  Model:           {model_name}")
    log(f"  LoRA targets:    {target_modules}")
    log(f"  Epochs:          {args.epochs}  (start: {start_epoch})")
    log(f"  Batch size:      {args.batch}  (effective: {args.batch * args.grad_accum})")
    log(f"  Grad accum:      {args.grad_accum}")
    log(f"  Learning rate:   {args.lr}")
    log(f"  Max seq length:  {args.max_length}")
    log(f"  Total steps:     {total_steps:,}")
    log(f"  Warmup steps:    {warmup_steps:,}")
    log(f"  Steps/epoch:     {len(train_loader):,}")
    log(f"  Output dir:      {out_dir}")
    log("")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labs) / args.grad_accum
            loss.backward()

            # Synchronize MPS after backward — surfaces hardware errors immediately
            # instead of silently crashing later
            if device.type == "mps":
                torch.mps.synchronize()

            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Step-level progress every 50 steps
            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                its     = (step + 1) / elapsed
                avg     = epoch_loss / (step + 1)
                log(f"  Ep {epoch:02d}  step {step+1:>6,}/{len(train_loader):,}  "
                    f"loss={avg:.4f}  speed={its:.2f}it/s  "
                    f"eta={((len(train_loader)-step-1)/its)/60:.0f}min")

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_labels, val_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out  = model(input_ids=ids, attention_mask=mask)
                probs = torch.softmax(out.logits, dim=-1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch["labels"].numpy())

        metrics = compute_metrics(val_labels, val_probs, prefix="val")

        log(f"\nEpoch {epoch:02d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_auroc={metrics['val_auroc']:.4f}  "
            f"val_auprc={metrics['val_auprc']:.4f}  "
            f"val_acc={metrics['val_accuracy']:.4f}  "
            f"time={epoch_time/60:.1f}min")

        history.append({"epoch": epoch, "train_loss": avg_loss,
                        "epoch_time_min": round(epoch_time/60, 1), **metrics})

        # Save best
        if metrics["val_auroc"] > best_val_auroc:
            best_val_auroc = metrics["val_auroc"]
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            log(f"  → Best model saved (val_auroc={best_val_auroc:.4f})")

        # Save latest (for resume)
        model.save_pretrained(out_dir / "latest")
        tokenizer.save_pretrained(out_dir / "latest")

        # Checkpoint every epoch
        ckpt_dir = out_dir / f"epoch_{epoch:02d}"
        model.save_pretrained(ckpt_dir)
        log(f"  → Checkpoint saved: {ckpt_dir.name}/")

        # Rolling save of results so far
        results_so_far = {
            "base_model":    model_name,
            "lora_r":        LORA_R,
            "lora_alpha":    LORA_ALPHA,
            "lora_targets":  target_modules,
            "epochs_done":   epoch,
            "epochs_total":  args.epochs,
            "best_val_auroc":best_val_auroc,
            "history":       history,
        }
        with open(out_dir / "training_results.json", "w") as f:
            json.dump(results_so_far, f, indent=2)

    # ── Test evaluation ─────────────────────────────────────────────────────
    log("\nEvaluating on held-out test set (chr8 + chr18)...")

    best_base = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        trust_remote_code=(model_name == MODEL_CANDIDATES[0][0]),
    )
    best_model = PeftModel.from_pretrained(best_base, out_dir / "best").to(device)
    best_model.eval()

    test_labels, test_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = best_model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(batch["labels"].numpy())

    test_metrics = compute_metrics(test_labels, test_probs, prefix="test")

    log(f"\n{'='*55}")
    log("FINAL TEST RESULTS (chr8 + chr18)")
    log(f"{'='*55}")
    for k, v in test_metrics.items():
        log(f"  {k:<25} {v:.4f}")
    log(f"  Best val AUROC:          {best_val_auroc:.4f}")
    log(f"{'='*55}")

    # Final save
    final_results = {
        "base_model":    model_name,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "lora_targets":  target_modules,
        "epochs":        args.epochs,
        "batch_size":    args.batch,
        "lr":            args.lr,
        "best_val_auroc":best_val_auroc,
        "test_metrics":  test_metrics,
        "history":       history,
    }
    with open(out_dir / "training_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    model.save_pretrained(out_dir / "final")
    tokenizer.save_pretrained(out_dir / "final")

    log(f"\nAll artifacts saved to: {out_dir}/")
    log("  best/                  — best validation checkpoint")
    log("  final/                 — final epoch checkpoint")
    log("  epoch_NN/              — per-epoch checkpoints")
    log("  training_results.json  — full metrics and history")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",       required=True)
    ap.add_argument("--out",        default="models/seedbreaker_lora_v2")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch",      type=int,   default=8)
    ap.add_argument("--grad-accum", type=int,   default=4)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--max-length", type=int,   default=256)
    ap.add_argument("--max-train",  type=int,   default=None)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Training data not found: {args.data}")

    train(args)


if __name__ == "__main__":
    main()
