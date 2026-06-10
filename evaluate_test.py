#!/usr/bin/env python3
"""
evaluate_test.py — SeedBreaker held-out test set evaluation
Evaluates best checkpoint (ESM-2 t33 650M base + LoRA) on chr8 + chr18
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL  = "facebook/esm2_t33_650M_UR50D"   # what training actually used
MODEL_PATH  = "models/seedbreaker_lora_v2/best"
DATA_PATH   = "training_data/training_data.parquet"
BATCH_SIZE  = 16
MAX_LENGTH  = 256

# ── Setup ──────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

df = pd.read_parquet(DATA_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"Test set: {len(test_df):,} examples ({(test_df['label']==1).sum():,} disruptive, {(test_df['label']==0).sum():,} preserving)")

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Loading base model ({BASE_MODEL})...")
cfg = AutoConfig.from_pretrained(BASE_MODEL)
cfg.num_labels = 2
print(f"  hidden_size={cfg.hidden_size}  num_layers={cfg.num_hidden_layers}  intermediate={cfg.intermediate_size}")

base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    config=cfg,
    ignore_mismatched_sizes=True,
)

print(f"Loading LoRA adapter from {MODEL_PATH}...")
model = PeftModel.from_pretrained(base, MODEL_PATH).to(device)
model.eval()
print("Model ready.\n")

# ── Inference ──────────────────────────────────────────────────────────────────
probs_all, labels_all = [], []
t0 = time.time()
n = len(test_df)

for i in range(0, n, BATCH_SIZE):
    batch = test_df.iloc[i:i+BATCH_SIZE]
    seqs = [r.ref_window + "NNN" + r.alt_window for _, r in batch.iterrows()]
    enc = tokenizer(
        seqs,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
    probs_all.extend(probs)
    labels_all.extend(batch["label"].tolist())

    if (i // BATCH_SIZE) % 100 == 0:
        done = min(i + BATCH_SIZE, n)
        elapsed = time.time() - t0
        eta = elapsed / max(done, 1) * (n - done)
        print(f"  {done:>6,}/{n:,}  ({done/n*100:.1f}%)  eta={eta/60:.1f}min", flush=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
labels_all = np.array(labels_all)
probs_all  = np.array(probs_all)
preds      = (probs_all >= 0.5).astype(int)

auroc = roc_auc_score(labels_all, probs_all)
auprc = average_precision_score(labels_all, probs_all)
acc   = (preds == labels_all).mean()

tp = ((preds == 1) & (labels_all == 1)).sum()
fp = ((preds == 1) & (labels_all == 0)).sum()
tn = ((preds == 0) & (labels_all == 0)).sum()
fn = ((preds == 0) & (labels_all == 1)).sum()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

elapsed = time.time() - t0
print(f"\n{'='*50}")
print("TEST SET RESULTS  (chr8 + chr18 held-out)")
print(f"{'='*50}")
print(f"  AUROC        {auroc:.4f}")
print(f"  AUPRC        {auprc:.4f}")
print(f"  Accuracy     {acc:.4f}")
print(f"  Sensitivity  {sensitivity:.4f}  (recall / TPR)")
print(f"  Specificity  {specificity:.4f}  (TNR)")
print(f"  PPV          {ppv:.4f}  (precision)")
print(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
print(f"{'='*50}")
print(f"Eval time: {elapsed/60:.1f} min")
