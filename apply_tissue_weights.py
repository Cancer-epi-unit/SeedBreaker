#!/usr/bin/env python3
"""
apply_tissue_weights.py — Apply tissue-specific weights to SeedBreaker scores
==============================================================================
Post-processes score.py output by joining tissue co-expression weights,
computing tissue-adjusted scores, and re-tiering variants.

The tissue_score reflects both the disruption probability AND whether the
miRNA binding site is biologically active in the target tissue.

Formula:
  tissue_score = combined_score × coexp_weight

  Where coexp_weight ∈ [0, 1]:
    ~0   → site not active in this tissue (gene/miRNA not co-expressed)
    ~0.5 → moderately expressed
    ~1   → both highly expressed → full weight to the disruption score

Tier thresholds (same as combined_score):
  HIGH   ≥ 0.8
  MEDIUM ≥ 0.5
  LOW    <  0.5

Usage:
    conda activate seedbreaker
    cd /Users/josh/Desktop/Projects/SeedBreaker

    # Score a VCF first (if not already done)
    python3 score.py --vcf my_variants.vcf.gz ... --out results/scored.tsv

    # Apply breast tissue weights
    python3 apply_tissue_weights.py \\
        --scores  results/scored.tsv \\
        --weights Pre_data/tissue/tissue_weights.tsv.gz \\
        --tissue  breast \\
        --out     results/scored_breast.tsv

    # Apply prostate tissue weights
    python3 apply_tissue_weights.py \\
        --scores  results/scored.tsv \\
        --weights Pre_data/tissue/tissue_weights.tsv.gz \\
        --tissue  prostate \\
        --out     results/scored_prostate.tsv

    # Show all available tissue labels in the weights file
    python3 apply_tissue_weights.py --list-tissues

Output TSV adds 4 columns after the existing 17:
  tissue          - tissue label used
  coexp_weight    - co-expression weight (0-1); NA if site not in weights file
  tissue_score    - combined_score × coexp_weight (NA if no weight)
  tissue_tier     - HIGH/MEDIUM/LOW based on tissue_score (UNKNOWN if no data)

Rows where coexp_weight=NA (binding site not in GTEx or gene not expressed)
are retained with tissue_score=NA and tissue_tier=UNKNOWN for traceability.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# score.py output column names (17 columns)
SCORE_COLS = [
    "variant_id", "chrom", "pos", "ref", "alt", "af",
    "site_id", "mirna", "gene", "strand", "site_type",
    "seed_pos", "wc_disrupted", "delta_mfe",
    "model_score", "combined_score", "tier",
]

# Tier thresholds
HIGH_THRESH   = 0.8
MEDIUM_THRESH = 0.5


def assign_tier(score):
    if pd.isna(score):
        return "UNKNOWN"
    if score >= HIGH_THRESH:
        return "HIGH"
    if score >= MEDIUM_THRESH:
        return "MEDIUM"
    return "LOW"


def list_tissues(weights_path: Path):
    print(f"Reading {weights_path}...")
    df = pd.read_csv(weights_path, sep="\t", usecols=["tissue"])
    tissues = sorted(df["tissue"].unique())
    print(f"\nAvailable tissue labels ({len(tissues)}):")
    for t in tissues:
        print(f"  {t}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    ap.add_argument("--scores",  type=Path,
                    help="score.py output TSV (or .tsv.gz)")
    ap.add_argument("--weights", type=Path,
                    default=Path("Pre_data/tissue/tissue_weights.tsv.gz"),
                    help="Tissue weights file from build_tissue_weights.py "
                         "(default: Pre_data/tissue/tissue_weights.tsv.gz)")
    ap.add_argument("--tissue",  type=str,
                    help="Tissue label to apply (e.g. breast, prostate). "
                         "Run --list-tissues to see options.")
    ap.add_argument("--out",     type=Path,
                    help="Output TSV path")
    ap.add_argument("--list-tissues", action="store_true",
                    help="Print available tissue labels and exit")
    ap.add_argument("--high-only", action="store_true",
                    help="Only write HIGH tissue_tier variants to output")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override HIGH threshold (default: 0.8)")
    args = ap.parse_args()

    # Override threshold
    global HIGH_THRESH, MEDIUM_THRESH
    if args.threshold:
        HIGH_THRESH = args.threshold

    # ── List tissues mode ─────────────────────────────────────────────────────
    if args.list_tissues:
        if not args.weights.exists():
            sys.exit(f"ERROR: {args.weights} not found.\n"
                     f"Run: python3 build_tissue_weights.py")
        list_tissues(args.weights)
        return

    # ── Validate args ─────────────────────────────────────────────────────────
    if not args.scores:
        ap.error("--scores is required (unless using --list-tissues)")
    if not args.tissue:
        ap.error("--tissue is required (e.g. --tissue breast). "
                 "Use --list-tissues to see options.")
    if not args.out:
        stem = args.scores.stem.replace(".tsv", "")
        args.out = args.scores.parent / f"{stem}_{args.tissue}.tsv"
        print(f"No --out specified. Writing to: {args.out}")

    if not args.scores.exists():
        sys.exit(f"ERROR: {args.scores} not found.")
    if not args.weights.exists():
        sys.exit(f"ERROR: {args.weights} not found.\n"
                 f"Run: python3 build_tissue_weights.py")

    # ── Load scored variants ──────────────────────────────────────────────────
    print(f"[1/4] Loading scored variants: {args.scores}...", flush=True)
    scores = pd.read_csv(args.scores, sep="\t", header=0)
    # Rename columns if needed (handle files that might have different col counts)
    if len(scores.columns) >= len(SCORE_COLS):
        scores.columns = list(SCORE_COLS) + list(scores.columns[len(SCORE_COLS):])
    print(f"  {len(scores):,} variants loaded", flush=True)

    # ── Load tissue weights for requested tissue ───────────────────────────────
    print(f"[2/4] Loading tissue weights ({args.tissue})...", flush=True)
    weights = pd.read_csv(
        args.weights, sep="\t",
        usecols=["site_id", "mirna", "tissue", "gene_tpm", "mirna_val", "coexp_weight"]
    )
    weights = weights[weights["tissue"] == args.tissue]
    if weights.empty:
        available = pd.read_csv(args.weights, sep="\t", usecols=["tissue"])["tissue"].unique()
        sys.exit(f"ERROR: Tissue '{args.tissue}' not found in weights file.\n"
                 f"Available: {sorted(available)}\n"
                 f"Use --list-tissues to see all options.")
    print(f"  {len(weights):,} site-tissue records for '{args.tissue}'", flush=True)

    # ── Join ──────────────────────────────────────────────────────────────────
    print(f"[3/4] Joining weights to scores...", flush=True)
    # Join on site_id (each ENCORI site is unique to one miRNA)
    merged = scores.merge(
        weights[["site_id", "gene_tpm", "mirna_val", "coexp_weight"]],
        on="site_id",
        how="left"
    )

    # Compute tissue score and tier
    merged["tissue"]        = args.tissue
    merged["coexp_weight"]  = merged["coexp_weight"].round(4)
    merged["tissue_score"]  = (
        merged["combined_score"] * merged["coexp_weight"]
    ).round(4)
    merged["tissue_tier"]   = merged["tissue_score"].apply(assign_tier)

    # Reorder columns: insert tissue cols after existing 17
    col_order = list(SCORE_COLS) + ["tissue", "coexp_weight", "tissue_score", "tissue_tier"]
    # keep any extra cols at end
    extra = [c for c in merged.columns if c not in col_order]
    merged = merged[col_order + extra]

    # ── Filter if requested ───────────────────────────────────────────────────
    if args.high_only:
        merged = merged[merged["tissue_tier"] == "HIGH"]
        print(f"  --high-only: {len(merged):,} HIGH tissue_tier variants retained",
              flush=True)

    # ── Write ─────────────────────────────────────────────────────────────────
    print(f"[4/4] Writing output...", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, sep="\t", index=False,
                  compression="gzip" if str(args.out).endswith(".gz") else None)

    # ── Summary ───────────────────────────────────────────────────────────────
    total     = len(merged)
    n_weight  = merged["coexp_weight"].notna().sum()
    n_high    = (merged["tissue_tier"] == "HIGH").sum()
    n_medium  = (merged["tissue_tier"] == "MEDIUM").sum()
    n_low     = (merged["tissue_tier"] == "LOW").sum()
    n_unknown = (merged["tissue_tier"] == "UNKNOWN").sum()
    mean_w    = merged["coexp_weight"].mean()

    print(f"\n{'='*60}", flush=True)
    print(f"Tissue: {args.tissue}", flush=True)
    print(f"  Total variants:        {total:,}", flush=True)
    print(f"  With tissue weight:    {n_weight:,} ({100*n_weight/max(total,1):.1f}%)", flush=True)
    print(f"  Mean co-expr weight:   {mean_w:.3f}", flush=True)
    print(f"", flush=True)
    print(f"  Tissue-adjusted tiers:", flush=True)
    print(f"    HIGH    (≥{HIGH_THRESH}):  {n_high:,}  ({100*n_high/max(total,1):.1f}%)", flush=True)
    print(f"    MEDIUM  (≥{MEDIUM_THRESH}):  {n_medium:,}  ({100*n_medium/max(total,1):.1f}%)", flush=True)
    print(f"    LOW:          {n_low:,}  ({100*n_low/max(total,1):.1f}%)", flush=True)
    print(f"    UNKNOWN:      {n_unknown:,}  (no weight data)", flush=True)
    print(f"", flush=True)

    # Top hits
    top = merged[merged["tissue_tier"] == "HIGH"].nlargest(10, "tissue_score")
    if not top.empty:
        print(f"  Top {args.tissue}-specific HIGH variants:", flush=True)
        for _, r in top.iterrows():
            print(f"    {r['variant_id']:<35}  gene={r['gene']:<12}  "
                  f"miRNA={r['mirna']:<20}  "
                  f"tissue_score={r['tissue_score']:.4f}  "
                  f"weight={r['coexp_weight']:.3f}", flush=True)

    print(f"\n  Output: {args.out}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
