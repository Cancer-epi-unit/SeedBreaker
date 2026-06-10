#!/usr/bin/env python3
"""
build_tissue_weights.py — Build site×tissue co-expression weight matrix
========================================================================
For each ENCORI miRNA binding site, compute a tissue-specific co-expression
weight reflecting whether both the miRNA and its target gene are expressed
in that tissue.

Weight formula:
  gene_w   = min(log2(gene_tpm + 1) / log2(101), 1.0)   # saturates at 100 TPM
  mirna_w  = min(log2(mirna_rpm + 1) / log2(11),  1.0)   # saturates at 10 RPM
  weight   = sqrt(gene_w * mirna_w)                       # geometric mean

Weights range 0–1:
  ~0.0  → one or both not expressed in this tissue (site not relevant)
  ~0.5  → moderate expression of both (moderate relevance)
  ~1.0  → both highly expressed (site highly relevant in this tissue)

Usage:
    conda activate seedbreaker
    cd /Users/josh/Desktop/Projects/SeedBreaker
    python3 build_tissue_weights.py

    # Use miRmine instead of GTEx small RNA:
    python3 build_tissue_weights.py --mirna-source mirmine

Output:
    Pre_data/tissue/tissue_weights.tsv.gz
    Pre_data/tissue/tissue_weights_summary.txt
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJ        = Path(__file__).parent
ENCORI      = PROJ / "Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
GENE_TPM    = PROJ / "Pre_data/tissue/gtex_v11_gene_median_tpm.gct.gz"
MIRNA_TPM   = PROJ / "Pre_data/tissue/gtex_mirna_tpm_matrix.txt.gz"   # miRNA_TPM_matrix_PORTAL_*.txt.gz
MIRMINE     = PROJ / "Pre_data/tissue/mirmine_hsa_expression.tsv.gz"
OUTDIR      = PROJ / "Pre_data/tissue"
OUTFILE     = OUTDIR / "tissue_weights.tsv.gz"

# Keyword → short label. Each keyword is matched as a substring of the GTEx
# column name (case-insensitive). Use the most distinctive word for each tissue
# to avoid false matches. Order matters: first match wins.
TISSUE_KEYWORDS = [
    ("mammary",    "breast"),      # "Breast - Mammary Tissue"
    ("breast",     "breast"),      # fallback if column is just "Breast"
    ("prostate",   "prostate"),
    ("sigmoid",    "colon_sigmoid"),
    ("transverse", "colon_transverse"),
    ("lung",       "lung"),
    ("liver",      "liver"),
    ("kidney",     "kidney"),
    ("ovary",      "ovary"),
    ("uterus",     "uterus"),
    ("bladder",    "bladder"),
    ("thyroid",    "thyroid"),
    ("skin",       "skin"),
    ("blood",      "blood"),
    ("cortex",     "brain"),       # "Brain - Cortex"
]

# Keep TISSUE_MAP as alias for downstream code that references it
TISSUE_MAP = {kw: label for kw, label in TISSUE_KEYWORDS}

# miRmine tissue label → short label (miRmine uses different naming)
MIRMINE_MAP = {
    "Breast":         "breast",
    "Prostate":       "prostate",
    "Lung":           "lung",
    "Colon":          "colon_sigmoid",
    "Liver":          "liver",
    "Kidney":         "kidney",
    "Ovary":          "ovary",
    "Uterus":         "uterus",
    "Bladder":        "bladder",
    "Thyroid":        "thyroid",
    "Skin":           "skin",
    "Blood":          "blood",
    "Brain":          "brain",
}


def parse_gct(path: Path, label: str) -> pd.DataFrame:
    """Parse GTEx GCT format. Returns DataFrame with gene/miRNA as index,
    tissues as columns."""
    print(f"  Reading {label} ({path.name})...", flush=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        # Skip first 2 metadata lines
        f.readline()  # #1.2
        f.readline()  # nrows  ncols
        df = pd.read_csv(f, sep="\t")
    # Column 0: Name (gene ID / miRNA ID), Column 1: Description (symbol)
    df = df.set_index("Description")
    df = df.drop(columns=["Name"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Deduplicate index (duplicate gene symbols → keep highest mean expression row)
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        df = df.groupby(df.index).mean()
        print(f"    (collapsed {n_dup} duplicate gene symbols by mean)", flush=True)
    print(f"    {df.shape[0]:,} features × {df.shape[1]} tissues", flush=True)
    return df


def parse_mirna_tpm(path: Path) -> pd.DataFrame:
    """Parse GTEx miRNA_TPM_matrix_PORTAL_*.txt.gz (plain TSV, not GCT).
    First column is miRNA name; remaining columns are tissues (median TPM)."""
    print(f"  Reading GTEx miRNA TPM ({path.name})...", flush=True)
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"    {df.shape[0]:,} miRNAs × {df.shape[1]} tissues", flush=True)
    return df


def parse_mirmine(path: Path) -> pd.DataFrame:
    """Parse miRmine TSV format. Returns DataFrame with miRNA as index."""
    print(f"  Reading miRmine ({path.name})...", flush=True)
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"    {df.shape[0]:,} miRNAs × {df.shape[1]} tissues", flush=True)
    return df


def weight(tpm: float, saturation: float) -> float:
    """Log-normalised expression weight, saturates at `saturation`."""
    return min(np.log2(tpm + 1) / np.log2(saturation + 1), 1.0)


def coexp_weight(gene_tpm: float, mirna_val: float) -> float:
    """Geometric mean of gene and miRNA expression weights."""
    gw = weight(gene_tpm, saturation=100)   # gene: saturates at 100 TPM
    mw = weight(mirna_val, saturation=10)   # miRNA: saturates at 10 RPM/CPM
    return float(np.sqrt(gw * mw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirna-source", choices=["gtex", "mirmine"], default="gtex",
                    help="miRNA expression source (default: gtex)")
    ap.add_argument("--min-weight", type=float, default=0.0,
                    help="Only write rows with weight >= this value (default: 0)")
    ap.add_argument("--tissues", nargs="+", default=None,
                    help="Restrict to specific tissue short labels "
                         "(e.g. breast prostate). Default: all in TISSUE_MAP")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ENCORI atlas ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading ENCORI atlas...", flush=True)
    encori_cols = ["chrom", "start", "end", "name", "score", "strand",
                   "mirna", "mimat", "gene_name", "gene_id",
                   "rbp", "clip_num", "targetscan", "tdmd_score",
                   "phylop", "site_type", "context_score"]
    encori = pd.read_csv(ENCORI, sep="\t", header=None)
    encori = encori.iloc[:, :17]
    encori.columns = encori_cols[:17]
    # Use the ENCORI BED name field as site_id — matches score.py output exactly
    # Format: "hsa-miR-205-5p:SAMD11:MIMAT0000266" (col 4 of BED)
    encori["site_id"] = encori["name"]
    print(f"  {len(encori):,} ENCORI sites loaded", flush=True)

    # Unique (mirna, gene_name) pairs
    pairs = encori[["site_id", "mirna", "gene_name"]].drop_duplicates()
    unique_genes  = set(pairs["gene_name"])
    unique_mirnas = set(pairs["mirna"])
    print(f"  {len(pairs):,} unique (site, miRNA) pairs | "
          f"{len(unique_genes):,} genes | {len(unique_mirnas):,} miRNAs",
          flush=True)

    # ── 2. Load gene expression ───────────────────────────────────────────────
    print(f"\n[2/4] Loading gene expression...", flush=True)
    if not GENE_TPM.exists():
        sys.exit(f"ERROR: {GENE_TPM} not found.\nRun: bash download_tissue_data.sh")
    gene_df = parse_gct(GENE_TPM, "GTEx gene TPM")

    # Map GTEx columns → short tissue labels using keyword matching
    def match_tissue(col: str) -> str | None:
        col_l = col.lower()
        for keyword, label in TISSUE_KEYWORDS:
            if keyword in col_l:
                return label
        return None

    tissue_col_map = {}
    for gtex_col in gene_df.columns:
        label = match_tissue(gtex_col)
        if label and label not in tissue_col_map.values():
            tissue_col_map[gtex_col] = label
    print(f"  Matched {len(tissue_col_map)} tissue columns in gene TPM matrix", flush=True)

    if not tissue_col_map:
        # Show available columns to help the user debug
        print("  Available tissue columns (first 20):")
        for c in list(gene_df.columns)[:20]:
            print(f"    '{c}'")
        sys.exit("ERROR: No tissue columns matched TISSUE_MAP. "
                 "Check GTEx file format and update TISSUE_MAP in this script.")

    # ── 3. Load miRNA expression ──────────────────────────────────────────────
    print(f"\n[3/4] Loading miRNA expression ({args.mirna_source})...", flush=True)

    if args.mirna_source == "mirmine":
        if not MIRMINE.exists():
            sys.exit(f"ERROR: {MIRMINE} not found.\n"
                     f"Download: wget 'http://guanfiles.dcmb.med.umich.edu/mirmine/hsa_expression.tsv.gz'"
                     f" -O {MIRMINE}")
        mirna_df = parse_mirmine(MIRMINE)
        mirna_tissue_map = {}
        for col in mirna_df.columns:
            for mirmine_name, short in MIRMINE_MAP.items():
                if mirmine_name.lower() in col.lower():
                    mirna_tissue_map[col] = short
                    break
    else:  # gtex — uses miRNA_TPM_matrix_PORTAL_*.txt.gz (plain TSV, normalised TPM)
        if not MIRNA_TPM.exists():
            print(f"  WARNING: {MIRNA_TPM} not found.")
            print("  Falling back to gene-expression-only weights (miRNA weight set to 1).")
            print("  To include miRNA expression: run bash download_tissue_data.sh")
            print("  Or copy manually: cp ~/Downloads/miRNA_TPM_matrix_PORTAL_*.txt.gz \\")
            print(f"                      {MIRNA_TPM}")
            mirna_df = None
            mirna_tissue_map = {}
        else:
            mirna_df = parse_mirna_tpm(MIRNA_TPM)
            mirna_tissue_map = {}
            for col in mirna_df.columns:
                label = match_tissue(col)
                if label and label not in mirna_tissue_map.values():
                    mirna_tissue_map[col] = label

    # ── 4. Compute weights ────────────────────────────────────────────────────
    print(f"\n[4/4] Computing co-expression weights...", flush=True)

    target_tissues = set(args.tissues) if args.tissues else set(TISSUE_MAP.values())
    # Intersect with tissues available in gene matrix
    available_tissues = set(tissue_col_map.values()) & target_tissues
    if not available_tissues:
        sys.exit("ERROR: No requested tissues found in gene expression matrix.")
    print(f"  Tissues to process: {sorted(available_tissues)}", flush=True)

    # Build reverse lookup: short_label → gene_df column name
    gene_tissue_cols  = {v: k for k, v in tissue_col_map.items()}
    mirna_tissue_cols = {v: k for k, v in mirna_tissue_map.items()} if mirna_tissue_map else {}

    rows = []
    n_pairs = len(pairs)
    for i, (_, row) in enumerate(pairs.iterrows()):
        site_id   = row["site_id"]
        mirna     = row["mirna"]
        gene_name = row["gene_name"]

        # Gene TPM lookup
        if gene_name not in gene_df.index:
            # Try case-insensitive
            matches = [g for g in gene_df.index if g.upper() == gene_name.upper()]
            gene_name_key = matches[0] if matches else None
        else:
            gene_name_key = gene_name

        # miRNA lookup
        if mirna_df is not None:
            if mirna not in mirna_df.index:
                matches = [m for m in mirna_df.index if m.lower() == mirna.lower()]
                mirna_key = matches[0] if matches else None
            else:
                mirna_key = mirna
        else:
            mirna_key = None

        for tissue in sorted(available_tissues):
            gene_col  = gene_tissue_cols.get(tissue)
            mirna_col = mirna_tissue_cols.get(tissue)

            # Gene TPM
            if gene_name_key and gene_col and gene_col in gene_df.columns:
                val = gene_df.loc[gene_name_key, gene_col]
                gene_tpm = float(val.iloc[0] if hasattr(val, "iloc") else val)
            else:
                gene_tpm = float("nan")

            # miRNA expression
            if mirna_key and mirna_col and mirna_col in mirna_df.columns:
                mirna_val = float(mirna_df.loc[mirna_key, mirna_col])
            elif mirna_df is None:
                mirna_val = float("nan")  # gene-only mode
            else:
                mirna_val = 0.0  # miRNA not found → not expressed

            # Co-expression weight
            if np.isnan(gene_tpm):
                w = float("nan")
            elif np.isnan(mirna_val):
                # gene-only mode: use gene weight alone
                w = weight(gene_tpm, saturation=100)
            else:
                w = coexp_weight(gene_tpm, mirna_val)

            if np.isnan(w) or w >= args.min_weight:
                rows.append({
                    "site_id":     site_id,
                    "mirna":       mirna,
                    "gene_name":   gene_name,
                    "tissue":      tissue,
                    "gene_tpm":    round(gene_tpm, 3) if not np.isnan(gene_tpm) else None,
                    "mirna_val":   round(mirna_val, 3) if (mirna_val is not None and not np.isnan(mirna_val)) else None,
                    "coexp_weight": round(w, 4) if not np.isnan(w) else None,
                })

        if (i + 1) % 50000 == 0:
            print(f"  {i+1:,}/{n_pairs:,} pairs processed...", flush=True)

    out_df = pd.DataFrame(rows)
    print(f"  {len(out_df):,} site×tissue records computed", flush=True)

    # Write
    out_df.to_csv(OUTFILE, sep="\t", index=False,
                  compression="gzip", na_rep="NA")
    print(f"\n{'='*60}", flush=True)
    print(f"Tissue weights written: {OUTFILE}", flush=True)
    print(f"  {len(out_df):,} rows, {out_df['tissue'].nunique()} tissues", flush=True)

    # Summary
    summary_path = OUTDIR / "tissue_weights_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Tissue-specific co-expression weight summary\n")
        f.write("=" * 50 + "\n\n")
        for tissue in sorted(out_df["tissue"].unique()):
            sub = out_df[out_df["tissue"] == tissue]["coexp_weight"].dropna()
            n_expressed = (sub > 0.2).sum()
            f.write(f"{tissue}:\n")
            f.write(f"  sites with weight>0.2: {n_expressed:,} / {len(sub):,} "
                    f"({100*n_expressed/max(len(sub),1):.1f}%)\n")
            f.write(f"  mean weight: {sub.mean():.3f}  median: {sub.median():.3f}\n\n")

    print(f"Summary: {summary_path}", flush=True)
    print(f"\nNext step:", flush=True)
    print(f"  python3 apply_tissue_weights.py \\", flush=True)
    print(f"    --scores results/validation/gtex_scored.tsv \\", flush=True)
    print(f"    --tissue breast \\", flush=True)
    print(f"    --out results/validation/gtex_scored_breast.tsv", flush=True)


if __name__ == "__main__":
    main()
