#!/usr/bin/env python3
"""
enrichment_analysis.py — Phase 7: SeedBreaker validation enrichment tests
==========================================================================

Tests:
  1. GTEx eQTL enrichment — are HIGH-tier SeedBreaker calls enriched among eQTLs?
  2. Score distribution comparison — eQTLs vs. gnomAD background
  3. ClinVar gold-standard — do known pathogenic 3'UTR variants score HIGH?

Usage:
  python3 enrichment_analysis.py \\
    --gtex    results/validation/gtex_eqtl_scored.tsv \\
    --gnomad  results/validation/gnomad_background_scored.tsv \\
    --clinvar results/validation/clinvar_scored.tsv \\
    --out     results/validation/enrichment_results.txt
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def load(path: str, label: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        print(f"  WARNING: {label} file not found: {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    print(f"  Loaded {label}: {len(df):,} rows")
    return df


def tier_counts(df: pd.DataFrame) -> dict:
    return df["tier"].value_counts().to_dict()


def enrichment_fisher(df_pos: pd.DataFrame, df_neg: pd.DataFrame,
                      threshold: float = 0.8) -> tuple:
    """
    Fisher's exact test: is combined_score >= threshold enriched in eQTLs (pos)
    vs background (neg)?
    """
    pos_hi = (df_pos["combined_score"] >= threshold).sum()
    pos_lo = (df_pos["combined_score"] <  threshold).sum()
    neg_hi = (df_neg["combined_score"] >= threshold).sum()
    neg_lo = (df_neg["combined_score"] <  threshold).sum()

    table = [[pos_hi, pos_lo], [neg_hi, neg_lo]]
    oddsratio, pvalue = stats.fisher_exact(table, alternative="greater")
    return pos_hi, pos_lo, neg_hi, neg_lo, oddsratio, pvalue


def mwu_test(df_pos: pd.DataFrame, df_neg: pd.DataFrame) -> tuple:
    """Mann-Whitney U: are eQTL combined_scores stochastically greater?"""
    return stats.mannwhitneyu(
        df_pos["combined_score"].dropna(),
        df_neg["combined_score"].dropna(),
        alternative="greater"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtex",    required=True)
    ap.add_argument("--gnomad",  required=True)
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Combined score cutoff for HIGH tier [0.8]")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 65)
    p("SEEDBREAKER PHASE 7 — VALIDATION ENRICHMENT ANALYSIS")
    p("=" * 65)

    print("\nLoading scored datasets...")
    gtex    = load(args.gtex,    "GTEx eQTL")
    gnomad  = load(args.gnomad,  "gnomAD background")
    clinvar = load(args.clinvar, "ClinVar")

    # ── 1. ClinVar gold standard ───────────────────────────────────────────────
    p("\n[1] ClinVar Pathogenic 3'UTR × ENCORI — Gold Standard")
    p("-" * 50)
    if clinvar is not None and len(clinvar) > 0:
        tc = tier_counts(clinvar)
        total = len(clinvar)
        p(f"  Total scoreable pathogenic variants: {total}")
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            n = tc.get(tier, 0)
            p(f"  {tier:6s}: {n:4d}  ({n/total*100:.1f}%)")

        hi_frac = tc.get("HIGH", 0) / total
        p(f"\n  Fraction HIGH-tier: {hi_frac:.3f}")
        p(f"  (Expected under null ~0.05–0.10 for random variants)")

        # Score stats
        p(f"\n  Combined score stats:")
        p(f"    Median : {clinvar['combined_score'].median():.4f}")
        p(f"    Mean   : {clinvar['combined_score'].mean():.4f}")
        p(f"    ≥ 0.8  : {(clinvar['combined_score'] >= 0.8).sum()} / {total}")

        # Top hits
        top = clinvar.nlargest(10, "combined_score")[
            ["variant_id", "chrom", "pos", "ref", "alt",
             "miRNA", "gene", "combined_score", "tier"]
        ]
        p("\n  Top 10 highest-scoring ClinVar variants:")
        p(top.to_string(index=False))
    else:
        p("  No ClinVar data available.")

    # ── 2. GTEx × gnomAD enrichment ───────────────────────────────────────────
    p("\n[2] GTEx eQTL Enrichment vs. gnomAD Background")
    p("-" * 50)
    if gtex is not None and gnomad is not None and len(gtex) > 0 and len(gnomad) > 0:
        p(f"  GTEx eQTL variants (in ENCORI sites): {len(gtex):,}")
        p(f"  gnomAD background variants:           {len(gnomad):,}")

        # Fisher's exact
        ph, pl, nh, nl, OR, pval = enrichment_fisher(
            gtex, gnomad, threshold=args.threshold
        )
        p(f"\n  Fisher's exact test (combined_score ≥ {args.threshold}):")
        p(f"    eQTL   HIGH={ph:,}  LOW={pl:,}  ({ph/(ph+pl)*100:.1f}% HIGH)")
        p(f"    gnomAD HIGH={nh:,}  LOW={nl:,}  ({nh/(nh+nl)*100:.1f}% HIGH)")
        p(f"    Odds ratio : {OR:.3f}")
        p(f"    p-value    : {pval:.2e}  {'*** SIGNIFICANT' if pval < 0.05 else ''}")

        # Mann-Whitney U
        stat, pval_mwu = mwu_test(gtex, gnomad)
        p(f"\n  Mann-Whitney U (eQTL scores > background):")
        p(f"    U statistic : {stat:.0f}")
        p(f"    p-value     : {pval_mwu:.2e}  {'*** SIGNIFICANT' if pval_mwu < 0.05 else ''}")

        # Score distributions
        p(f"\n  Score distribution (combined_score):")
        for label, df in [("eQTL", gtex), ("gnomAD", gnomad)]:
            p(f"    {label:6s}  median={df['combined_score'].median():.4f}  "
              f"mean={df['combined_score'].mean():.4f}  "
              f"std={df['combined_score'].std():.4f}")

        # Tier breakdown
        p(f"\n  Tier breakdown:")
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            n_e = (gtex["tier"]   == tier).sum()
            n_g = (gnomad["tier"] == tier).sum()
            p(f"    {tier:6s}  eQTL: {n_e:5,} ({n_e/len(gtex)*100:.1f}%)   "
              f"gnomAD: {n_g:6,} ({n_g/len(gnomad)*100:.1f}%)")

        # Per-miRNA enrichment (top disrupted miRNAs in eQTLs)
        p(f"\n  Top miRNAs by HIGH-tier eQTL count:")
        top_mi = gtex[gtex["tier"]=="HIGH"]["miRNA"].value_counts().head(10)
        p(top_mi.to_string())

    else:
        p("  Insufficient data for enrichment test.")

    # ── 3. Score correlation with eQTL effect size ────────────────────────────
    # GTEx variant ID encodes chr_pos_ref_alt — no direct effect size here
    # (effect sizes would need the per-tissue files, noted for future work)
    p("\n[3] Notes for paper")
    p("-" * 50)
    p("  Effect size correlation (SeedBreaker score vs GTEx beta) requires")
    p("  per-tissue eQTL association files. Download from GTEx portal if needed.")
    p("  The Fisher's exact and MWU tests above are sufficient for the paper.")

    p("\n" + "=" * 65)

    # Write to file
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
