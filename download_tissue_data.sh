#!/usr/bin/env bash
# =============================================================================
# download_tissue_data.sh — Download GTEx expression matrices for
#                           tissue-specific miRNA binding weight analysis
# =============================================================================
# Downloads two files:
#   1. GTEx gene median TPM      (~120 MB)  — 54 tissues × ~56k genes
#   2. GTEx miRNA TPM matrix     (~207 MB)  — 54 tissues × miRNAs (normalised)
#
# Usage:
#   conda activate seedbreaker
#   cd /Users/josh/Desktop/Projects/SeedBreaker
#   bash download_tissue_data.sh
#
# After download, run:
#   python3 build_tissue_weights.py
# =============================================================================

set -uo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$PROJ/Pre_data/tissue"
mkdir -p "$OUTDIR"

echo "=== GTEx tissue expression download: $(date) ==="

# ─── Gene median TPM ─────────────────────────────────────────────────────────
GENE_OUT="$OUTDIR/gtex_v11_gene_median_tpm.gct.gz"
GENE_URL="https://storage.googleapis.com/adult-gtex/bulk-tissue-expression/v11/rna-seq/GTEx_Analysis_v11_RNASeq_RSEMv1.3.3_gene_median_tpm.gct.gz"

if [ -f "$GENE_OUT" ]; then
    echo "[1/2] Gene TPM already downloaded — skipping"
else
    echo "[1/2] Downloading gene median TPM matrix (~120 MB)..."
    if wget -c -q --show-progress "$GENE_URL" -O "$GENE_OUT"; then
        echo "  OK: $GENE_OUT"
    else
        echo ""
        echo "  ERROR: Download failed. Possible causes:"
        echo "  - URL may have changed. Find the current link at:"
        echo "    https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression"
        echo "  - Look for: 'Gene TPM' under RNA-Seq section (v11)"
        echo "  - Expected filename: GTEx_Analysis_v11_RNASeq_RSEMv1.3.3_gene_median_tpm.gct.gz"
        rm -f "$GENE_OUT"
    fi
fi

# ─── miRNA TPM matrix (normalised, portal version) ───────────────────────────
# File: miRNA_TPM_matrix_PORTAL_2025_03_17.txt.gz
# Format: tab-delimited .txt.gz (NOT GCT). Columns: miRNA name, then one
# column per tissue with median TPM values. Use this — NOT the raw read counts.
MIRNA_OUT="$OUTDIR/gtex_mirna_tpm_matrix.txt.gz"
MIRNA_URL="https://storage.googleapis.com/adult-gtex/bulk-tissue-expression/v10/small-rna-seq/miRNA_TPM_matrix_PORTAL_2025_03_17.txt.gz"

if [ -f "$MIRNA_OUT" ]; then
    echo "[2/2] miRNA TPM already downloaded — skipping"
else
    echo "[2/2] Downloading miRNA TPM matrix (~207 MB)..."
    if wget -c -q --show-progress "$MIRNA_URL" -O "$MIRNA_OUT"; then
        echo "  OK: $MIRNA_OUT"
    else
        echo ""
        echo "  ERROR: Download failed."
        echo "  If you already downloaded miRNA_TPM_matrix_PORTAL_2025_03_17.txt.gz"
        echo "  manually, copy it here:"
        echo "    cp ~/Downloads/miRNA_TPM_matrix_PORTAL_2025_03_17.txt.gz \\"
        echo "       Pre_data/tissue/gtex_mirna_tpm_matrix.txt.gz"
        rm -f "$MIRNA_OUT"
    fi
fi

# ─── Verify ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Download complete: $(date) ==="
for F in "$GENE_OUT" "$MIRNA_OUT"; do
    if [ -f "$F" ]; then
        SIZE=$(du -sh "$F" | cut -f1)
        echo "  ✓ $(basename $F)  ($SIZE)"
    else
        echo "  ✗ $(basename $F)  MISSING — see errors above"
    fi
done

echo ""
echo "Next step:"
echo "  python3 build_tissue_weights.py"
echo "  # Builds Pre_data/tissue/tissue_weights.tsv.gz (~2-5 min)"
