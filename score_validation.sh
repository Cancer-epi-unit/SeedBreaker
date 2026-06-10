#!/usr/bin/env bash
# =============================================================================
# score_validation.sh — Phase 7: Score all validation datasets with score.py
# =============================================================================
# Run AFTER download_validation_data.sh completes.
#
# Scores:
#   1. ClinVar pathogenic 3'UTR × ENCORI variants
#   2. GTEx v10 eQTL × ENCORI SNVs
#   3. gnomAD common 3'UTR SNVs (background)
#
# Then runs enrichment_analysis.py to test whether HIGH-tier variants
# are enriched among eQTLs vs. background.
#
# Usage:
#   conda activate seedbreaker
#   cd /Users/josh/Desktop/Projects/SeedBreaker
#   bash score_validation.sh
# =============================================================================

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
MODEL="$PROJ/models/seedbreaker_lora_v2/best"
SITES="$PROJ/Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
GENOME="$PROJ/Pre_data/Genome/GRCh38.primary_assembly.genome.fa"
SEEDS="$PROJ/Pre_data/miRBase/mirna_seeds_hsa.json"
RESULTS="$PROJ/results/validation"
LOG="$RESULTS/scoring_log.txt"

mkdir -p "$RESULTS"
exec > >(tee -a "$LOG") 2>&1
echo "=== Validation scoring started: $(date) ==="

SCORE="python3 $PROJ/score.py \
  --sites  $SITES \
  --genome $GENOME \
  --seeds  $SEEDS \
  --model  $MODEL \
  --no-mfe"


# ─── 1. ClinVar ───────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Scoring ClinVar pathogenic 3'UTR variants..."
VCF="$PROJ/Pre_data/ClinVar/clinvar_pathogenic_3utr_encori.vcf.gz"

if [ ! -f "$VCF" ]; then
    echo "  ERROR: $VCF not found. Run download_validation_data.sh first."
else
    $SCORE --vcf "$VCF" --out "$RESULTS/clinvar_scored.tsv"
    echo "  → Results: results/validation/clinvar_scored.tsv"
fi


# ─── 2. GTEx eQTLs ────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Scoring GTEx v10 eQTL × ENCORI SNVs..."
VCF="$PROJ/Pre_data/GTEx/gtex_v10_encori_snvs.vcf.gz"

if [ ! -f "$VCF" ]; then
    echo "  ERROR: $VCF not found. Run download_validation_data.sh first."
else
    $SCORE --vcf "$VCF" --out "$RESULTS/gtex_eqtl_scored.tsv"
    echo "  → Results: results/validation/gtex_eqtl_scored.tsv"
fi


# ─── 3. gnomAD background ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Scoring gnomAD background 3'UTR SNVs..."
echo "  (This may take several hours for the full genome)"
VCF="$PROJ/Pre_data/gnomAD/gnomad_v4_common_3utr_snvs.vcf.gz"

if [ ! -f "$VCF" ]; then
    echo "  ERROR: $VCF not found. Run download_validation_data.sh first."
else
    $SCORE --vcf "$VCF" --out "$RESULTS/gnomad_background_scored.tsv"
    echo "  → Results: results/validation/gnomad_background_scored.tsv"
fi


# ─── Enrichment analysis ──────────────────────────────────────────────────────
echo ""
echo "[4/4] Running enrichment analysis..."
python3 "$PROJ/enrichment_analysis.py" \
    --gtex    "$RESULTS/gtex_eqtl_scored.tsv" \
    --gnomad  "$RESULTS/gnomad_background_scored.tsv" \
    --clinvar "$RESULTS/clinvar_scored.tsv" \
    --out     "$RESULTS/enrichment_results.txt"

echo ""
echo "=== Validation complete: $(date) ==="
echo "Results in: results/validation/"
