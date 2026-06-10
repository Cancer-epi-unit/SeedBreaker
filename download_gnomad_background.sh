#!/usr/bin/env bash
# =============================================================================
# download_gnomad_background.sh — Stream gnomAD v4.1 common SNVs at ENCORI sites
# =============================================================================
# Streams only the ENCORI binding site regions from gnomAD (never downloads
# full chr VCFs). Filters to AF > 0.01, SNVs only.
# Expected output: ~50k-200k variants, ~30-60 min total.
#
# Usage:
#   conda activate seedbreaker
#   cd /Users/josh/Desktop/Projects/SeedBreaker
#   bash download_gnomad_background.sh
# =============================================================================

set -uo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
ENCORI="$PROJ/Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
OUTDIR="$PROJ/Pre_data/gnomAD"
TMPDIR="$OUTDIR/tmp_chroms"
REGIONS_BED="$OUTDIR/encori_regions_merged.bed"
OUTVCF="$OUTDIR/gnomad_v4_common_encori_snvs.vcf.gz"
GNOMAD="https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/genomes"

mkdir -p "$OUTDIR" "$TMPDIR"

echo "=== gnomAD background download started: $(date) ==="

# Build merged ENCORI regions BED (collapse overlapping sites)
echo "[0] Building ENCORI target regions..."
gzip -dc "$ENCORI" | awk 'BEGIN{OFS="\t"}{print $1,$2,$3}' | \
    sort -k1,1 -k2,2n | bedtools merge -i stdin -d 2000 > "$REGIONS_BED"
N=$(wc -l < "$REGIONS_BED")
echo "  $N merged ENCORI regions (after 2kb merge distance)"

# Stream each chromosome
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 \
        chr20 chr21 chr22"

echo "[1] Streaming gnomAD v4.1 per chromosome..."
for CHR in $CHROMS; do
    OUTF="$TMPDIR/${CHR}.vcf.gz"
    if [ -f "$OUTF" ]; then
        N=$(bcftools view -H "$OUTF" | wc -l)
        echo "  $CHR — already done ($N variants), skipping"
        continue
    fi

    echo "  $CHR streaming..."
    URL="${GNOMAD}/gnomad.genomes.v4.1.sites.${CHR}.vcf.bgz"

    bcftools view "$URL" \
        --regions-file "$REGIONS_BED" \
        --include 'AF>0.01 && TYPE="snp"' \
        -O z -o "$OUTF" 2>/dev/null

    if [ -f "$OUTF" ] && [ -s "$OUTF" ]; then
        if bcftools index -t "$OUTF" 2>/dev/null; then
            N=$(bcftools view -H "$OUTF" 2>/dev/null | wc -l)
            echo "  $CHR done — $N variants"
        else
            echo "  $CHR WARNING: index failed, file likely corrupt — removing"
            rm -f "$OUTF" "${OUTF}.tbi"
        fi
    else
        echo "  $CHR WARNING: empty or missing output — skipping"
        rm -f "$OUTF"
    fi
done

# Merge all chromosomes
echo ""
echo "[2] Merging chromosomes..."
FILES=$(find "$TMPDIR" -name "chr*.vcf.gz" | sort -V)
N_FILES=$(echo "$FILES" | wc -l)
echo "  Merging $N_FILES chromosome files..."

bcftools concat --allow-overlaps $FILES \
    -O z -o "$OUTVCF"
bcftools index -t "$OUTVCF"

TOTAL=$(bcftools view -H "$OUTVCF" | wc -l)
echo ""
echo "=== gnomAD streaming complete: $(date) ==="
echo "  Total common SNVs at ENCORI sites: $TOTAL"
echo "  Output: $OUTVCF"
echo ""
echo "Next step:"
echo "  python3 score.py \\"
echo "    --vcf    $OUTVCF \\"
echo "    --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \\"
echo "    --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \\"
echo "    --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \\"
echo "    --model  models/seedbreaker_lora_v2/best \\"
echo "    --out    results/validation/gnomad_background_scored.tsv"
