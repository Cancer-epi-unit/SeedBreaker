#!/usr/bin/env bash
# =============================================================================
# process_gtex.sh — Process GTEx v10 eQTL tar after manual download
# =============================================================================
# Run AFTER manually downloading GTEx_Analysis_v10_eQTL.tar.gz from
# gtexportal.org → Downloads → GTEx Analysis V10 → Single-Tissue cis-QTL Data
#
# Usage:
#   conda activate seedbreaker
#   cd /Users/josh/Desktop/Projects/SeedBreaker
#   bash process_gtex.sh
# =============================================================================

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
GDIR="$PROJ/Pre_data/GTEx"
ENCORI="$PROJ/Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
TAR="$GDIR/GTEx_Analysis_v10_eQTL.tar.gz"

if [ ! -f "$TAR" ]; then
    echo "ERROR: $TAR not found."
    echo "Download from: gtexportal.org → Downloads → GTEx Analysis V10"
    echo "→ Single-Tissue cis-QTL Data → GTEx_Analysis_v10_eQTL.tar.gz"
    exit 1
fi

echo "=== GTEx v10 processing started: $(date) ==="

echo "[1/4] Extracting significant variant-gene pair files..."
tar -xzf "$TAR" -C "$GDIR" \
    --wildcards "*.signif_variant_gene_pairs.txt.gz" 2>/dev/null || true

N_TISSUES=$(find "$GDIR" -name "*.signif_variant_gene_pairs.txt.gz" | wc -l)
echo "  $N_TISSUES tissues found"

echo "[2/4] Merging all tissues → unique variant list..."
MERGED="$GDIR/gtex_v10_all_signif_variants.txt.gz"
find "$GDIR" -name "*.signif_variant_gene_pairs.txt.gz" | sort | \
    xargs -I{} sh -c 'zcat "{}" | tail -n +2 | cut -f1' | \
    sort -u | gzip > "$MERGED"
N=$(zcat "$MERGED" | wc -l)
echo "  $N unique significant eQTL variants across all tissues"

echo "[3/4] Converting to BED + intersecting with ENCORI..."
zcat "$MERGED" | awk -F'_' '{
    chr=$1; pos=$2; ref=$3; alt=$4
    print chr"\t"(pos-1)"\t"pos"\t"$0"\t"ref"\t"alt
}' | bgzip > "$GDIR/gtex_v10_signif_variants.bed.gz"
tabix -p bed "$GDIR/gtex_v10_signif_variants.bed.gz"

bedtools intersect \
    -a "$GDIR/gtex_v10_signif_variants.bed.gz" \
    -b "$ENCORI" -wa -u | \
    bgzip > "$GDIR/gtex_v10_signif_encori_overlap.bed.gz"
tabix -p bed "$GDIR/gtex_v10_signif_encori_overlap.bed.gz"

N2=$(zcat "$GDIR/gtex_v10_signif_encori_overlap.bed.gz" | wc -l)
echo "  $N2 significant eQTL variants overlap ENCORI miRNA sites"

echo "[4/4] Building scoreable VCF..."
(
echo '##fileformat=VCFv4.2'
echo '##INFO=<ID=AF,Number=A,Type=Float,Description="Not available for GTEx">'
printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
zcat "$GDIR/gtex_v10_signif_encori_overlap.bed.gz" | \
    awk -F'\t' '{
        split($4, a, "_")
        chr=a[1]; pos=a[2]; ref=a[3]; alt=a[4]
        if (length(ref)==1 && length(alt)==1)
            print chr"\t"pos"\t"$4"\t"ref"\t"alt"\t.\tPASS\tAF=."
    }'
) | bgzip > "$GDIR/gtex_v10_encori_snvs.vcf.gz"
bcftools index -t "$GDIR/gtex_v10_encori_snvs.vcf.gz"

N3=$(bcftools view -H "$GDIR/gtex_v10_encori_snvs.vcf.gz" | wc -l)
echo ""
echo "=== GTEx processing complete ==="
echo "  $N3 SNVs ready for score.py"
echo "  File: Pre_data/GTEx/gtex_v10_encori_snvs.vcf.gz"
echo ""
echo "Next: python3 score.py --vcf Pre_data/GTEx/gtex_v10_encori_snvs.vcf.gz ..."
