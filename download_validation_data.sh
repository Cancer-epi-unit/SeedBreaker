#!/usr/bin/env bash
# =============================================================================
# download_validation_data.sh — Phase 7 validation data download
# =============================================================================
# Downloads and preps:
#   1. ClinVar GRCh38 VCF            (~100 MB)
#   2. GTEx v10 significant eQTL pairs (~2 GB)
#   3. gnomAD v4 common 3'UTR SNVs   (streamed per-chromosome, filtered)
#
# Run from the SeedBreaker project root:
#   conda activate seedbreaker
#   cd /Users/josh/Desktop/Projects/SeedBreaker
#   bash download_validation_data.sh
#
# Requires: wget, bcftools, bedtools, bgzip, tabix (all in seedbreaker env)
# Expected disk: ~5 GB total for all three datasets
# =============================================================================

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
DATA="$PROJ/Pre_data"
ENCORI="$DATA/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
LOG="$PROJ/Pre_data/download_log.txt"

mkdir -p "$DATA/ClinVar" "$DATA/GTEx" "$DATA/gnomAD"
exec > >(tee -a "$LOG") 2>&1
echo "=== Download started: $(date) ==="


# ─── 1. ClinVar ───────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Downloading ClinVar GRCh38 VCF..."
CDIR="$DATA/ClinVar"

wget -c -q --show-progress \
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz" \
    -O "$CDIR/clinvar.vcf.gz"

wget -c -q \
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi" \
    -O "$CDIR/clinvar.vcf.gz.tbi"

echo "  ClinVar downloaded. Filtering to pathogenic 3'UTR variants..."

# Filter: pathogenic/likely_pathogenic + 3'UTR molecular consequence
bcftools view "$CDIR/clinvar.vcf.gz" \
    --include 'INFO/CLNSIG~"Pathogenic" && INFO/MC~"3_prime_UTR"' \
    -O z -o "$CDIR/clinvar_pathogenic_3utr.vcf.gz"
bcftools index -t "$CDIR/clinvar_pathogenic_3utr.vcf.gz"

N=$(bcftools view -H "$CDIR/clinvar_pathogenic_3utr.vcf.gz" | wc -l)
echo "  → $N pathogenic 3'UTR variants in ClinVar"

# ClinVar uses '1' not 'chr1' — rename to match ENCORI
echo "  Renaming chromosomes (1→chr1) to match ENCORI..."
for i in {1..22} X Y MT; do echo "$i chr$i"; done > /tmp/clinvar_chr_rename.txt
bcftools annotate \
    --rename-chrs /tmp/clinvar_chr_rename.txt \
    "$CDIR/clinvar_pathogenic_3utr.vcf.gz" \
    -O z -o "$CDIR/clinvar_pathogenic_3utr_chr.vcf.gz"
bcftools index -t "$CDIR/clinvar_pathogenic_3utr_chr.vcf.gz"

# Intersect with ENCORI — keep VCF header with -header flag so output is indexable
echo "  Intersecting with ENCORI atlas..."
bedtools intersect \
    -a "$CDIR/clinvar_pathogenic_3utr_chr.vcf.gz" \
    -b "$ENCORI" -wa -u -header \
    | bgzip > "$CDIR/clinvar_pathogenic_3utr_encori.vcf.gz"
bcftools index -t "$CDIR/clinvar_pathogenic_3utr_encori.vcf.gz"

N2=$(bcftools view -H "$CDIR/clinvar_pathogenic_3utr_encori.vcf.gz" | wc -l)
echo "  → $N2 overlap ENCORI atlas (ready for score.py)"


# ─── 2. GTEx v10 ──────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Downloading GTEx v10 significant eQTL pairs..."
GDIR="$DATA/GTEx"

# Significant variant-gene pairs for all tissues (~2 GB tar)
GTEX_URL="https://storage.googleapis.com/adult-gtex/bulk-qtl/v10/single-tissue-cis-qtl/GTEx_Analysis_v10_eQTL.tar.gz"

if [ ! -f "$GDIR/GTEx_Analysis_v10_eQTL.tar.gz" ]; then
    wget -c -q --show-progress "$GTEX_URL" -O "$GDIR/GTEx_Analysis_v10_eQTL.tar.gz"
else
    echo "  GTEx tar already exists, skipping download"
fi

echo "  Extracting significant variant-gene pair files..."
# Extract only the *.signif_variant_gene_pairs.txt.gz files (small per-tissue)
tar -xzf "$GDIR/GTEx_Analysis_v10_eQTL.tar.gz" \
    -C "$GDIR" \
    --wildcards "*.signif_variant_gene_pairs.txt.gz" \
    2>/dev/null || true

echo "  Merging all tissues into single eQTL variant list..."
MERGED="$GDIR/gtex_v10_all_signif_variants.txt.gz"

# GTEx variant format: chr_pos_ref_alt_b38
# Extract unique variant IDs across all tissues
find "$GDIR" -name "*.signif_variant_gene_pairs.txt.gz" | sort | \
    xargs -I{} sh -c 'zcat "{}" | tail -n +2 | cut -f1' | \
    sort -u | gzip > "$MERGED"

N=$(zcat "$MERGED" | wc -l)
echo "  → $N unique significant eQTL variants across all GTEx tissues"

# Convert to BED for intersection (chr_pos_ref_alt_b38 → chrom:pos)
echo "  Converting to BED format..."
zcat "$MERGED" | awk -F'_' '{
    chr=$1; pos=$2; ref=$3; alt=$4
    # GTEx uses chr1 format already
    print chr"\t"(pos-1)"\t"pos"\t"$0"\t"ref"\t"alt
}' | bgzip > "$GDIR/gtex_v10_signif_variants.bed.gz"
tabix -p bed "$GDIR/gtex_v10_signif_variants.bed.gz"

# Intersect with ENCORI to find eQTLs in miRNA binding sites
echo "  Finding eQTLs overlapping ENCORI atlas..."
bedtools intersect \
    -a "$GDIR/gtex_v10_signif_variants.bed.gz" \
    -b "$ENCORI" -wa -u | \
    bgzip > "$GDIR/gtex_v10_signif_encori_overlap.bed.gz"
tabix -p bed "$GDIR/gtex_v10_signif_encori_overlap.bed.gz"

N2=$(zcat "$GDIR/gtex_v10_signif_encori_overlap.bed.gz" | wc -l)
echo "  → $N2 significant eQTL variants overlap ENCORI miRNA sites"

# Build a scoreable VCF from GTEx eQTL × ENCORI overlaps
echo "  Building scoreable VCF from GTEx eQTL × ENCORI overlaps..."
(
echo '##fileformat=VCFv4.2'
echo '##INFO=<ID=AF,Number=A,Type=Float,Description="Not available for GTEx">'
echo '#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO'
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
echo "  → $N3 SNVs ready for score.py (VCF at Pre_data/GTEx/gtex_v10_encori_snvs.vcf.gz)"


# ─── 3. gnomAD v4 — 3'UTR common variants (streamed) ─────────────────────────
echo ""
echo "[3/3] Streaming gnomAD v4 common 3'UTR variants..."
NDIR="$DATA/gnomAD"

# Build 3'UTR BED from ENCORI (already our target regions)
UTRBED="$NDIR/encori_3utr_regions.bed"
zcat "$ENCORI" | awk 'BEGIN{OFS="\t"}{print $1,$2,$3}' | \
    sort -k1,1 -k2,2n | bedtools merge > "$UTRBED"
N=$(wc -l < "$UTRBED")
echo "  Using $N merged ENCORI regions as 3'UTR target"

# Stream each chromosome — filter to SNVs with AF>0.01 in ENCORI regions
GNOMAD_BASE="https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/genomes"
OUTVCF="$NDIR/gnomad_v4_common_3utr_snvs.vcf.gz"

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22"

echo "  Streaming ${CHROMS} from gnomAD (this will take ~30-60 min)..."

# Temp dir for per-chrom files
TMPDIR_GNOMAD="$NDIR/tmp_chroms"
mkdir -p "$TMPDIR_GNOMAD"

for CHR in $CHROMS; do
    OUTF="$TMPDIR_GNOMAD/${CHR}_common_3utr.vcf.gz"
    if [ -f "$OUTF" ]; then
        echo "    $CHR — already done, skipping"
        continue
    fi
    echo "    Streaming $CHR..."
    URL="${GNOMAD_BASE}/gnomad.genomes.v4.1.sites.${CHR}.vcf.bgz"
    bcftools view "$URL" \
        --regions-file "$UTRBED" \
        --include 'AF[0]>0.01 && TYPE="snp"' \
        --min-ac 1 \
        -O z -o "$OUTF" 2>/dev/null || {
            echo "    WARNING: $CHR failed (network or region issue) — skipping"
            continue
        }
    bcftools index -t "$OUTF"
    N=$(bcftools view -H "$OUTF" | wc -l)
    echo "    $CHR → $N variants"
done

echo "  Merging chromosomes..."
FILES=$(find "$TMPDIR_GNOMAD" -name "*.vcf.gz" | sort)
if [ -n "$FILES" ]; then
    bcftools concat --allow-overlaps $FILES -O z -o "$OUTVCF"
    bcftools index -t "$OUTVCF"
    N=$(bcftools view -H "$OUTVCF" | wc -l)
    echo "  → $N common 3'UTR SNVs in gnomAD v4 (ready for score.py)"
    # Clean up tmp
    rm -rf "$TMPDIR_GNOMAD"
else
    echo "  WARNING: No gnomAD files downloaded. Check network access and bcftools htslib version."
fi


# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 7 data download complete: $(date)"
echo ""
echo "Files ready for score.py:"
echo "  ClinVar:  Pre_data/ClinVar/clinvar_pathogenic_3utr_encori.vcf.gz"
echo "  GTEx:     Pre_data/GTEx/gtex_v10_encori_snvs.vcf.gz"
echo "  gnomAD:   Pre_data/gnomAD/gnomad_v4_common_3utr_snvs.vcf.gz"
echo ""
echo "Next step: run score_validation.sh to score all three"
echo "============================================================"
