#!/usr/bin/env python3
"""
process_gtex.py — Extract GTEx v11 eQTL variants and build scoreable VCF
=========================================================================
Reads all *.signif_pairs.parquet files from GTEx_Analysis_v11_eQTL.tar,
merges unique variants, intersects with ENCORI atlas, and writes a VCF
ready for score.py.

Usage:
    conda activate seedbreaker
    cd /Users/josh/Desktop/Projects/SeedBreaker
    python3 process_gtex.py
"""

import os
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

TAR     = "Pre_data/GTEx/GTEx_Analysis_v11_eQTL.tar"
ENCORI  = "Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
OUTDIR  = "Pre_data/GTEx"
OUTVCF  = f"{OUTDIR}/gtex_v11_encori_snvs.vcf.gz"


def main():
    if not Path(TAR).exists():
        sys.exit(f"ERROR: {TAR} not found.")

    print(f"[1/4] Reading GTEx v11 significant pairs from tar...")
    all_variants = {}   # variant_id → af

    with tarfile.open(TAR, "r") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith(".signif_pairs.parquet")]
        print(f"  {len(members)} tissues found")

        for i, m in enumerate(members):
            tissue = Path(m.name).stem.split(".")[0]
            f = tf.extractfile(m)
            df = pd.read_parquet(io.BytesIO(f.read()),
                                 columns=["variant_id", "af"])
            for _, row in df.iterrows():
                vid = row["variant_id"]
                if vid not in all_variants:
                    all_variants[vid] = row["af"]

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(members)} tissues processed, "
                      f"{len(all_variants):,} unique variants so far", flush=True)

    print(f"  Total unique significant eQTL variants: {len(all_variants):,}")

    print(f"\n[2/4] Parsing variant IDs (chr_pos_ref_alt_b38 format)...")
    rows = []
    skipped = 0
    for vid, af in all_variants.items():
        parts = vid.split("_")
        if len(parts) < 4:
            skipped += 1
            continue
        chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
        if len(ref) != 1 or len(alt) != 1:
            skipped += 1
            continue   # SNVs only
        rows.append((chrom, int(pos), ref, alt, af, vid))

    print(f"  {len(rows):,} SNVs parsed  ({skipped:,} indels/multi-allelic skipped)")

    print(f"\n[3/4] Intersecting with ENCORI atlas (bedtools)...")
    # Write BED
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as f:
        tmp_bed = f.name
        for chrom, pos, ref, alt, af, vid in rows:
            # Escape | in vid (used as delimiter in score.py tag)
            f.write(f"{chrom}\t{pos-1}\t{pos}\t{vid}|{ref}|{alt}|{af:.6f}\n")

    try:
        r = subprocess.run(
            ["bedtools", "intersect", "-a", tmp_bed, "-b", ENCORI, "-wa", "-u"],
            capture_output=True, text=True, check=True
        )
    finally:
        os.unlink(tmp_bed)

    hit_ids = set()
    for line in r.stdout.splitlines():
        c = line.split("\t")
        if len(c) >= 4:
            vid = c[3].split("|")[0]
            hit_ids.add(vid)

    print(f"  {len(hit_ids):,} eQTL variants overlap ENCORI miRNA sites")

    print(f"\n[4/4] Writing VCF...")
    variant_lookup = {vid: (chrom, pos, ref, alt, af)
                      for chrom, pos, ref, alt, af, vid in rows}

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    # Sort by chromosome then position
    CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
    CHROM_ORDER.update({"chrX": 23, "chrY": 24, "chrM": 25, "chrMT": 25})

    def sort_key(vid):
        chrom, pos, ref, alt, af = variant_lookup[vid]
        return (CHROM_ORDER.get(chrom, 99), pos)

    sorted_ids = sorted(hit_ids, key=sort_key)

    with tempfile.NamedTemporaryFile("w", suffix=".vcf", delete=False) as f:
        tmp_vcf = f.name
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="GTEx allele frequency">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for vid in sorted_ids:
            chrom, pos, ref, alt, af = variant_lookup[vid]
            f.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\tAF={af:.6f}\n")

    # bgzip + index
    subprocess.run(f"bgzip -c {tmp_vcf} > {OUTVCF}", shell=True, check=True)
    os.unlink(tmp_vcf)
    subprocess.run(["bcftools", "index", "-t", OUTVCF], check=True)

    n = int(subprocess.check_output(
        f"bcftools view -H {OUTVCF} | wc -l", shell=True
    ).strip())

    print(f"\n{'='*50}")
    print(f"GTEx v11 processing complete")
    print(f"  {n:,} SNVs ready for score.py")
    print(f"  Output: {OUTVCF}")
    print(f"\nNext step:")
    print(f"  python3 score.py \\")
    print(f"    --vcf    {OUTVCF} \\")
    print(f"    --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \\")
    print(f"    --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \\")
    print(f"    --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \\")
    print(f"    --model  models/seedbreaker_lora_v2/best \\")
    print(f"    --out    results/validation/gtex_scored.tsv")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
