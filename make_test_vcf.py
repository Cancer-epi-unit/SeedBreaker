#!/usr/bin/env python3
"""
make_test_vcf.py — Generate a synthetic test VCF for score.py

Picks ~30 ENCORI binding sites, places a synthetic SNV at seed position 4
(middle of the seed), looks up the real reference base via samtools faidx,
then chooses an alt allele that breaks Watson-Crick pairing.

Run:
    conda activate seedbreaker
    cd /Users/josh/Desktop/Projects/SeedBreaker
    python3 make_test_vcf.py
    # → test_data/test_variants.vcf
"""

import json
import os
import subprocess
import sys
import gzip

ENCORI  = "Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz"
GENOME  = "Pre_data/Genome/GRCh38.primary_assembly.genome.fa"
SEEDS   = "Pre_data/miRBase/mirna_seeds_hsa.json"
OUTDIR  = "test_data"
OUTVCF  = f"{OUTDIR}/test_variants.vcf"
N_SITES = 50     # try this many sites (some will fail ref extraction)
TARGET  = 30     # stop once we have this many valid variants


def faidx(chrom, pos0):
    """Get reference base at 0-based position."""
    region = f"{chrom}:{pos0+1}-{pos0+1}"
    r = subprocess.run(
        ["samtools", "faidx", GENOME, region],
        capture_output=True, text=True
    )
    if r.returncode != 0 or not r.stdout:
        return None
    lines = r.stdout.strip().split("\n")
    seq = "".join(lines[1:]).upper()
    return seq[0] if seq else None


def wc_complement(base):
    return {"A":"T","T":"A","C":"G","G":"C","U":"A"}.get(base.upper())


def disruptive_alt(ref, mirna_nt):
    """Return an alt base that is NOT WC with mirna_nt (i.e., breaks the pair)."""
    wc = wc_complement(mirna_nt)
    # pick a base ≠ ref and ≠ WC complement of mirna_nt
    for b in ["A","C","G","T"]:
        if b != ref and b != wc:
            return b
    return None


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load seeds
    print("Loading miRNA seeds...")
    with open(SEEDS) as f:
        seeds_raw = json.load(f)
    seed_by_lower = {k.lower(): v["seed"] for k, v in seeds_raw.items()}

    def get_seed(name):
        lo = name.lower()
        if lo in seed_by_lower:
            return seed_by_lower[lo]
        for k, v in seed_by_lower.items():
            if lo in k or k in lo:
                return v
        return None

    # Read ENCORI sites
    print("Scanning ENCORI atlas...")
    sites = []
    with gzip.open(ENCORI, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 7:
                continue
            sites.append({
                "chrom":  c[0],
                "start":  int(c[1]),
                "end":    int(c[2]),
                "strand": c[5],
                "mirna":  c[6],
            })
            if len(sites) >= N_SITES * 3:
                break

    # Build variants
    print(f"Extracting reference bases and building SNVs...")
    records = []
    seed_pos_target = 4   # seed position 4 (middle of 2-8)

    for site in sites:
        if len(records) >= TARGET:
            break

        seed_seq = get_seed(site["mirna"])
        if not seed_seq or len(seed_seq) < 7:
            continue

        # Compute genomic position of seed_pos=4
        if site["strand"] == "+":
            # seed_pos = site_end - var_pos_0 → var_pos_0 = site_end - seed_pos
            var_0 = site["end"] - seed_pos_target
        else:
            # seed_pos = var_pos_0 - site_start + 1 → var_pos_0 = site_start + seed_pos - 1
            var_0 = site["start"] + seed_pos_target - 1

        ref = faidx(site["chrom"], var_0)
        if not ref or ref == "N":
            continue

        # miRNA nt at seed_pos=4 (0-indexed into seed_seq which is pos 2-8)
        mirna_nt = seed_seq[seed_pos_target - 2].upper()

        # For + strand: ref on target strand = ref base
        # For - strand: ref on target strand = complement of ref (but mirna pairs with complement of genomic base)
        if site["strand"] == "+":
            target_base = ref
        else:
            target_base = wc_complement(ref)

        # Only include if ref forms a WC pair (i.e., it's a clean site)
        wc = wc_complement(mirna_nt)
        if target_base != wc:
            continue   # ref already non-WC; skip

        alt = disruptive_alt(ref, mirna_nt)
        if alt is None:
            continue

        af = round(1e-4, 6)   # synthetic rare variant

        records.append({
            "chrom": site["chrom"],
            "pos":   var_0 + 1,   # 1-based VCF
            "ref":   ref,
            "alt":   alt,
            "af":    af,
            "mirna": site["mirna"],
            "gene":  site.get("gene",""),
        })

    print(f"  {len(records)} valid synthetic variants generated")

    if not records:
        print("ERROR: No valid variants generated. Check GENOME/ENCORI/SEEDS paths.")
        sys.exit(1)

    # Write VCF
    with open(OUTVCF, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write("##FILTER=<ID=PASS,Description=\"All filters passed\">\n")
        out.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">\n")
        out.write("##INFO=<ID=SOURCE,Number=1,Type=String,Description=\"Synthetic from ENCORI site\">\n")
        out.write(f"##reference={GENOME}\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in records:
            info = f"AF={r['af']};SOURCE=synthetic_{r['mirna']}"
            out.write(f"{r['chrom']}\t{r['pos']}\t.\t{r['ref']}\t{r['alt']}\t.\tPASS\t{info}\n")

    print(f"\nWrote {len(records)} variants to {OUTVCF}")
    print(f"\nTo score them:")
    print(f"  python3 score.py \\")
    print(f"    --vcf    {OUTVCF} \\")
    print(f"    --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \\")
    print(f"    --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \\")
    print(f"    --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \\")
    print(f"    --model  models/seedbreaker_lora_v2/best \\")
    print(f"    --out    test_data/test_scores.tsv")


if __name__ == "__main__":
    main()
