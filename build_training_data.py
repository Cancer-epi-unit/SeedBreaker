#!/usr/bin/env python3
"""
build_training_data.py
----------------------
SeedBreaker Phase 4 — training data construction via in-silico saturation mutagenesis.

For each miRNA binding site in the ENCORI atlas, systematically generates all possible
single-nucleotide variants at seed positions 2-8, labels them as disruptive or
preserving, extracts ±window nt sequence context, and writes a balanced parquet file
ready for LoRA fine-tuning.

Label logic:
  DISRUPTIVE (1) — variant breaks Watson-Crick pairing at a seed position
                   AND does not create a compensatory G:U wobble
  PRESERVING (0) — variant maintains WC pairing, or improves a wobble to WC
  EXCLUDED       — variant creates a G:U wobble (ambiguous, excluded by default)
                   OR the reference base is already a mismatch (non-canonical site)

Class balancing:
  Disruptive examples are downsampled to --balance-ratio x preserving count.
  Default ratio is 3:1 (disruptive:preserving).

Chromosome splits (no transcript leakage):
  Test  — chr8, chr18
  Val   — chr4
  Train — everything else

Usage:
  python3 build_training_data.py \
    --atlas   Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \
    --seeds   Pre_data/miRBase/mirna_seeds_hsa.json \
    --fasta   Pre_data/Genome/GRCh38.primary_assembly.genome.fa \
    --out     training_data/training_data.parquet

Requirements:
  pip install pyfaidx pandas pyarrow tqdm
"""

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import pyfaidx
except ImportError:
    sys.exit("Missing dependency — run: pip install pyfaidx")

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("Missing dependency — run: pip install pyarrow")


# ── Constants ──────────────────────────────────────────────────────────────────

DNA_BASES = ["A", "C", "G", "T"]

# Watson-Crick complement: for a given miRNA base (DNA representation),
# what is the expected target (3'UTR) base?
WC = {"A": "T", "T": "A", "G": "C", "C": "G"}

# G:U wobble pairs (miRNA base -> target base)
WOBBLE = {"G": "T", "T": "G"}

# Evidence tier weights for label_confidence
SITE_TYPE_WEIGHT = {
    "8mer":         1.00,
    "7mer-m8":      0.85,
    "7mer-A1":      0.75,
    "unclassified": 0.60,
    "unknown":      0.50,
}

# Chromosomes held out from training
TEST_CHROMS = {"chr8", "chr18"}
VAL_CHROMS  = {"chr4"}


# ── Base pairing ───────────────────────────────────────────────────────────────

def pairing(mirna_base: str, target_base: str) -> str:
    """Return 'WC', 'wobble', or 'mismatch' for a miRNA:target base pair."""
    m = mirna_base.upper().replace("U", "T")
    t = target_base.upper().replace("U", "T")
    if WC.get(m) == t:
        return "WC"
    if WOBBLE.get(m) == t:
        return "wobble"
    return "mismatch"


def assign_label(mirna_base: str, ref_base: str, alt_base: str,
                 include_wobble: bool) -> tuple:
    """
    Assign a disruption label to a REF->ALT change at one seed position.

    Returns (label, reason) where:
      label = 1     -> disruptive
      label = 0     -> preserving
      label = None  -> exclude this example
    """
    ref_pair = pairing(mirna_base, ref_base)
    alt_pair = pairing(mirna_base, alt_base)

    # Reference is already a mismatch — non-canonical site, skip
    if ref_pair == "mismatch":
        return None, "ref_mismatch"

    if ref_pair == "WC":
        if alt_pair == "WC":
            return 0, "WC_to_WC"
        if alt_pair == "mismatch":
            return 1, "WC_to_mismatch"
        if alt_pair == "wobble":
            if include_wobble:
                return 0, "WC_to_wobble"
            return None, "wobble_excluded"

    if ref_pair == "wobble":
        if alt_pair == "mismatch":
            return 1, "wobble_to_mismatch"
        if alt_pair == "WC":
            return 0, "wobble_to_WC"
        if alt_pair == "wobble":
            return 0, "wobble_to_wobble"

    return None, "unhandled"


# ── Sequence window extraction ─────────────────────────────────────────────────

def extract_windows(fasta, chrom: str, pos: int, ref: str,
                    alt: str, window: int):
    """
    Extract REF and ALT sequence windows of +/-window nt around the variant.
    pos is 0-based.
    Returns (ref_window, alt_window) or None on failure.
    """
    try:
        chrom_len = len(fasta[chrom])
        start     = max(0, pos - window)
        end       = min(chrom_len, pos + len(ref) + window)
        region    = fasta[chrom][start:end].seq.upper()
        offset    = pos - start

        ref_window = region
        alt_window = region[:offset] + alt + region[offset + len(ref):]

        # Trim both windows to consistent length
        ref_window = ref_window[: window * 2 + 1]
        alt_window = alt_window[: window * 2 + 1]

        return ref_window, alt_window

    except Exception:
        return None


# ── Atlas parser ───────────────────────────────────────────────────────────────

def iter_atlas(atlas_path: str, min_clip: int):
    """
    Yield site dicts from the ENCORI annotated BED atlas.
    Skips sites with fewer than min_clip supporting CLIP experiments.
    """
    opener = gzip.open if atlas_path.endswith(".gz") else open
    with opener(atlas_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 15:
                continue
            try:
                clip_num = int(float(fields[11]))
            except (ValueError, IndexError):
                clip_num = 0
            if clip_num < min_clip:
                continue
            yield {
                "chrom":     fields[0],
                "start":     int(fields[1]),
                "end":       int(fields[2]),
                "strand":    fields[5],
                "mirna":     fields[6],
                "mimat":     fields[7],
                "gene":      fields[8],
                "gene_id":   fields[9],
                "clip_num":  clip_num,
                "site_type": fields[15] if len(fields) > 15 else "unclassified",
                "phylop":    fields[14] if len(fields) > 14 else "NA",
            }


# ── Main ───────────────────────────────────────────────────────────────────────

def build(args):

    # ── Load seeds ────────────────────────────────────────────────────────────
    print("Loading miRBase seeds...", flush=True)
    with open(args.seeds) as f:
        seeds = json.load(f)
    print(f"  {len(seeds):,} human miRNA sequences")

    # ── Open genome ───────────────────────────────────────────────────────────
    print("Opening reference genome...", flush=True)
    fasta = pyfaidx.Fasta(args.fasta, sequence_always_upper=True)
    print(f"  {len(fasta.keys())} chromosomes indexed")

    # ── Count sites for progress bar ──────────────────────────────────────────
    print("Counting atlas sites...", flush=True)
    n_total = sum(1 for _ in iter_atlas(args.atlas, args.min_clip))
    n_sites = min(n_total, args.max_sites) if args.max_sites else n_total
    print(f"  {n_sites:,} sites to process")

    # ── Saturation mutagenesis ────────────────────────────────────────────────
    print(f"\nRunning saturation mutagenesis...")
    print(f"  window        = +/-{args.window} nt")
    print(f"  min_clip      = {args.min_clip}")
    print(f"  wobble        = {'included as preserving' if args.include_wobble else 'excluded'}")
    print(f"  balance ratio = {args.balance_ratio}:1 (disruptive:preserving)\n")

    rows        = []
    stats       = defaultdict(int)
    chunk_size  = 100_000
    chunk_index = 0
    chunk_files = []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def flush_chunk(rows):
        nonlocal chunk_index
        if not rows:
            return
        path = args.out.replace(".parquet", f"_tmp{chunk_index:04d}.parquet")
        pd.DataFrame(rows).to_parquet(path, index=False)
        chunk_files.append(path)
        chunk_index += 1

    with tqdm(total=n_sites, unit="site", ncols=80) as bar:
        for i, site in enumerate(iter_atlas(args.atlas, args.min_clip)):
            if args.max_sites and i >= args.max_sites:
                break

            mirna    = site["mirna"]
            chrom    = site["chrom"]
            strand   = site["strand"]
            s        = site["start"]
            e        = site["end"]
            site_len = e - s

            if mirna not in seeds:
                stats["mirna_missing"] += 1
                bar.update(1)
                continue

            mirna_seq = seeds[mirna]["sequence"]   # 22nt DNA string

            # Label confidence weight for this site
            type_w = SITE_TYPE_WEIGHT.get(site["site_type"], 0.5)
            clip_w = min(1.0, site["clip_num"] / 5.0)
            conf   = round((type_w + clip_w) / 2, 3)

            # Seed positions 2-8 (0-indexed: 1-7)
            for seed_idx in range(1, 8):
                if seed_idx >= len(mirna_seq):
                    continue

                mirna_base = mirna_seq[seed_idx]

                # Map miRNA seed position to genomic coordinate.
                # miRNA runs 5'->3'; target strand is antiparallel.
                # Seed position 2 (idx 1) pairs with the 3'-most target base.
                if strand == "+":
                    genomic_pos = s + (site_len - 1 - seed_idx)
                else:
                    genomic_pos = s + seed_idx

                if genomic_pos < 0 or genomic_pos >= e + 5:
                    stats["pos_out_of_range"] += 1
                    continue

                # Get reference base from genome
                try:
                    ref_base = fasta[chrom][genomic_pos].seq.upper()
                except Exception:
                    stats["fasta_error"] += 1
                    continue

                # Generate all 3 possible SNVs at this position
                for alt_base in DNA_BASES:
                    if alt_base == ref_base:
                        continue

                    label, reason = assign_label(
                        mirna_base, ref_base, alt_base, args.include_wobble
                    )

                    if label is None:
                        stats[f"excluded_{reason}"] += 1
                        continue

                    windows = extract_windows(
                        fasta, chrom, genomic_pos, ref_base, alt_base, args.window
                    )
                    if windows is None:
                        stats["window_failed"] += 1
                        continue

                    ref_window, alt_window = windows

                    rows.append({
                        "site_id":          f"{chrom}:{s}-{e}:{mirna}",
                        "mirna":            mirna,
                        "mimat":            site["mimat"],
                        "gene":             site["gene"],
                        "gene_id":          site["gene_id"],
                        "chrom":            chrom,
                        "pos":              genomic_pos,
                        "ref":              ref_base,
                        "alt":              alt_base,
                        "ref_window":       ref_window,
                        "alt_window":       alt_window,
                        "seed_pos":         seed_idx + 1,
                        "mirna_base":       mirna_base,
                        "pairing_ref":      pairing(mirna_base, ref_base),
                        "pairing_alt":      pairing(mirna_base, alt_base),
                        "label":            label,
                        "label_reason":     reason,
                        "label_confidence": conf,
                        "site_type":        site["site_type"],
                        "clip_num":         site["clip_num"],
                        "strand":           strand,
                        "phylop":           site["phylop"],
                    })

                    stats["total"] += 1
                    stats[f"label_{label}"] += 1

            if len(rows) >= chunk_size:
                flush_chunk(rows)
                rows = []

            stats["sites_processed"] += 1
            bar.update(1)

    flush_chunk(rows)

    # ── Merge chunks ──────────────────────────────────────────────────────────
    print(f"\nMerging {len(chunk_files)} chunks...", flush=True)
    if not chunk_files:
        sys.exit("No data generated — check atlas, seeds, and fasta paths.")

    df = pd.concat(
        [pd.read_parquet(p) for p in chunk_files], ignore_index=True
    )
    for p in chunk_files:
        os.remove(p)

    # ── Balance classes ───────────────────────────────────────────────────────
    n_preserving = (df["label"] == 0).sum()
    n_disruptive = (df["label"] == 1).sum()
    target_pos   = min(n_disruptive, n_preserving * args.balance_ratio)

    df_neg = df[df["label"] == 0]
    df_pos = df[df["label"] == 1].sample(target_pos, random_state=42)
    df = (
        pd.concat([df_neg, df_pos], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # ── Chromosome-level splits ───────────────────────────────────────────────
    df["split"] = "train"
    df.loc[df["chrom"].isin(VAL_CHROMS),  "split"] = "val"
    df.loc[df["chrom"].isin(TEST_CHROMS), "split"] = "test"

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(args.out, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_dis = (df["label"] == 1).sum()
    n_pre = (df["label"] == 0).sum()

    print(f"""
{'='*55}
TRAINING DATA SUMMARY
{'='*55}
Sites processed:        {stats['sites_processed']:>10,}
miRNA not in seeds:     {stats['mirna_missing']:>10,}

Before balancing:
  Disruptive (1):       {n_disruptive:>10,}
  Preserving (0):       {n_preserving:>10,}
  Raw class balance:    {n_disruptive/(n_disruptive+n_preserving)*100:>9.1f}% disruptive

After balancing ({args.balance_ratio}:1):
  Disruptive (1):       {n_dis:>10,}
  Preserving (0):       {n_pre:>10,}
  Final ratio:          {n_dis/max(n_pre,1):>9.1f}:1

Splits:
  Train:                {(df['split']=='train').sum():>10,}
  Val   (chr4):         {(df['split']=='val').sum():>10,}
  Test  (chr8+18):      {(df['split']=='test').sum():>10,}

Label confidence:
  Mean:                 {df['label_confidence'].mean():>10.3f}
  Std:                  {df['label_confidence'].std():>10.3f}

Site type breakdown:
{df['site_type'].value_counts().to_string()}

Output: {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)
{'='*55}
""")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--atlas",          required=True,
                    help="ENCORI annotated BED.gz")
    ap.add_argument("--seeds",          required=True,
                    help="miRBase seed JSON (mirna_seeds_hsa.json)")
    ap.add_argument("--fasta",          required=True,
                    help="hg38 FASTA (must be indexed with samtools faidx)")
    ap.add_argument("--out",            default="training_data/training_data.parquet",
                    help="Output parquet path")
    ap.add_argument("--window",         type=int, default=30,
                    help="Nucleotides each side of variant (default: 30)")
    ap.add_argument("--min-clip",       type=int, default=1,
                    help="Min CLIP experiments per site (default: 1)")
    ap.add_argument("--balance-ratio",  type=int, default=3,
                    help="Disruptive:preserving ratio after balancing (default: 3)")
    ap.add_argument("--include-wobble", action="store_true",
                    help="Include G:U wobble changes as label=0 (default: exclude)")
    ap.add_argument("--max-sites",      type=int, default=None,
                    help="Process only N sites for testing (default: all)")
    args = ap.parse_args()

    for path, label in [(args.atlas, "atlas"), (args.seeds, "seeds"),
                         (args.fasta, "fasta")]:
        if not os.path.exists(path):
            sys.exit(f"File not found ({label}): {path}")

    fai = args.fasta + ".fai"
    if not os.path.exists(fai):
        sys.exit(
            f"FASTA index missing: {fai}\n"
            f"Run: samtools faidx {args.fasta}"
        )

    build(args)


if __name__ == "__main__":
    main()
