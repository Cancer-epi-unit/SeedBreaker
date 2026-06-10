#!/usr/bin/env python3
"""
score.py — SeedBreaker Phase 6: Variant Scoring Engine
=======================================================
Scores rare germline SNVs for miRNA binding site disruption.

For each variant that falls in a seed position (2-8) of an ENCORI-confirmed
miRNA binding site, computes:
  • physics_score — Watson-Crick disruption rule (0 or 1)
  • model_score   — ESM-2 + LoRA fine-tuned classifier probability
  • delta_mfe     — ViennaRNA RNAduplex Δfree energy (kcal/mol), optional
  • combined      — 0.4 × physics + 0.6 × model

Usage:
  python3 score.py \\
    --vcf    variants.vcf.gz \\
    --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \\
    --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \\
    --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \\
    --model  models/seedbreaker_lora_v2/best \\
    --out    results/scored_variants.tsv

Requires: bedtools, samtools in PATH; conda activate seedbreaker
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Watson-Crick helpers ───────────────────────────────────────────────────────

_COMP = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")

def rc(seq: str) -> str:
    return seq.translate(_COMP)[::-1]

def wc_complement(base: str) -> str:
    """Single-base WC complement (DNA, T≡U)."""
    return base.upper().replace("U","T").translate(str.maketrans("ACGT","TGCA"))

def is_wc(b1: str, b2: str) -> bool:
    """True if b1 and b2 form a Watson-Crick pair (DNA/RNA mixed ok)."""
    b1 = b1.upper().replace("U","T")
    b2 = b2.upper().replace("U","T")
    return (b1,b2) in {("A","T"),("T","A"),("C","G"),("G","C")}


# ── Seed position logic ────────────────────────────────────────────────────────

def get_seed_pos(var_pos_0: int, site_start: int, site_end: int,
                 strand: str) -> int | None:
    """
    Return miRNA seed position (2–8) for a variant, or None if outside seed.

    Convention (canonical TargetScan seed-match geometry):
      + strand: miRNA 5'→3' aligns antiparallel to target 3'→5'.
                seed pos k maps to genomic pos = site_end - k  (0-based)
      − strand: target 5'→3' runs right→left in genomic coords.
                seed pos k maps to genomic pos = site_start + k - 1  (0-based)
    """
    if strand == "+":
        k = site_end - var_pos_0
        if 2 <= k <= 8:
            return k
    else:
        k = var_pos_0 - site_start + 1
        if 2 <= k <= 8:
            return k
    return None


# ── WC disruption check ────────────────────────────────────────────────────────

def check_wc(ref: str, alt: str, mirna_seed: str,
             seed_pos: int, strand: str) -> bool | None:
    """
    True  = WC pair broken (DISRUPTIVE)
    False = WC pair maintained (PRESERVING)
    None  = ref was already non-WC (ambiguous, skip)

    mirna_seed: 7-char string, positions 2-8, 5'→3', DNA notation (T not U).
    """
    mi = mirna_seed[seed_pos - 2].upper()   # miRNA nt at this seed position

    # Get the TARGET strand nucleotide at this position
    if strand == "+":
        ref_t = ref.upper()
        alt_t = alt.upper()
    else:
        # Genomic base is on minus strand; target base = its complement
        ref_t = wc_complement(ref)
        alt_t = wc_complement(alt)

    ref_wc = is_wc(ref_t, mi)
    alt_wc = is_wc(alt_t, mi)

    if not ref_wc:
        return None      # non-canonical reference
    return not alt_wc    # True if alt breaks the pair


# ── Sequence extraction ────────────────────────────────────────────────────────

def faidx_seq(genome_fa: str, chrom: str, start0: int, end0: int) -> str:
    """Extract sequence [start0, end0) using samtools faidx (0-based coords)."""
    region = f"{chrom}:{start0+1}-{end0}"
    r = subprocess.run(
        ["samtools", "faidx", genome_fa, region],
        capture_output=True, text=True, check=True
    )
    return "".join(r.stdout.split("\n")[1:]).upper().replace(" ","")


def make_windows(genome_fa: str, chrom: str, pos1: int,
                 ref: str, alt: str, half: int = 25):
    """
    Build (ref_window, alt_window) of length 2*half+1 centered on the variant.
    pos1 is 1-based VCF POS.  Returns (None, None) on reference mismatch.
    """
    center = pos1 - 1                     # 0-based
    start  = max(0, center - half)
    end    = center + half + 1

    seq = faidx_seq(genome_fa, chrom, start, end)
    idx = center - start                  # variant offset in extracted sequence

    if not seq or idx >= len(seq) or seq[idx] != ref.upper():
        return None, None                 # ref mismatch

    ref_win = seq
    alt_win = seq[:idx] + alt.upper() + seq[idx+1:]
    return ref_win, alt_win


# ── VCF + bedtools intersection ────────────────────────────────────────────────

def vcf_to_bed(vcf_path: str) -> list[tuple]:
    """
    Parse VCF and emit SNV records as (chrom, pos0, pos1, tag) tuples.
    tag = 'ID|REF|ALT|AF'
    """
    opener = ["gzip", "-dc"] if vcf_path.endswith(".gz") else ["cat"]
    proc = subprocess.run(opener + [vcf_path],
                          capture_output=True, text=True, check=True)
    records = []
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        chrom, pos, vid, ref, alt_field = cols[0], int(cols[1]), cols[2], cols[3], cols[4]

        # SNVs only; skip multi-allelic for now
        alts = alt_field.split(",")
        for alt in alts:
            if len(ref) != 1 or len(alt) != 1 or alt == ".":
                continue
            af = None
            if len(cols) > 7:
                for f in cols[7].split(";"):
                    if f.startswith("AF="):
                        try:
                            af = float(f[3:].split(",")[0])
                        except Exception:
                            pass
            # Use chrom:pos:ref:alt as fallback if ID is missing/dot
            vid_out = vid if (vid and vid != ".") else f"{chrom}:{pos}:{ref}:{alt}"
            tag = f"{vid_out}|{ref}|{alt}|{af if af is not None else '.'}"
            records.append((chrom, pos-1, pos, tag))
    return records


def intersect(vcf_records: list, sites_bed: str) -> list[dict]:
    """
    bedtools intersect SNV positions against ENCORI sites.
    Returns list of dicts with variant + site info.

    ENCORI BED column order (0-based):
      0:chr 1:start 2:end 3:name 4:score 5:strand
      6:miRNA 7:MIMAT 8:gene_name 9:gene_id 10:RBP
      11:clip_num 12:targetscan 13:tdmd_score 14:phylop
      15:site_type 16:context_score
    After bedtools -wa -wb with our 4-col variant BED, site cols are offset by 4.
    """
    if not vcf_records:
        return []

    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as f:
        tmp = f.name
        for chrom, s, e, tag in vcf_records:
            f.write(f"{chrom}\t{s}\t{e}\t{tag}\n")

    try:
        r = subprocess.run(
            ["bedtools", "intersect", "-a", tmp, "-b", sites_bed, "-wa", "-wb"],
            capture_output=True, text=True, check=True
        )
    finally:
        os.unlink(tmp)

    overlaps = []
    for line in r.stdout.splitlines():
        c = line.split("\t")
        if len(c) < 21:      # 4 variant cols + 17 site cols
            continue
        vid, ref, alt, af_s = c[3].split("|", 3)
        overlaps.append({
            "chrom":       c[0],
            "pos":         int(c[1]) + 1,   # back to 1-based
            "variant_id":  vid,
            "ref":         ref,
            "alt":         alt,
            "af":          float(af_s) if af_s != "." else None,
            "site_start":  int(c[5]),
            "site_end":    int(c[6]),
            "site_name":   c[7],
            "strand":      c[9],
            "mirna":       c[10],
            "gene":        c[12],
            "site_type":   c[19] if len(c) > 19 else "unclassified",
        })
    return overlaps


# ── Model loading & inference ──────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    from peft import PeftModel

    print(f"  Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)

    print(f"  Loading ESM-2 t33 650M base model...", flush=True)
    cfg = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
    cfg.num_labels = 2
    base = AutoModelForSequenceClassification.from_pretrained(
        "facebook/esm2_t33_650M_UR50D", config=cfg, ignore_mismatched_sizes=True
    )
    print(f"  Loading LoRA adapter from {model_path}...", flush=True)
    model = PeftModel.from_pretrained(base, model_path).to(device)
    model.eval()
    return tok, model


def score_batch(tok, model, pairs: list[tuple[str,str]],
                device, batch_size: int = 32) -> list[float]:
    """Score (ref_window, alt_window) pairs; returns list of disruption probs."""
    seqs = [r + "NNN" + a for r, a in pairs]
    probs = []
    for i in range(0, len(seqs), batch_size):
        enc = tok(seqs[i:i+batch_size], max_length=256,
                  padding="max_length", truncation=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        probs.extend(torch.softmax(out.logits, dim=-1)[:,1].cpu().tolist())
    return probs


# ── ViennaRNA ─────────────────────────────────────────────────────────────────

def delta_mfe(mirna_seed: str, ref_win: str, alt_win: str) -> float | None:
    """
    Δmfe = mfe(alt duplex) − mfe(ref duplex) via RNA.duplexfold.
    Uses 14nt target centred on variant (seed + flanks).
    Positive = destabilised = more disruptive.
    """
    try:
        import RNA
        mid = len(ref_win) // 2
        ref_t = ref_win[max(0,mid-6):mid+8].replace("T","U")
        alt_t = alt_win[max(0,mid-6):mid+8].replace("T","U")
        mi    = mirna_seed.replace("T","U")
        ref_e = RNA.duplexfold(mi, ref_t).energy
        alt_e = RNA.duplexfold(mi, alt_t).energy
        return round(alt_e - ref_e, 3)
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vcf",    required=True,  help="Input VCF or VCF.gz (SNVs)")
    ap.add_argument("--sites",  required=True,  help="ENCORI atlas BED.gz")
    ap.add_argument("--genome", required=True,  help="GRCh38 FASTA (samtools-indexed)")
    ap.add_argument("--seeds",  required=True,  help="mirna_seeds_hsa.json")
    ap.add_argument("--model",  required=True,  help="LoRA checkpoint directory")
    ap.add_argument("--out",    required=True,  help="Output TSV path")
    ap.add_argument("--batch",  type=int, default=32,  help="Model batch size [32]")
    ap.add_argument("--no-mfe", action="store_true",   help="Skip ViennaRNA Δmfe")
    ap.add_argument("--all",    action="store_true",
                    help="Score all overlapping variants, not only seed pos 2-8")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Combined score cutoff for DISRUPTIVE call [0.5]")
    args = ap.parse_args()

    t0 = time.time()

    # ── Validate inputs ────────────────────────────────────────────────────────
    for p, name in [(args.vcf,"--vcf"),(args.sites,"--sites"),
                    (args.genome,"--genome"),(args.seeds,"--seeds"),
                    (args.model,"--model")]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {name} path not found: {p}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ── Load seed lookup ───────────────────────────────────────────────────────
    print("[1/5] Loading miRNA seeds...", flush=True)
    with open(args.seeds) as f:
        seeds_raw = json.load(f)

    # Build two lookups: exact name → seed, and lowercase→seed for fuzzy match
    seed_by_name  = {name: d["seed"] for name, d in seeds_raw.items()}
    seed_by_lower = {name.lower(): d["seed"] for name, d in seeds_raw.items()}
    print(f"      {len(seed_by_name):,} miRNA seeds loaded", flush=True)

    def get_seed(mirna_name: str) -> str | None:
        if mirna_name in seed_by_name:
            return seed_by_name[mirna_name]
        lo = mirna_name.lower()
        if lo in seed_by_lower:
            return seed_by_lower[lo]
        # partial match: hsa-miR-X-5p might be stored as miR-X-5p etc.
        for k, v in seed_by_lower.items():
            if lo in k or k in lo:
                return v
        return None

    # ── Intersect VCF with sites ───────────────────────────────────────────────
    print("[2/5] Intersecting VCF with ENCORI atlas...", flush=True)
    vcf_records = vcf_to_bed(args.vcf)
    print(f"      {len(vcf_records):,} SNVs in VCF", flush=True)

    overlaps = intersect(vcf_records, args.sites)
    print(f"      {len(overlaps):,} variant × site overlaps found", flush=True)

    if not overlaps:
        print("\nNo overlaps found.")
        print("Check: do chromosome names match? (VCF may use 'chr1' and BED '1' or vice versa)")
        sys.exit(0)

    # ── Filter to seed positions & build examples ──────────────────────────────
    print("[3/5] Checking seed positions & extracting windows...", flush=True)
    examples = []
    skipped_nonseed = 0
    skipped_refmm   = 0

    for ov in overlaps:
        var_0 = ov["pos"] - 1
        sp = get_seed_pos(var_0, ov["site_start"], ov["site_end"], ov["strand"])

        if sp is None and not args.all:
            skipped_nonseed += 1
            continue

        rw, aw = make_windows(
            args.genome, ov["chrom"], ov["pos"], ov["ref"], ov["alt"]
        )
        if rw is None:
            skipped_refmm += 1
            continue

        seed_seq = get_seed(ov["mirna"])
        wc = check_wc(ov["ref"], ov["alt"], seed_seq, sp, ov["strand"]) \
             if (seed_seq and sp) else None

        examples.append({**ov, "seed_pos": sp, "seed_seq": seed_seq,
                         "ref_win": rw, "alt_win": aw, "wc_disrupted": wc})

    print(f"      {len(examples):,} examples at seed positions", flush=True)
    if skipped_nonseed:
        print(f"      {skipped_nonseed:,} skipped (outside seed positions 2-8)", flush=True)
    if skipped_refmm:
        print(f"      {skipped_refmm:,} skipped (reference mismatch)", flush=True)

    if not examples:
        print("\nNo scorable variants. Exiting.")
        sys.exit(0)

    # ── Model inference ────────────────────────────────────────────────────────
    print("[4/5] Running model inference...", flush=True)
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}", flush=True)
    tok, mdl = load_model(args.model, device)

    pairs = [(e["ref_win"], e["alt_win"]) for e in examples]
    t_inf = time.time()
    model_scores = score_batch(tok, mdl, pairs, device, args.batch)
    print(f"      {len(model_scores):,} variants scored in {time.time()-t_inf:.1f}s",
          flush=True)

    # ── ViennaRNA Δmfe ────────────────────────────────────────────────────────
    mfe_scores = [None] * len(examples)
    if not args.no_mfe:
        print("[4b]  Computing Δmfe (ViennaRNA)...", flush=True)
        n_mfe = 0
        for i, e in enumerate(examples):
            if e["seed_seq"]:
                mfe_scores[i] = delta_mfe(e["seed_seq"], e["ref_win"], e["alt_win"])
                if mfe_scores[i] is not None:
                    n_mfe += 1
        if n_mfe > 0:
            print(f"      Δmfe computed for {n_mfe:,} variants", flush=True)
        else:
            print("      ViennaRNA not available — Δmfe skipped", flush=True)

    # ── Combine & write ────────────────────────────────────────────────────────
    print("[5/5] Writing output...", flush=True)
    ALPHA, BETA = 0.4, 0.6   # physics weight, model weight

    rows = []
    for e, ms, dmfe in zip(examples, model_scores, mfe_scores):
        wc = e["wc_disrupted"]
        physics = 1.0 if wc is True else (0.0 if wc is False else 0.5)
        combined = round(ALPHA * physics + BETA * ms, 4)

        tier = "HIGH" if combined >= 0.8 else \
               ("MEDIUM" if combined >= args.threshold else "LOW")

        rows.append({
            "variant_id":     e["variant_id"],
            "chrom":          e["chrom"],
            "pos":            e["pos"],
            "ref":            e["ref"],
            "alt":            e["alt"],
            "af":             e["af"],
            "site_id":        e["site_name"],
            "miRNA":          e["mirna"],
            "gene":           e["gene"],
            "strand":         e["strand"],
            "site_type":      e["site_type"],
            "seed_pos":       e["seed_pos"],
            "wc_disrupted":   wc,
            "delta_mfe":      dmfe,
            "model_score":    round(ms, 4),
            "combined_score": combined,
            "tier":           tier,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False, float_format="%.4f")

    elapsed = time.time() - t0
    n_hi = (df["tier"]=="HIGH").sum()
    n_me = (df["tier"]=="MEDIUM").sum()
    n_lo = (df["tier"]=="LOW").sum()

    print(f"\n{'='*55}")
    print(f"SeedBreaker scoring complete  ({elapsed:.0f}s)")
    print(f"  Variants scored : {len(df):,}")
    print(f"  HIGH tier       : {n_hi:,}  (combined ≥ 0.8)")
    print(f"  MEDIUM tier     : {n_me:,}  (combined ≥ {args.threshold})")
    print(f"  LOW tier        : {n_lo:,}  (combined < {args.threshold})")
    print(f"  Output          : {args.out}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
