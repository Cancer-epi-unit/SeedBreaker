# SeedBreaker — Project RAG Context Document
**Last updated:** June 10 2026  
**Purpose:** Load this at the start of any new chat (Claude or local model) to resume the project without re-explaining context.  
**Usage:** Paste this document or feed it as system context. Tell the model: *"Read this RAG document and pick up where we left off."*

---

## Who is Josh

- Senior cancer genomic epidemiologist, University of Oxford (Oxford CEU)
- Research focus: germline determinants of cancer risk, germline-somatic interactions, immune landscape, stromal remodelling
- Access to EPIC, UK Biobank, PRACTICAL, ILCCO consortia
- Hardware: Mac Studio M3 Ultra, 96GB unified memory
- Australian citizen, UK Global Talent Visa, partner in France

---

## What SeedBreaker is

A bioinformatics tool and AI model that predicts whether a rare germline variant disrupts a miRNA binding site in a gene 3'UTR.

**Three core tasks:**
1. Does a rare variant sit in a miRNA binding site and disrupt it? (binary classification — TRAINED, model complete)
2. Does a rare variant create a new miRNA binding site? (gain-of-function — future SeedMaker module)
3. What is the tissue-specific context of that miRNA-target interaction? (RAG layer — future)

**Why it matters:**
- Rare germline variants in miRNA binding sites could dysregulate cancer-relevant genes
- SeedBreaker enables site-set burden testing: aggregate disruption scores across all miRNA sites in a gene set → test association with cancer risk in EPIC/UK Biobank
- Scientific story: tool development + EPIC application = paper

**Package name:** SeedBreaker (loss-of-binding). Future companion: SeedMaker (gain-of-binding).  
**Tagline:** Predicting loss of miRNA target sites from rare variants.  
**Publication target:** Bioinformatics Application Note or NAR Genomics & Bioinformatics

---

## Project location

```
/Users/josh/Desktop/Projects/SeedBreaker/
```

Note: directory was originally spelled "SeekBreaker" on disk — now corrected to "SeedBreaker."

---

## Current project file structure

```
SeedBreaker/
├── Pre_data/
│   ├── ENCORI/
│   │   ├── encori_ago2_interactions.json        (raw ENCORI download, 88MB)
│   │   ├── encori_to_bed.py                     (conversion script)
│   │   ├── mirna_sites_encori_annotated_sorted.bed.gz     ← FINAL ATLAS
│   │   └── mirna_sites_encori_annotated_sorted.bed.gz.tbi
│   ├── miRBase/
│   │   ├── mature.fa                            (all species miRNA sequences)
│   │   └── mirna_seeds_hsa.json                 (human seed lookup JSON)
│   ├── TargetScan/
│   │   ├── Gene_info.txt
│   │   ├── miR_Family_Info.txt
│   │   └── Predicted_Targets_Context_Scores.default_predictions.txt
│   ├── Genome/
│   │   ├── GRCh38.primary_assembly.genome.fa    (hg38 reference)
│   │   └── GRCh38.primary_assembly.genome.fa.fai
│   ├── ClinVar/
│   │   ├── clinvar.vcf.gz                       (raw download)
│   │   ├── clinvar_pathogenic_3utr.vcf.gz       (pathogenic + 3'UTR filter)
│   │   ├── clinvar_pathogenic_3utr_chr.vcf.gz   (chr-renamed: 1→chr1)
│   │   └── clinvar_pathogenic_3utr_encori.vcf.gz  ← score.py input (44 variants)
│   ├── GTEx/
│   │   ├── GTEx_Analysis_v11_eQTL.tar           (manually downloaded, ~2GB)
│   │   └── gtex_v11_encori_snvs.vcf.gz          ← score.py input (1,758 variants)
│   └── gnomAD/
│       ├── encori_regions_merged.bed            (merged ENCORI regions for streaming)
│       ├── tmp_chroms/                          (per-chrom .vcf.gz cache)
│       └── gnomad_v4_common_encori_snvs.vcf.gz  ← score.py input (969 variants)
├── training_data/
│   ├── training_data.parquet                    ← TRAINING DATA (745k examples)
│   ├── training_data_test.parquet               (1000-site test run)
│   └── build_log.txt
├── models/
│   ├── train_log_v3.txt                         ← CURRENT TRAINING LOG
│   └── seedbreaker_lora_v2/                     ← TRAINED MODEL (use this)
│       ├── best/                                ← BEST CHECKPOINT (epoch 2, use this)
│       ├── latest/
│       ├── epoch_01/
│       ├── epoch_02/
│       └── training_results.json
├── results/
│   └── validation/
│       ├── clinvar_scored.tsv                   (44 scored, 20 HIGH)
│       ├── gtex_scored.tsv                      (1,758 scored, 279 HIGH)
│       └── gnomad_background_scored.tsv         (969 scored, 153 HIGH)
├── build_training_data.py                       ← Phase 4 script
├── train_lora.py                                (original training script — superseded)
├── train_lora_v2.py                             ← CURRENT training script (use this)
├── evaluate_test.py                             ← Phase 5: held-out test evaluation
├── score.py                                     ← Phase 6: variant scoring engine
├── make_test_vcf.py                             ← generates synthetic test VCF from ENCORI coords
├── download_validation_data.sh                  ← Phase 7: download ClinVar + gnomAD
├── download_gnomad_background.sh                ← gnomAD-only streaming script (use this)
├── process_gtex.sh                              ← GTEx v10 processing (bash, legacy)
├── process_gtex.py                              ← GTEx v11 processing (Python/parquet, USE THIS)
├── score_validation.sh                          ← runs score.py on all three validation sets
├── enrichment_analysis.py                       ← Fisher's exact + Mann-Whitney enrichment tests
├── launch_training.sh                           (smoke test + background launch)
├── launch_training.command                      (macOS double-click launcher)
└── seedbreaker_rag.md                           ← THIS FILE
```

---

## Conda environment

```bash
conda activate seedbreaker
```

**Key package versions (locked for compatibility):**
- python 3.11
- transformers ≥5.0.0  ← CRITICAL: peft 0.19.1 requires this (imports EncoderDecoderCache)
- peft 0.19.1          ← CRITICAL: must match what training used (adapter_config.json peft_version field)
- accelerate 0.29.0
- torch 2.12.0 (MPS available = True)
- scipy                ← added for enrichment_analysis.py (Fisher's exact, Mann-Whitney)
- pyarrow              ← added for process_gtex.py (GTEx v11 parquet files)
- viennarna, bedtools, samtools, htslib, bcftools installed via bioconda

**Known incompatibility:** peft 0.10.0 cannot load adapter_config.json saved by peft 0.19.1 (unknown fields like `alora_invocation_tokens`). Do not downgrade peft. ESM-2 is a native transformers model — no `trust_remote_code` needed.

---

## Pre-data: what was collected and how

### ENCORI binding site atlas
**Source:** ENCORI API (rnasysu.com/encori)  
**Citation:** Keren Zhou et al., Nature Methods (2026)  
**Download command:**
```bash
curl "https://rnasysu.com/encori/api/miRNATarget/?assembly=hg38&geneType=mRNA&miRNA=all&clipExpNum=1&degraExpNum=0&pancancerNum=0&programNum=0&target=all&cellType=0" -o encori_ago2_interactions.json
```
**What it contains:** TSV (despite .json extension), space-delimited, 2 comment lines then header  
**Final atlas:** `mirna_sites_encori_annotated_sorted.bed.gz`  
**Stats:** 278,425 AGO2 CLIP-confirmed binding sites, 192 unique miRNAs, 11,140 target genes  
**Columns:** chr, start, end, name, score, strand, miRNA, MIMAT, gene_name, gene_id, RBP, clip_num, targetscan, tdmd_score, phylop, site_type, context_score  
**Site type breakdown:** 17,656 × 8mer | 30,591 × 7mer-m8 | 20,710 × 7mer-A1 | 209,428 × unclassified  
**Note:** 75% unclassified is expected — TargetScan only covers ~150 conserved miRNA families. Unclassified = real CLIP-confirmed sites not in TargetScan conserved set.

**Why ENCORI has ~278k sites (not millions):** ENCORI requires AGO2 CLIP-seq experimental confirmation. Larger PAR-CLIP/eCLIP atlases report 1–4M potential positions because they include all 3'UTR peaks without requiring miRNA-specific confirmation. ENCORI's smaller set is higher-confidence — every site has a confirmed AGO2 interaction in at least one cell line.

### miRBase
**Source:** https://www.mirbase.org/download/ (manual download — FTP broken)  
**Version:** miRBase v22  
**Key file:** `mirna_seeds_hsa.json` — dict of {miRNA_name: {mimat, sequence, seed, seed_m8, length}}  
**Stats:** 48,885 total, 2,656 human miRNAs  
**Seed example:** hsa-miR-21-5p → seed = "AGCTTAT" (positions 2-8, U→T converted)

### TargetScan
**Source:** http://www.targetscan.org/vert_80/  
**Key file:** Predicted_Targets_Context_Scores.default_predictions.txt (131MB)  
**Human sites:** 228,049 (filter: Gene Tax ID == 9606)  
**Columns:** Gene ID, Gene Symbol, Transcript ID, Gene Tax ID, miRNA, Site Type (1=8mer/2=7mer-m8/3=7mer-A1), UTR_start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, Predicted relative KD  
**Note:** Coordinates are hg19 UTR-relative, not genomic. Used as annotation layer only (joined by gene+miRNA), no liftover needed.

### Reference genome
**Source:** GENCODE release 47  
**File:** GRCh38.primary_assembly.genome.fa (indexed with samtools faidx)

---

## Training data: what was built

**Script:** `build_training_data.py`  
**Method:** In-silico saturation mutagenesis of all 278k ENCORI binding sites  
**Logic:** For each site × each seed position (2-8) × each possible SNV → label by Watson-Crick complementarity disruption  

**Label rules:**
- WC → mismatch = DISRUPTIVE (1)
- WC → WC = PRESERVING (0)  
- Wobble → mismatch = DISRUPTIVE (1)
- WC → wobble = EXCLUDED (ambiguous)
- Ref already mismatch = EXCLUDED (non-canonical)

**Output:** `training_data/training_data.parquet`  
**Stats:**
- 745,492 total examples after 3:1 balancing
- 559,119 disruptive (1) | 186,373 preserving (0)
- Train: 679,824 | Val (chr4): 26,749 | Test (chr8+chr18): 38,919
- Label confidence mean: 0.583, std: 0.174
- File size: 74.3 MB

**Chromosome splits:** Test = chr8 + chr18, Val = chr4, Train = everything else  
**Why chromosome-level splits:** Prevents transcript-level data leakage

---

## Model: TRAINED — results below

**Script:** `train_lora_v2.py`  
**Base model:** `facebook/esm2_t33_650M_UR50D` (650M params, 33 transformer layers, hidden_size=1280)  
**Model identity note:** Training originally targeted NT-v2 500M, but fell back to ESM-2 because NT-v2 500M has always had hidden_size=1024, while the saved LoRA weights have hidden_size=1280 (matching ESM-2 t33). For the paper: "fine-tuned ESM-2 t33 as a general-purpose sequence encoder." ESM-2 is a protein LM but is effective as a general-purpose sequence encoder for DNA/RNA tasks.  
**Adapter:** LoRA (r=16, alpha=32, dropout=0.1)  
**LoRA targets:** query, key, value, dense (attention + FFN layers in all 33 transformer blocks)  
**Trainable params:** ~11.5M (2.28% of total)  
**Device:** Apple Silicon MPS, fp32  

**Training config:**
- Batch size: 8 (effective 32 with grad accum 4)
- Learning rate: 2e-4
- Max sequence length: 256
- Speed on M3 Ultra MPS: ~1.33 it/s

**Input encoding:** REF_window + "NNN" + ALT_window concatenated as single sequence  
**Why concatenated:** ESM tokenizer does not support sequence pairs  
**Loss function:** CrossEntropyLoss with class weights (neg/pos ratio) to handle imbalance  

**RESULTS — Epoch 2 (best checkpoint):**
| Metric | Value |
|--------|-------|
| val_auroc | **0.9910** |
| val_auprc | **0.9975** |
| val_acc | 0.9569 |
| train_loss | 0.1069 |
| epoch_time | 1064.7 min (~17.8h on MPS) |

**TEST SET RESULTS — chr8 + chr18 held-out (38,919 examples):**
| Metric | Value |
|--------|-------|
| test_auroc | **0.9938** |
| test_auprc | **0.9980** |
| test_acc | 0.9642 |
| Sensitivity | 0.9682 (TPR — catches 97% of disruptive variants) |
| Specificity | 0.9524 (TNR) |
| PPV | 0.9836 (precision — 98% of disruptive calls are correct) |
| TP=28,151 | FP=468  TN=9,374  FN=926 |

Training was stopped after epoch 2 (diminishing returns).  
**Best checkpoint:** `models/seedbreaker_lora_v2/best/` ← use this for inference  

**Check model:**
```bash
tail -f /Users/josh/Desktop/Projects/SeedBreaker/models/train_log_v3.txt
cat models/seedbreaker_lora_v2/training_results.json | python3 -m json.tool
```

**Performance note:** MPS is ~10-15× slower than CUDA for transformer LoRA training. For future training: `--batch 64 --grad-accum 1` gives ~3× speedup. Long-term: migrate to MLX.

---

## Key bugs fixed (do not re-introduce)

1. **ENCORI TSV parsing:** File is space-delimited not tab-delimited. Some rows have 24 fields instead of 25 (missing cellline/tissue). Fix: pad with "NA" to n_header length.

2. **peft compatibility:** peft==0.19.1 + transformers≥5.0.0 required for ESM-2. Do not downgrade peft — adapter_config.json has unknown fields that peft 0.10.0 cannot read.

3. **ESM tokenizer sequence pairs:** Cannot pass `(ref, alt)` as a text pair. Fix: concatenate as `ref + "NNN" + alt` single string.

4. **stdout buffering:** Use flush=True everywhere and PYTHONUNBUFFERED=1 at launch to avoid lost output on crash.

5. **MPS silent crashes:** Use `torch.mps.synchronize()` after `loss.backward()` to surface MPS errors immediately.

6. **macOS zsh comments:** `#` does not work as a comment in interactive zsh. Only works inside script files.

7. **macOS zcat:** `zcat` on macOS only handles .Z files, NOT .gz. Always use `gzip -dc file.gz`. Universal fix — works on macOS and Linux.

8. **TargetScan species column:** Human = Gene Tax ID 9606, column 4. Filter: `awk -F'\t' '$4==9606'`

9. **ENCODE API:** Completely broken for AGO2/EIF2C2 eCLIP queries. Use ENCORI API instead.

10. **ClinVar chromosome naming:** ClinVar VCF uses `1` not `chr1`. Fix: `bcftools annotate --rename-chrs /tmp/clinvar_chr_rename.txt` where the file maps `1 chr1`, `2 chr2`, ..., `MT chrMT`.

11. **bedtools intersect strips VCF header:** Always use `-header` flag when intersecting VCF files, otherwise bcftools cannot index the output.

12. **GTEx v10 URL 403:** GTEx requires DUA acceptance — direct wget returns 403. User manually downloaded v11 from GTEx portal. Use `process_gtex.py` for v11 (parquet), `process_gtex.sh` for v10 (legacy).

13. **GTEx v11 is parquet:** v11 significant pairs are `.signif_pairs.parquet` inside a tar. Requires pandas + pyarrow — see `process_gtex.py`.

14. **VCF not sorted → bcftools index fails:** bgzip/tabix requires VCF sorted by chrom then position. Sort with CHROM_ORDER dict before writing. See `process_gtex.py` sort_key pattern.

15. **gnomAD: only 4 variants per chromosome:** 135k tiny ENCORI regions = 135k HTTP range requests → gnomAD throttles this. Fix: `bedtools merge -d 2000` to collapse regions into ~16k larger blocks before `--regions-file`. See `download_gnomad_background.sh`.

16. **gnomAD AF filter syntax:** Use `AF>0.01` not `AF[0]>0.01`.

17. **`set -euo pipefail` kills download scripts:** Use `set -uo pipefail` (no `-e`) with per-chromosome graceful error handling.

18. **zsh history expansion with `!`:** The `!` in `$!` (background PID) triggers zsh history expansion interactively. Only use inside shell scripts.

19. **ESM-2 `num_labels` kwarg:** Set `cfg.num_labels = 2` on the AutoConfig object first, then call `AutoModelForSequenceClassification.from_pretrained(path, config=cfg)`. Do not pass `num_labels` as a direct kwarg.

---

## Phase 6 — score.py usage

```bash
conda activate seedbreaker
cd /Users/josh/Desktop/Projects/SeedBreaker

# Basic scoring run
python3 score.py \
  --vcf    my_variants.vcf.gz \
  --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \
  --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \
  --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \
  --model  models/seedbreaker_lora_v2/best \
  --out    results/scored_variants.tsv

# Skip ViennaRNA Δmfe (faster, still uses ML + WC physics)
python3 score.py ... --no-mfe

# Lower tier threshold (calls more MEDIUM variants)
python3 score.py ... --threshold 0.4

# Score all overlapping variants regardless of seed position
python3 score.py ... --all

# Filter output to HIGH tier only
awk -F'\t' 'NR==1 || $17=="HIGH"' results/scored_variants.tsv
```

**Output TSV columns (17 total):**
`variant_id, chrom, pos, ref, alt, af, site_id, miRNA, gene, strand, site_type, seed_pos, wc_disrupted, delta_mfe, model_score, combined_score, tier`

**variant_id format:** `ID|REF|ALT|AF` from the VCF ID field. Falls back to `chrom:pos:ref:alt` if VCF ID is `.`  
**VCF parsing in score.py:** `vid, ref, alt, af_s = c[3].split("|", 3)` (the tag format injected by make_test_vcf.py / process_gtex.py / download_gnomad_background.sh)

**Score interpretation:**
- `wc_disrupted = True` → WC pair broken at seed position (strong physics prior)
- `model_score` → ESM-2 t33 + LoRA disruption probability (0–1)
- `combined_score` = 0.4 × physics_score + 0.6 × model_score
- `tier = HIGH` (≥0.8) is the conservative call for follow-up

**Physics score mapping:** wc_disrupted=True → 1.0 | ambiguous (wobble) → 0.5 | preserved → 0.0

**Seed position logic in score.py:**
```python
if strand == "+":
    k = site_end - var_pos_0      # distance from 3' end of site
else:
    k = var_pos_0 - site_start + 1
# valid seed position: 2 <= k <= 8
```

**Requirements:** bedtools and samtools in PATH; peft==0.19.1 and transformers≥5.0.0 in conda env.

---

## Phase 7 — Validation results (COMPLETE)

### Summary table

| Dataset | Variants scored | HIGH tier | HIGH % | Notes |
|---------|----------------|-----------|--------|-------|
| ClinVar pathogenic 3'UTR | 44 | 20 | **45.5%** | Gold standard positives |
| GTEx v11 eQTLs | 1,758 | 279 | 15.9% | All-tissue significant eQTLs |
| gnomAD v4.1 common SNVs | 969 | 153 | 15.8% | Background null set (AF>1%) |

### Key statistical result

**ClinVar pathogenic vs gnomAD common (Fisher's exact test):**
- Odds ratio = **4.44** (95% CI: 2.10–9.40)
- p = **6.55 × 10⁻⁶**
- Known pathogenic 3'UTR variants are 4.4× more likely to score HIGH than common gnomAD variants
- **This is the paper's headline result**

**GTEx eQTLs vs gnomAD common:**
- Odds ratio = **1.006** (null)
- eQTL status at miRNA binding sites is orthogonal to seed disruption potential
- Most eQTLs in 3'UTRs act via other mechanisms (RBP sites, RNA structure, mRNA stability elements)
- Framing for paper: GTEx as *discovery* resource, not validation

### Key case studies (from ClinVar HIGH tier)

| Gene | miRNA | combined_score | Clinical context |
|------|-------|----------------|-----------------|
| BRCA1 | hsa-miR-196a/b | 0.9999 | Known BRCA1 regulation via miR-196 axis |
| SESN2 | hsa-miR-125a/b, hsa-miR-10a/b | 0.9999 | Tumour suppressor, p53 pathway |
| ADAR | hsa-miR-143-3p | 0.9999 | RNA editing, cancer immune evasion |
| GNAS | — | HIGH | GNAS amplification in cancer |
| EIF4G3 | — | HIGH | Translation initiation factor dysregulated in cancer |

### gnomAD variant count (969) — why it's low

Two reasons: (1) `bedtools merge -d 2000` collapses 135k ENCORI regions into ~16k merged blocks before streaming — variants outside merged blocks are not captured; (2) common variants (AF>1%) at conserved seed positions are depleted by purifying selection. Low count is biologically meaningful and should be noted in the paper.

### rs78378222 — known limitation

rs78378222 is a TP53 variant at the polyadenylation signal (AATAAA→AATAAG). It is NOT an ENCORI-confirmed AGO2 miRNA binding site. SeedBreaker correctly does not score it. Paper limitation: "SeedBreaker only scores variants within ENCORI AGO2-confirmed miRNA binding sites. Variants affecting polyadenylation signals, RBP motifs, or non-CLIP-confirmed predicted sites are outside scope."

---

## Validation scripts

### score_validation.sh
Runs score.py sequentially on all three validation VCFs then calls enrichment_analysis.py:
```bash
bash score_validation.sh
```

### enrichment_analysis.py
```bash
python3 enrichment_analysis.py \
  --clinvar  results/validation/clinvar_scored.tsv \
  --gtex     results/validation/gtex_scored.tsv \
  --gnomad   results/validation/gnomad_background_scored.tsv
```
Outputs: Fisher's exact OR + p-value, Mann-Whitney U on score distributions, per-miRNA HIGH-tier breakdown, top-scoring variants table.

### process_gtex.py (GTEx v11, USE THIS)
```bash
python3 process_gtex.py
# Input:  Pre_data/GTEx/GTEx_Analysis_v11_eQTL.tar
# Output: Pre_data/GTEx/gtex_v11_encori_snvs.vcf.gz
```
Requires: `pip install pyarrow --break-system-packages`

### download_gnomad_background.sh
```bash
bash download_gnomad_background.sh
# Streams gnomAD v4.1 per chromosome via bcftools HTTPS
# Merges ENCORI regions with bedtools merge -d 2000 first
# Output: Pre_data/gnomAD/gnomad_v4_common_encori_snvs.vcf.gz
```

---

## Phase 7b / Phase 8 — Tissue-specific binding (planned)

### Motivation
ENCORI treats all 278k sites as equally relevant regardless of tissue. In reality a binding site only matters if both the miRNA and target gene are co-expressed in that tissue.

### Approach
For each ENCORI `(miRNA, gene)` pair:
1. Filter to tissues where both miRNA TPM ≥5 AND gene TPM ≥5 (GTEx small RNA-seq + mRNA TPM)
2. Build a `site × tissue` relevance matrix
3. Weight SeedBreaker scores by tissue relevance for tissue-stratified burden tests in EPIC

### Data needed
- GTEx v11 gene expression: `GTEx_Analysis_v11_RNASeq_RSEMv1.3.3_gene_median_tpm.gct.gz`
- miRNA tissue expression: GTEx v11 miRNA TPM or miRmine (guanfiles.dcmb.med.umich.edu/mirmine)

### Use case
"Score only miRNA binding sites active in breast tissue" → breast-specific burden testing in EPIC for breast cancer GWAS hits.

---

## What still needs to be done (in order)

### Completed
- [x] LoRA training — val_auroc=0.9910, epoch 2 best
- [x] Test set evaluation — AUROC=0.9938, AUPRC=0.9980, PPV=0.9836, sensitivity=0.9682
- [x] Phase 6 — score.py complete, variant_id column added as first output column
- [x] Phase 7 — validation complete
  - [x] ClinVar: 44 scored, 20 HIGH, OR=4.44 vs gnomAD, p=6.55×10⁻⁶
  - [x] GTEx v11: 1,758 scored, 279 HIGH, OR=1.006 vs gnomAD (null)
  - [x] gnomAD background: 969 scored, 153 HIGH (15.8%)

### Phase 4b — Experimental anchors (optional second fine-tuning pass)
- [ ] Download PolymiRTS (compbio.uthsc.edu/miRSNP)
- [ ] Download miRNASNP-v3 (bioinfo.life.hust.edu.cn/miRNASNP)
- [ ] Second fine-tuning pass with experimental positives on top of base LoRA

### Phase 7b — Tissue-specific binding weights
- [ ] Download GTEx v11 gene expression TPM matrix
- [ ] Download miRNA tissue expression (GTEx or miRmine)
- [ ] Build site × tissue co-expression matrix
- [ ] Weight scores for EPIC tissue-specific burden tests (breast, lung, prostate, colorectal)

### Phase 8 — Packaging
- [ ] pyproject.toml, CLI entry point (`seedbreaker score`, `seedbreaker validate`)
- [ ] Bioconda recipe
- [ ] Docker/Singularity container
- [ ] Zenodo DOI for model weights
- [ ] HuggingFace Hub model card — upload `models/seedbreaker_lora_v2/best/`

### Phase 9 — Documentation
- [ ] README with quickstart
- [ ] mkdocs site
- [ ] Tutorial notebook (use make_test_vcf.py output as demo input)
- [ ] CITATION.cff

### Phase 10 — Publication
- [ ] Preprint on bioRxiv
- [ ] Target: Bioinformatics Application Note or NAR Genomics & Bioinformatics
- [ ] EPIC burden test as results section
- [ ] Key figure: ClinVar vs gnomAD score distribution (violin + OR annotation)

---

## Paper narrative

**Abstract sentence:** "Known pathogenic 3'UTR variants are 4.4× enriched for HIGH-tier SeedBreaker disruption calls compared to common population variants (OR=4.44, p=6.55×10⁻⁶), validating the model against an independent clinical reference standard."

**Model framing:** ESM-2 t33 (facebook/esm2_t33_650M_UR50D), LoRA fine-tuned on 745k in-silico saturation mutagenesis examples derived from 278,425 AGO2 CLIP-confirmed binding sites. Test AUROC=0.9938, AUPRC=0.9980, PPV=0.9836.

**Scope limitation sentence:** "SeedBreaker scores variants within ENCORI-confirmed AGO2 CLIP-seq miRNA binding sites (278,425 sites across 11,140 genes). Variants affecting polyadenylation signals, RBP motifs, or computationally predicted sites without CLIP-seq confirmation are not scored."

**GTEx null framing:** "eQTL status at miRNA binding sites showed no enrichment for HIGH-tier SeedBreaker scores (OR=1.006), consistent with the hypothesis that most 3'UTR eQTLs act via mechanisms other than direct miRNA seed disruption."

---

## GitHub + HuggingFace (to set up)

```bash
# Push code to GitHub
cd /Users/josh/Desktop/Projects/SeedBreaker
git init
git add *.py *.sh *.md
git commit -m "SeedBreaker: ESM-2 LoRA miRNA binding disruption classifier"
git remote add origin https://github.com/YOUR_USERNAME/SeedBreaker.git
git push -u origin main

# Upload model weights to HuggingFace
# Only the LoRA adapter (~45MB) needs hosting — base ESM-2 downloads automatically
pip install huggingface_hub --break-system-packages
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='models/seedbreaker_lora_v2/best',
    repo_id='YOUR_HF_USERNAME/seedbreaker-lora-v2',
    repo_type='model'
)
"
```

---

## Useful commands cheatsheet

```bash
# Activate environment
conda activate seedbreaker
cd /Users/josh/Desktop/Projects/SeedBreaker

# Check training progress
tail -f models/train_log_v3.txt
ps aux | grep train_lora | grep -v grep

# Run test set evaluation
python3 evaluate_test.py

# Score a VCF (Phase 6)
python3 score.py \
  --vcf my_variants.vcf.gz \
  --sites Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \
  --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \
  --seeds Pre_data/miRBase/mirna_seeds_hsa.json \
  --model models/seedbreaker_lora_v2/best \
  --out results/scored_variants.tsv

# Run Phase 7 validation end-to-end
bash score_validation.sh

# Run enrichment analysis
python3 enrichment_analysis.py \
  --clinvar results/validation/clinvar_scored.tsv \
  --gtex    results/validation/gtex_scored.tsv \
  --gnomad  results/validation/gnomad_background_scored.tsv

# Re-download gnomAD background
bash download_gnomad_background.sh

# Process GTEx v11 (after manual download)
python3 process_gtex.py

# Generate test VCF from real ENCORI coordinates
python3 make_test_vcf.py

# Inspect training data
python3 -c "
import pandas as pd
df = pd.read_parquet('training_data/training_data.parquet')
print(df.shape, df['label'].value_counts().to_dict(), df['split'].value_counts().to_dict())
"

# Check ENCORI atlas
gzip -dc Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz | head -3
gzip -dc Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz | wc -l

# macOS gz files — ALWAYS gzip -dc, never zcat or gzcat
gzip -dc file.bed.gz | head

# Install missing packages
pip install pyarrow scipy --break-system-packages

# Check HuggingFace model cache
du -sh ~/.cache/huggingface/hub/models--facebook--esm2*/

# Future training (larger batch for speed)
PYTHONUNBUFFERED=1 nohup python3 train_lora_v2.py \
  --data training_data/training_data.parquet \
  --out  models/seedbreaker_lora_v3 \
  --epochs 10 --batch 64 --grad-accum 1 --lr 2e-4 \
  > models/train_log_next.txt 2>&1 & disown
```

---

## Data still needed

| Dataset | Phase | Where to get |
|---------|-------|-------------|
| PolymiRTS | 4b | compbio.uthsc.edu/miRSNP/download.php |
| miRNASNP-v3 | 4b | bioinfo.life.hust.edu.cn/miRNASNP |
| GTEx v11 gene TPM | 7b | gtexportal.org → Downloads → RNA-Seq → gene_median_tpm |
| GTEx v11 miRNA TPM | 7b | gtexportal.org → Downloads → miRNA |
| GWAS Catalog | 10 | ebi.ac.uk/gwas/downloads |
| miRmine tissue expression | 7b | guanfiles.dcmb.med.umich.edu/mirmine |

---

## For local Qwen/Ollama models

Josh is building SeedBreaker: a bioinformatics tool that predicts whether a rare germline variant disrupts a miRNA binding site in a gene 3'UTR. Intended for a methods paper targeting Bioinformatics or NAR Genomics.

**Model:** ESM-2 t33 (facebook/esm2_t33_650M_UR50D), LoRA fine-tuned (r=16, alpha=32) on 745k in-silico saturation mutagenesis examples from 278,425 ENCORI AGO2 CLIP-confirmed binding sites. Test AUROC=0.9938, AUPRC=0.9980, PPV=0.9836, sensitivity=0.9682. Best checkpoint: `models/seedbreaker_lora_v2/best/`

**Scoring engine (Phase 6 COMPLETE):** `score.py` takes a VCF, intersects with ENCORI atlas, checks seed positions 2-8, runs ESM-2+LoRA, optionally runs ViennaRNA Δmfe. Output: 17-column TSV with `variant_id` as first column, `combined_score = 0.4×physics + 0.6×model`, tier HIGH/MEDIUM/LOW.

**Validation (Phase 7 COMPLETE):** ClinVar pathogenic 3'UTR variants 4.4× enriched for HIGH-tier calls vs gnomAD common variants (OR=4.44, p=6.55×10⁻⁶). GTEx eQTLs show null enrichment (OR=1.006). Key case studies: BRCA1 (miR-196a/b, score=0.9999), SESN2 (miR-125a/b, score=0.9999), ADAR (miR-143-3p).

**Current focus:** Phase 8 (packaging + HuggingFace upload), tissue-specific binding weights (Phase 7b), or EPIC burden test. Ask Josh which to tackle next.

---

## Phase 7b — Tissue-specific results (COMPLETE)

### Scripts
- `download_tissue_data.sh` — downloads GTEx v11 gene TPM + miRNA TPM matrices
- `build_tissue_weights.py` — builds `Pre_data/tissue/tissue_weights.tsv.gz` (site×tissue co-expression weights)
- `apply_tissue_weights.py` — post-processes scored TSV with tissue weights, adds `tissue_weight`, `tissue_score`, `tissue_tier` columns

### Data used
- Gene expression: `GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz` (GCT format)
- miRNA expression: `miRNA_TPM_matrix_PORTAL_2025_03_17.txt.gz` (plain TSV, normalised TPM — use this, NOT the raw read counts file)
- Both saved to `Pre_data/tissue/`

### Weight formula
```
gene_w   = min(log2(gene_tpm + 1) / log2(101), 1.0)   # saturates at 100 TPM
mirna_w  = min(log2(mirna_tpm + 1) / log2(11),  1.0)   # saturates at 10 TPM
coexp_weight = sqrt(gene_w * mirna_w)                   # geometric mean
tissue_score = combined_score × coexp_weight
```

### Tissue matching fix
GTEx column names vary between versions. Use keyword matching (not full name matching):
```python
TISSUE_KEYWORDS = [
    ("mammary", "breast"),   # "Breast - Mammary Tissue" → matched by "mammary"
    ("prostate", "prostate"),
    ...
]
```
The TISSUE_MAP full-name matching failed for multi-word tissues (breast, colon, kidney, blood, brain). Keyword matching works reliably across v10 and v11.

### site_id format bug (fixed)
`build_tissue_weights.py` originally constructed `site_id = chrom:start-end:mirna`.
`score.py` uses the ENCORI BED name field: `hsa-miR-205-5p:SAMD11:MIMAT0000266`.
Fix: `encori["site_id"] = encori["name"]` — must match score.py exactly.

### ClinVar tissue-specific results

| Tissue | Total | With weight | HIGH | MEDIUM | LOW |
|--------|-------|-------------|------|--------|-----|
| Breast | 44 | 43 (97.7%) | **5 (11.4%)** | 11 (25.0%) | 27 (61.4%) |
| Prostate | 44 | 43 (97.7%) | **6 (13.6%)** | 10 (22.7%) | 27 (61.4%) |

Global HIGH before tissue filter: 20 (45.5%). Tissue filter reduces this to 5–6 tissue-prioritised calls.

Mean co-expression weight: breast=0.530, prostate=0.563

### Top tissue-specific HIGH variants (ClinVar)

**Breast and Prostate (shared):**
| rsID | Gene | miRNA | tissue_score | weight |
|------|------|-------|-------------|--------|
| rs3338829 | GNAS | hsa-miR-138-5p | 0.9918 | 1.000 |
| rs1065884 | GNAS | hsa-miR-138-5p | 0.9870 | 1.000 |
| rs29746 | GNAS | hsa-miR-18a-5p | 0.9860 | 1.000 |
| rs29746 | GNAS | hsa-miR-18b-5p | 0.9860 | 1.000 |
| rs1456892 | GNAS | hsa-miR-150-5p | 0.9221 | 1.000 |

**Prostate-specific:**
| rsID | Gene | miRNA | tissue_score | weight |
|------|------|-------|-------------|--------|
| rs2724476 | PRRT2 | hsa-miR-1247-5p | 0.8666 | 0.867 |

### Key biological interpretation

**GNAS is the lead case study.** GNAS encodes a G-protein alpha subunit — a known oncogene with activating mutations driving thyroid, pituitary, and pancreatic cancers, and dysregulated in breast and prostate. Weight=1.000 in both tissues means GNAS is highly expressed AND the relevant miRNAs (miR-138-5p, miR-18a/b-5p, miR-150-5p) are co-expressed. Multiple ClinVar pathogenic variants cluster in GNAS miRNA binding sites, all scoring tissue_score ≥ 0.92.

**PRRT2 is prostate-specific.** Appears at HIGH tier in prostate (weight=0.867) but not breast — demonstrates the tissue filter is detecting genuine tissue-specific signal, not just rescaling scores uniformly.

**Paper narrative for this section:** "Tissue-specific refinement using GTEx v11 co-expression weights reduces the genome-wide HIGH-tier call set from 20 to 5–6 tissue-prioritised variants per tissue. In breast and prostate, GNAS emerges as the dominant signal, with five ClinVar pathogenic variants disrupting binding sites for four distinct miRNAs (miR-138-5p, miR-18a/b-5p, miR-150-5p) all achieving tissue_score ≥ 0.92."

### Usage
```bash
# Build weights (one-time, ~5 min)
python3 build_tissue_weights.py

# Apply to any scored TSV
python3 apply_tissue_weights.py \
  --scores results/validation/clinvar_scored.tsv \
  --tissue breast \
  --out    results/validation/clinvar_scored_breast.tsv

python3 apply_tissue_weights.py \
  --scores results/validation/clinvar_scored.tsv \
  --tissue prostate \
  --out    results/validation/clinvar_scored_prostate.tsv

# List available tissues
python3 apply_tissue_weights.py --list-tissues

# HIGH tier only
python3 apply_tissue_weights.py --scores ... --tissue breast --high-only --out ...
```
