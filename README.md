# SeedBreaker

**Predicting germline variant disruption of miRNA binding sites in gene 3′UTRs**

SeedBreaker is a command-line tool and trained classifier that scores rare germline SNVs for their likelihood of disrupting a microRNA (miRNA) seed-region binding site in a gene 3′UTR. It combines Watson-Crick base-pairing physics with an ESM-2 LoRA deep learning classifier, validated against ClinVar pathogenic variants and gnomAD population controls.

> Known pathogenic 3′UTR variants (ClinVar) are **4.4× enriched** for HIGH-tier SeedBreaker calls versus common population variants (gnomAD v4.1), p = 6.6 × 10⁻⁶.

---

## How it works

SeedBreaker scores each input variant in three stages:

1. **Intersection** — variant is intersected with the ENCORI AGO2 CLIP-seq atlas (278,425 confirmed miRNA binding sites across 11,140 genes). Only variants falling within a confirmed binding site proceed.

2. **Seed position check** — the variant's position within the binding site is mapped to seed positions 2–8. A Watson-Crick disruption score (physics_score) is assigned: 1.0 if the variant breaks a WC pair, 0.5 if ambiguous (wobble), 0.0 if preserving.

3. **ESM-2 LoRA inference** — the reference and alternate 50-bp windows (concatenated as `ref + NNN + alt`) are passed through a LoRA-fine-tuned ESM-2 t33 classifier (650M parameters, fine-tuned on 745,492 in-silico saturation mutagenesis examples). The model outputs a disruption probability (model_score).

**Combined score:**
```
combined_score = 0.4 × physics_score + 0.6 × model_score
```

**Tiers:** HIGH ≥ 0.8 | MEDIUM ≥ 0.5 | LOW < 0.5

Optionally, ViennaRNA `duplexfold` is used to compute Δmfe between reference and alternate duplexes as a third evidence layer.

---

## Reference data

SeedBreaker requires three reference files at runtime:

| File | Source | Size |
|------|--------|------|
| ENCORI AGO2 CLIP atlas (BED) | [ENCORI API](https://rnasysu.com/encori) | ~30 MB |
| miRBase human seeds (JSON) | [miRBase v22](https://www.mirbase.org/download/) | <1 MB |
| hg38 reference genome (FASTA + FAI) | [GENCODE release 47](https://www.gencodegenes.org) | ~3 GB |

Pre-built atlas: `mirna_sites_encori_annotated_sorted.bed.gz` — 278,425 AGO2 CLIP-confirmed binding sites, 192 miRNAs, 11,140 genes (hg38).

---

## Installation

### Prerequisites
- macOS or Linux
- conda / mamba
- bedtools, samtools, bcftools, bgzip, tabix (install via bioconda)
- ViennaRNA (optional, for Δmfe)

### Conda environment

```bash
conda create -n seedbreaker python=3.11
conda activate seedbreaker

# Bioinformatics tools
conda install -c bioconda bedtools samtools htslib bcftools viennarna

# Python packages
pip install torch transformers>=5.0.0 peft==0.19.1 accelerate
pip install pandas numpy scipy pyarrow
```

### Clone and set up

```bash
git clone git@github.com:Cancer-epi-unit/SeedBreaker.git
cd SeedBreaker
```

Download reference data (ENCORI atlas, miRBase seeds, hg38 genome) and place in `Pre_data/`:

```
Pre_data/
├── ENCORI/
│   ├── mirna_sites_encori_annotated_sorted.bed.gz
│   └── mirna_sites_encori_annotated_sorted.bed.gz.tbi
├── miRBase/
│   └── mirna_seeds_hsa.json
└── Genome/
    ├── GRCh38.primary_assembly.genome.fa
    └── GRCh38.primary_assembly.genome.fa.fai
```

### Model weights

Download the trained LoRA adapter from HuggingFace:

```bash
# The base ESM-2 model (facebook/esm2_t33_650M_UR50D) downloads automatically.
# Only the LoRA adapter (~45 MB) needs to be fetched:
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Cancer-epi-unit/seedbreaker-lora-v2',
                  local_dir='models/seedbreaker_lora_v2/best')
"
```

---

## Usage

### Score a VCF

```bash
conda activate seedbreaker
cd SeedBreaker

python3 score.py \
  --vcf    variants.vcf.gz \
  --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \
  --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \
  --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \
  --model  models/seedbreaker_lora_v2/best \
  --out    results/scored_variants.tsv
```

**Options:**
```
--no-mfe       Skip ViennaRNA Δmfe (faster, no ViennaRNA required)
--threshold    MEDIUM tier threshold (default: 0.5)
--all          Score all overlapping variants, not just seed positions 2-8
```

**Input VCF requirements:** bgzipped and tabix-indexed, hg38 coordinates, `chr` prefix on chromosomes.

### Output format

17-column TSV:

| Column | Description |
|--------|-------------|
| variant_id | VCF ID field (or chrom:pos:ref:alt if ID is `.`) |
| chrom, pos, ref, alt, af | Variant coordinates and allele frequency |
| site_id | ENCORI binding site ID (miRNA:gene:MIMAT) |
| miRNA, gene, strand, site_type | Binding site annotation |
| seed_pos | Position within seed (2–8) |
| wc_disrupted | True if Watson-Crick pair broken |
| delta_mfe | ViennaRNA Δmfe (NA if --no-mfe) |
| model_score | ESM-2+LoRA disruption probability |
| combined_score | 0.4×physics + 0.6×model |
| tier | HIGH / MEDIUM / LOW |

### Filter to HIGH tier

```bash
awk -F'\t' 'NR==1 || $17=="HIGH"' results/scored_variants.tsv
```

---

## Tissue-specific scoring

Apply GTEx v11 co-expression weights to prioritise variants in tissue-active binding sites:

```bash
# Build weights once (requires GTEx expression matrices in Pre_data/tissue/)
python3 build_tissue_weights.py

# Apply to scored output
python3 apply_tissue_weights.py \
  --scores results/scored_variants.tsv \
  --tissue breast \
  --out    results/scored_variants_breast.tsv

# Available tissues
python3 apply_tissue_weights.py --list-tissues
```

The tissue_score = combined_score × co-expression weight (geometric mean of gene TPM and miRNA TPM, log-normalised). A binding site with weight=0 means the gene or miRNA is not expressed in that tissue.

---

## Validation

### Phase 7 — Independent dataset benchmarking

```bash
# Download validation data
bash download_validation_data.sh
bash download_gnomad_background.sh

# Process GTEx eQTLs
python3 process_gtex.py

# Score all three sets and run enrichment analysis
bash score_validation.sh
python3 enrichment_analysis.py \
  --clinvar results/validation/clinvar_scored.tsv \
  --gtex    results/validation/gtex_scored.tsv \
  --gnomad  results/validation/gnomad_background_scored.tsv
```

### Results summary

| Dataset | Variants scored | HIGH tier | HIGH % |
|---------|----------------|-----------|--------|
| ClinVar pathogenic 3′UTR | 44 | 20 | 45.5% |
| GTEx v11 eQTLs | 1,758 | 279 | 15.9% |
| gnomAD v4.1 common SNVs | 969 | 153 | 15.8% |

**ClinVar vs gnomAD enrichment:** OR = 4.44 (95% CI 2.10–9.40), p = 6.55 × 10⁻⁶ (Fisher's exact test)

**GTEx vs gnomAD:** OR = 1.006 (null) — seed disruption is orthogonal to eQTL status

### Tissue-specific results (ClinVar, breast and prostate)

Tissue co-expression weights (GTEx v11) reduce the global HIGH call set from 20 to 5–6 tissue-prioritised variants. **GNAS** dominates both tissues (weight = 1.000), with five ClinVar pathogenic variants disrupting binding sites for miR-138-5p, miR-18a/b-5p, and miR-150-5p, all achieving tissue_score ≥ 0.92.

---

## Model performance

**Training data:** 745,492 examples (in-silico saturation mutagenesis of 278,425 ENCORI sites)  
**Base model:** ESM-2 t33 — `facebook/esm2_t33_650M_UR50D` (650M params)  
**Adapter:** LoRA (r=16, α=32), ~11.5M trainable parameters  

**Test set results (chr8 + chr18 held-out, 38,919 examples):**

| Metric | Value |
|--------|-------|
| AUROC | **0.9938** |
| AUPRC | **0.9980** |
| Sensitivity | 0.9682 |
| Specificity | 0.9524 |
| PPV | 0.9836 |

---

## Scope and limitations

SeedBreaker scores variants within **ENCORI AGO2 CLIP-confirmed miRNA binding sites only** (278,425 sites across 11,140 genes). Variants outside these sites — including those affecting polyadenylation signals, RBP motifs, RNA secondary structure, or computationally predicted (non-CLIP) binding sites — are not scored.

The classifier is trained on in-silico Watson-Crick disruption labels. Experimental validation of individual predictions in functional assays has not been performed.

---

## Citation

> Preprint forthcoming on bioRxiv.

If you use SeedBreaker, please also cite:

- **ENCORI:** Zhou K et al. (2026) *Nature Methods*
- **ESM-2:** Lin Z et al. (2023) *Science* — Evolutionary-scale prediction of atomic-level protein structure with a language model
- **miRBase v22:** Kozomara A et al. (2019) *Nucleic Acids Research*

---

## Project structure

```
SeedBreaker/
├── score.py                     Main scoring engine
├── build_tissue_weights.py      GTEx tissue co-expression weight builder
├── apply_tissue_weights.py      Apply tissue weights to scored output
├── build_training_data.py       Training data generation
├── train_lora_v2.py             LoRA fine-tuning script
├── evaluate_test.py             Held-out test set evaluation
├── process_gtex.py              GTEx v11 eQTL processing
├── enrichment_analysis.py       Fisher's exact + Mann-Whitney enrichment tests
├── make_test_vcf.py             Synthetic test VCF generator
├── download_validation_data.sh  Phase 7 validation data download
├── download_gnomad_background.sh gnomAD v4.1 streaming
├── score_validation.sh          Validation scoring pipeline
└── seedbreaker_rag.md           Full project context document
```

---

## License

MIT License — see LICENSE file.

---

*Developed at the Oxford Cancer Epidemiology Unit, University of Oxford.*
