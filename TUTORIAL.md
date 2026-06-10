# SeedBreaker — Setup and Usage Tutorial

This tutorial walks you through setting up SeedBreaker on a new machine and scoring your first variants. It takes roughly 20–30 minutes depending on your internet speed.

---

## What you need before you start

- macOS (Apple Silicon or Intel) or Linux
- At least 8 GB RAM (16 GB recommended)
- At least 10 GB free disk space
- Internet connection
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://github.com/conda-forge/miniforge) installed

You do **not** need a GPU. A Mac M-series chip or a standard Linux machine works fine for scoring small-to-medium variant sets.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/Cancer-epi-unit/SeedBreaker.git
cd SeedBreaker
```

---

## Step 2 — Create the conda environment

```bash
conda create -n seedbreaker python=3.11 -y
conda activate seedbreaker
```

Install bioinformatics tools:

```bash
conda install -c bioconda -c conda-forge \
    bedtools samtools htslib bcftools \
    -y
```

Install Python packages:

```bash
pip install torch torchvision torchaudio
pip install transformers>=5.0.0 peft==0.19.1 accelerate
pip install pandas numpy scipy pyarrow huggingface_hub
```

> **Optional:** Install ViennaRNA if you want Δmfe scores (not required for standard scoring):
> ```bash
> conda install -c bioconda viennarna -y
> ```

---

## Step 3 — Download the reference data

The reference data package contains the ENCORI binding site atlas, miRBase seed sequences, and pre-computed tissue weights. Download it from Zenodo:

```bash
# Create the Pre_data directory structure
mkdir -p Pre_data/ENCORI Pre_data/miRBase Pre_data/tissue

# Download and unzip the reference package
# Replace the URL below with the actual Zenodo DOI link once published
wget https://zenodo.org/record/XXXXXXX/files/seedbreaker_reference_data_v1.zip
unzip seedbreaker_reference_data_v1.zip -d .
rm seedbreaker_reference_data_v1.zip
```

After unzipping your `Pre_data/` folder should look like this:

```
Pre_data/
├── ENCORI/
│   ├── mirna_sites_encori_annotated_sorted.bed.gz
│   └── mirna_sites_encori_annotated_sorted.bed.gz.tbi
├── miRBase/
│   └── mirna_seeds_hsa.json
└── tissue/
    └── tissue_weights.tsv.gz
```

---

## Step 4 — Download the reference genome

SeedBreaker needs the GRCh38 primary assembly to extract sequence context around each variant. This is a ~900 MB download:

```bash
mkdir -p Pre_data/Genome
cd Pre_data/Genome

wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.primary_assembly.genome.fa.gz

# Decompress and index
gunzip GRCh38.primary_assembly.genome.fa.gz
samtools faidx GRCh38.primary_assembly.genome.fa

cd ../..
```

> **Already have hg38?** Just symlink or copy your existing FASTA and FAI here — any standard GRCh38 primary assembly works.

---

## Step 5 — Download the LoRA model adapter

The pre-trained SeedBreaker LoRA adapter (~45 MB) is hosted on HuggingFace. The base ESM-2 model (~2.5 GB) downloads automatically the first time you run inference.

```bash
mkdir -p models/seedbreaker_lora_v2

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='c3114203/seedbreaker-lora-v2',
    local_dir='models/seedbreaker_lora_v2/best'
)
"
```

> The ESM-2 base model will download automatically on first run and cache in `~/.cache/huggingface/`. This takes a few minutes the first time only.

---

## Step 6 — Verify the setup

Check that everything is in place:

```bash
conda activate seedbreaker

# Check bioinformatics tools
bedtools --version
samtools --version
bcftools --version

# Check Python packages
python3 -c "import torch, transformers, peft; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('peft:', peft.__version__)"

# Check reference files
ls -lh Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz
ls -lh Pre_data/miRBase/mirna_seeds_hsa.json
ls -lh Pre_data/Genome/GRCh38.primary_assembly.genome.fa.fai
ls -lh models/seedbreaker_lora_v2/best/
```

---

## Step 7 — Run a test with the synthetic VCF

SeedBreaker includes a script to generate a small synthetic test VCF so you can confirm everything works before running on real data:

```bash
conda activate seedbreaker

# Generate a synthetic test VCF (10 variants in known ENCORI sites)
python3 make_test_vcf.py --out test_data/test_variants.vcf.gz

# Score it
python3 score.py \
  --vcf    test_data/test_variants.vcf.gz \
  --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \
  --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \
  --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \
  --model  models/seedbreaker_lora_v2/best \
  --out    results/test_scored.tsv \
  --no-mfe
```

Expected output — you should see a TSV with columns including `combined_score` and `tier`:

```
variant_id   chrom  pos  ref  alt  site_id                              tier
...          chr1   ...  A    T    hsa-miR-205-5p:SAMD11:MIMAT0000266  HIGH
```

---

## Step 8 — Score your own VCF

Your input VCF must be:
- bgzipped and tabix-indexed
- hg38 coordinates
- `chr` prefix on chromosomes (chr1, chr2 ... chrX)

```bash
# Index your VCF if not already done
bgzip variants.vcf
tabix -p vcf variants.vcf.gz

# Score
python3 score.py \
  --vcf    variants.vcf.gz \
  --sites  Pre_data/ENCORI/mirna_sites_encori_annotated_sorted.bed.gz \
  --genome Pre_data/Genome/GRCh38.primary_assembly.genome.fa \
  --seeds  Pre_data/miRBase/mirna_seeds_hsa.json \
  --model  models/seedbreaker_lora_v2/best \
  --out    results/scored_variants.tsv
```

Filter to HIGH tier only:

```bash
awk -F'\t' 'NR==1 || $17=="HIGH"' results/scored_variants.tsv > results/high_tier.tsv
```

---

## Step 9 — Apply tissue-specific weights (optional)

If you want to prioritise variants by tissue co-expression:

```bash
# See available tissues
python3 apply_tissue_weights.py --list-tissues

# Apply breast tissue weights
python3 apply_tissue_weights.py \
  --scores results/scored_variants.tsv \
  --tissue breast \
  --out    results/scored_breast.tsv

# Filter to HIGH tissue tier only
python3 apply_tissue_weights.py \
  --scores results/scored_variants.tsv \
  --tissue prostate \
  --out    results/scored_prostate.tsv \
  --high-only
```

---

## Output columns

| Column | Description |
|--------|-------------|
| `variant_id` | VCF ID field (or chrom:pos:ref:alt) |
| `chrom`, `pos`, `ref`, `alt` | Variant coordinates |
| `af` | Allele frequency from VCF (if present) |
| `site_id` | ENCORI binding site ID (miRNA:gene:MIMAT) |
| `mirna`, `gene` | miRNA family and target gene |
| `strand`, `site_type` | Binding site strand and type (7mer-m8 etc.) |
| `seed_pos` | Position within seed region (2–8) |
| `wc_disrupted` | True if Watson-Crick pair is broken |
| `delta_mfe` | ViennaRNA Δmfe (NA if --no-mfe) |
| `model_score` | ESM-2 + LoRA disruption probability |
| `combined_score` | 0.4 × physics + 0.6 × model |
| `tier` | HIGH / MEDIUM / LOW |

For tissue-weighted output, three additional columns are appended: `coexp_weight`, `tissue_score`, `tissue_tier`.

---

## Troubleshooting

**"ENCORI sites not found"** — check your VCF has `chr` prefix. Run `bcftools view -h variants.vcf.gz | head` and confirm chromosome names start with `chr`.

**"model not found"** — confirm `models/seedbreaker_lora_v2/best/` contains `adapter_config.json` and `adapter_model.safetensors`. Re-run Step 5 if missing.

**Slow inference on CPU** — normal. On a Mac M-series PyTorch uses MPS automatically, which is ~5–10× faster. On CPU, expect ~1–2 seconds per variant.

**Memory error** — ESM-2 needs ~5 GB RAM. Close other applications or increase swap space.

**"chromosome not in genome"** — your genome FASTA may use `1, 2, 3` instead of `chr1, chr2, chr3`. Run:
```bash
# Add chr prefix to genome
sed 's/^>/>chr/' GRCh38.primary_assembly.genome.fa > GRCh38.primary_assembly.genome.chr.fa
samtools faidx GRCh38.primary_assembly.genome.chr.fa
```

---

## Questions

Open an issue at https://github.com/Cancer-epi-unit/SeedBreaker/issues
