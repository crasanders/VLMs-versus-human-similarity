# VLMs versus Human Similarity

Code, stimuli, and data for comparing Vision-Language Model (VLM) and human perceptual similarity judgments, multidimensional scaling (MDS) representations, and Generalized Context Model (GCM) categorization fits across natural rock stimuli and schematic faces.

This repository accompanies [`2510.20859v1.pdf`](2510.20859v1.pdf) and additionally includes:
1. **360-Rock VLM Similarity & Vision Embeddings** — GPT-4o pairwise similarity collection, CLIP/timm vision embedding extraction, and rock MDS / scatterplot notebooks.
2. **Human Similarity Subsampling (MDS & GCM)** — Testing how human MDS spaces and GCM categorization fits vary as a function of the amount of human similarity data (10%–100% in 10% increments, 10 replicates each, metric & nonmetric 8D SMACOF).
3. **Schematic Face Stimuli & Analysis (Nosofsky, 1991)** — Parametric generation of the 34 schematic faces from [`nosofsky1991.pdf`](nosofsky1991.pdf), VLM similarity and classification collection (GPT-4o & Jev), 2D MDS, and GCM modeling.

---

## Repository Structure

```text
.
├── 2510.20859v1.pdf                      # Preprint manuscript
├── nosofsky1991.pdf                      # Nosofsky (1991) reference for schematic faces
├── requirements.txt                      # Python dependencies
│
├── # 1. 360-Rock VLM Similarity & Vision Embeddings
├── get_similarities.py                   # GPT-4o rock pairwise similarity collection
├── number_map.json                       # Digit-to-token mapping for logprob expectation
├── rocks_embeddings_generation/          # CLIP & timm vision embedding extraction scripts
│   ├── embed_clip.py
│   ├── embed_vision.py
│   └── README.md
├── make_plots.ipynb                      # Rock similarity scatterplots & prompt comparisons
├── plot_mds.ipynb                        # 8D MDS dimension scatterplots with rock thumbnails
├── encourage_middle_plots.ipynb          # Baseline vs. encourage-middle prompt analysis
├── reliability.ipynb                     # Similarity rating reliability analysis
├── make_mds_correlation_plots.py         # MDS dimension correlation plots
│
├── # 2. Subsampled Human MDS & GCM Analysis
├── subsample_human_mds.py                # 10%–100% subsampled metric & nonmetric 8D MDS
├── plot_subsampled_mds.py                # 2D scatterplots with rock images & dimension corr
├── fit_subsampled_gcm.py                 # Core (P=3) & Weighted 8D (P=10) GCM fits & plots
├── make_jev_plots.py                     # Shared plotting utilities
├── probmatind.dat                        # Human classification proportions (120 igneous rocks)
├── traincoverage.dat                     # Training stimuli for the igneous coverage condition
├── rotated_mds_rocks360.txt              # Published 8D human MDS solution (Nosofsky et al.)
├── human_360_similarity_matrix.csv       # Full 360x360 human similarity matrix
├── chatgpt_360_similarity_matrix.csv     # Full 360x360 GPT-4o similarity matrix
│
├── # 3. Schematic Face Stimuli & Analysis
├── generate_faces.py                     # Generates the 34 Nosofsky (1991) schematic faces
├── get_face_similarities.py              # GPT-4o pairwise similarity ratings on faces
├── fast_model_face_similarities.py       # Jev pairwise similarity ratings on faces
├── fast_model_face_classification.py     # Jev face categorization judgments
├── fast_model_face_mds.py                # 2D MDS & Procrustes alignment on face ratings
├── fast_model_face_gcm.py                # GCM fits on human & VLM face MDS spaces
│
├── data/
│   ├── 360 Rocks/                        # 360 rock stimulus images (.jpg)
│   ├── Rocks Library/                    # High-res rock images (.png) for MDS plots
│   ├── Rocks-360 Similarity-Judgment Study/  # Individual & collapsed human rating files
│   ├── MDS Solutions/                    # Rotated GPT-4o 8D/10D MDS solutions
│   ├── 360_rocks_table.csv               # Rock metadata & human dimension ratings
│   ├── rocks_360_similarities.csv        # Aggregated human 360-rock similarities
│   ├── 34 Faces/                         # Generated schematic face PNGs
│   ├── nosofsky1991_table_a1.csv         # Face coordinates & human MDS solution
│   └── nosofsky1991_table_1.csv          # Human face categorization probabilities
│
└── output/
    ├── baseline_360/                     # GPT-4o baseline 360-rock ratings (+ raw_responses.jsonl.gz)
    ├── encourage_middle_360/             # GPT-4o encourage-middle ratings (+ raw_responses.jsonl.gz)
    ├── figures/                          # 360-rock & 30-rock scatterplots
    ├── human_mds_subsampling/            # 200 subsampled MDS CSVs, 400 GCM fits & plots
    ├── gpt4o/                            # GPT-4o face ratings (+ raw_responses.jsonl.gz)
    ├── jev/                              # Jev face ratings, classifications & plots
    ├── face_mds/                         # Face MDS & GCM outputs (Jev / human)
    └── face_mds_gpt4o/                   # Face MDS & GCM outputs (GPT-4o / human)
```

---

## Setup

Create a Python 3.10+ virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### API Keys

Scripts that query OpenAI (`get_similarities.py`, `get_face_similarities.py`) read `OPENAI_API_KEY` from the environment (or fall back to a local, git-ignored `openai_key.txt` file):

```bash
export OPENAI_API_KEY="sk-..."   # On Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

> **Note:** `fast_model_face_*.py` scripts use the internal `typesafe_sdk` client for Jev model calls; all precomputed Jev outputs are already included under `output/jev/` and `output/face_mds/` so downstream MDS/GCM analyses run without external API access.

### Compressed Raw Response Logs (`.jsonl.gz`)

To stay within standard GitHub file size limits without Git LFS, the raw OpenAI API response logs (which contain full token logprobs for every rated pair) are stored gzip-compressed:
- `output/baseline_360/raw_responses.jsonl.gz` (64,620 pairs)
- `output/encourage_middle_360/raw_responses.jsonl.gz` (64,620 pairs)
- `output/gpt4o/raw_responses.jsonl.gz` (496 face pairs)

To decompress any file in place:

```bash
gzip -dk output/baseline_360/raw_responses.jsonl.gz
```

Or read directly in Python with `gzip.open(..., "rt", encoding="utf-8")`.

---

## Running the Analyses

### 1. 360-Rock VLM Similarity & Vision Embeddings

```bash
# Collect GPT-4o rock pairwise similarity ratings (requires OPENAI_API_KEY)
python get_similarities.py --help

# Extract CLIP / OpenCLIP and timm vision embeddings
python rocks_embeddings_generation/embed_clip.py --help
python rocks_embeddings_generation/embed_vision.py --help
```

See `rocks_embeddings_generation/README.md` for full embedding extraction options.

### 2. Human Similarity Subsampling (MDS & GCM)

```bash
# Fit 200 rotated 8D MDS solutions (10%–100% in 10% steps x 10 replicates x metric/nonmetric)
python subsample_human_mds.py

# Plot a randomly chosen 100% nonmetric solution with rock thumbnails & dimension correlations
python plot_subsampled_mds.py

# Fit Core (P=3) and Weighted 8D (P=10) GCMs to all 200 solutions + paper benchmarks and generate plots
python fit_subsampled_gcm.py

# Regenerate GCM plots only from saved fit CSVs
python fit_subsampled_gcm.py --replot-only
```

### 3. Schematic Face Stimuli, Similarity, MDS & GCM

```bash
# Regenerate the 34 schematic face stimuli into data/34 Faces/
python generate_faces.py

# Collect GPT-4o face pairwise similarities (requires OPENAI_API_KEY)
python get_face_similarities.py

# Fit 2D MDS & Procrustes rotations on face similarity matrices
python fast_model_face_mds.py

# Fit GCM to human & VLM face MDS spaces
python fast_model_face_gcm.py
```

---

## Components from `2510.20859v1.pdf` Not Yet in This Repository

The following assets referenced in [`2510.20859v1.pdf`](2510.20859v1.pdf) were developed separately and are **not** included in this initial port:

- **Open-weight VLM rating scripts & outputs** (Ollama runs for Llama-4-Scout, Qwen2.5-VL, and Gemma-3).
- **Precomputed `image_embeddings.zip`** (can be regenerated locally using `rocks_embeddings_generation/embed_clip.py` and `rocks_embeddings_generation/embed_vision.py`).
- **Multi-model manuscript figure and table scripts** (Figures 2–6, Tables 1–2 across all VLMs, and the 5-supplementary-dimension GCM fitting scripts). Note that `fit_subsampled_gcm.py` reproduces the Table 2 Core GCM benchmarks for Human MDS, GPT-4o baseline, and GPT-4o encourage-middle directly from `probmatind.dat` and `traincoverage.dat`.
