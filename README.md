# Network Alignment

A graph-based pipeline for aligning two geospatial networks (wastewater or road) by finding node-to-node correspondences using spatial and structural information.

---

## Methods

| Method | Description |
|--------|-------------|
| **CENA** | Main method — structural similarity + compound graph + random walks + Word2Vec embeddings + Hungarian matching |
| **SANA** | Simulated Annealing Network Alignment (baseline) |
| **DeepWalk** | Classic random walk graph embedding (baseline) |
| **Struc2Vec** | Structure-aware graph embedding (baseline, separate environment) |

---

## Project Structure

```
Network_Alignment/
├── config/
│   └── default.yaml          # All parameters (mode, datasets, thresholds...)
├── data/
│   ├── raw/                  # Input GeoJSON files
│   │   ├── Dataset1/
│   │   ├── Dataset5/
│   │   ├── IGN/
│   │   ├── OSM/
│   │   └── Prades/
│   ├── processed/            # Intermediate outputs (pickles)
│   └── results/              # Final alignment outputs
├── scripts/                  # Pipeline scripts (01 → 09)
├── src/                      # Source modules
│   ├── data/loaders.py
│   ├── graph/
│   ├── matching/
│   └── embeddings/
├── outputs/                  # Evaluation results, figures
├── requirements_SANA_CENA_DeepWalk.txt
├── requirements_Struc2vec.txt
└── venv_struc2vec/           # Dedicated venv for Struc2Vec
```

---

## Datasets

The datasets used in this project are publicly available on Figshare:

**[Download datasets](https://doi.org/10.6084/m9.figshare.31628335)**

After downloading, place the dataset folders inside `data/raw/` following this structure:

```
data/raw/
├── Dataset1/
├── Dataset5/
├── IGN/
├── OSM/
└── Prades/
```

---

## Installation

### For CENA / SANA / DeepWalk

```bash
python3.11 -m venv .venv 
source .venv/bin/activate   
pip install -r requirements_SANA_CENA_DeepWalk.txt
```

### For Struc2Vec (separate environment)

```bash
python -m venv .venv_struc2vec
source .venv_struc2vec/bin/activate   
pip install -r requirements_Struc2vec.txt
```

---

## Configuration

All parameters are set in `config/default.yaml`:

```yaml
mode: "wastewater"       # "wastewater" or "road"
zone: "Prades"           # "full" or "Prades"

dataset_pairs:
  - ["Dataset1", "Dataset5"]

spatial:
  radius: 20             # meters — spatial candidate search radius

structural:
  K: 3                   # k-hop neighborhood depth
  alpha: 1.0             # decay parameter

compound_graph:
  method: "CENA"         # "CENA" or "SANA"

alignment:
  alpha: 1               # embedding weight
  beta: 0                # spatial weight (should equal 1 - alpha)
```

---

## Usage

### Run Full Pipeline (CENA or SANA)

From the project root:

```bash
python scripts/main_SANA_CENA.py
```

The method used is determined by `compound_graph.method` in `config/default.yaml`.

### Run DeepWalk Baseline

```bash
python scripts/main_deepwalk.py
```

### Run Struc2Vec Baseline

```bash
source venv_struc2vec/bin/activate
python scripts/main_struc2vec.py
```

### Run Steps Individually

Each step can also be run standalone from the project root:

```bash
python scripts/01_prepare_data.py
python scripts/02_spatial_candidates.py
python scripts/03_structural_similarity.py   # CENA only
python scripts/04_build_compound_graph.py
python scripts/05_generate_random_walks.py
python scripts/06_generate_embeddings.py
python scripts/07_embedding_analysis.py
python scripts/08_hungarian_alignment.py
python scripts/09_evaluate_alignment.py
```

> **Note:** Steps must be run from the project root directory (not from `scripts/`), as paths are resolved relative to it.

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_prepare_data.py` | Load GeoJSON datasets and build graphs |
| 2 | `02_spatial_candidates.py` | Find candidate node pairs within spatial radius (KD-tree) |
| 3 | `03_structural_similarity.py` | Compute structural similarity scores *(CENA only)* |
| 4 | `04_build_compound_graph.py` | Build compound graph from candidates |
| 5 | `05_generate_random_walks.py` | Generate random walks on the compound graph |
| 6 | `06_generate_embeddings.py` | Train Word2Vec embeddings on random walks |
| 7 | `07_embedding_analysis.py` | Analyse embedding space |
| 8 | `08_hungarian_alignment.py` | Solve optimal assignment with Hungarian algorithm |
| 9 | `09_evaluate_alignment.py` | Evaluate against ground truth (Precision / Recall / F1) |

---

## Evaluation

Step 9 compares predictions against ground-truth anchors (nodes within `0.1 m` of each other) and reports:

- **Precision** — fraction of predicted matches that are correct
- **Recall** — fraction of true matches that were found
- **F1-score** — harmonic mean of Precision and Recall

Results and error layers are exported to the `outputs/` directory (including GeoPackage files for GIS visualization).
