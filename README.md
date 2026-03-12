# Network Alignment

A graph-based pipeline for aligning two geospatial networks (wastewater or road) by finding node-to-node correspondences using spatial and structural information.

---

## Methods

| Method | Description |
|--------|-------------|
| **SANA** | Main method — Simulated Annealing Network Alignment, compound graph + random walks + Word2Vec embeddings + Hungarian matching |
| **CENA** | Cross-network Embedding Network Alignment (baseline) — structural similarity + compound graph + random walks + Word2Vec embeddings |
| **DeepWalk** | Classic random walk graph embedding (baseline) |
| **Struc2Vec** | Structure-aware graph embedding (baseline, separate environment) |

---

## Project Structure

```
Network_Alignment/
├── config/
│   ├── default.yaml          # Main config (CENA / SANA)
│   ├── deepwalk.yaml         # Config for DeepWalk baseline
│   └── struc2vec.yaml        # Config for Struc2Vec baseline
├── data/
│   ├── raw/                  # Input GeoJSON files (downloaded from Figshare)
│   │   ├── Dataset1/
│   │   ├── Dataset5/
│   │   ├── IGN/
│   │   ├── OSM/
│   │   └── Prades/
│   ├── processed/            # Intermediate outputs (pickles, edgelists, coords)
│   └── results/              # (reserved for future use)
├── scripts/                  # Pipeline scripts (01 → 09) + mains
├── src/                      # Source modules
│   ├── data/loaders.py
│   ├── graph/
│   ├── matching/
│   └── embeddings/
├── outputs/                  # Evaluation plots and GeoPackage alignment files
├── requirements_SANA_CENA_DeepWalk.txt
├── requirements_Struc2vec.txt
└── venv_struc2vec/           # Dedicated venv for Struc2Vec
```

---

## Datasets

The datasets used in this project are publicly available on Figshare:

**[Download datasets](https://doi.org/10.6084/m9.figshare.31628335)**

### What is in the download?

The archive contains geospatial networks representing wastewater and road infrastructure, organized by source/provider. Each dataset folder contains two GeoJSON files:

- **`Nodes.geojson`** — network nodes (manholes, junctions, etc.) with point geometries and attributes
- **`Pipes.geojson`** — network edges (pipes, road segments) with linestring geometries and attributes

| Folder | Description |
|--------|-------------|
| `Dataset1/` | Wastewater network — source 1 (full zone) |
| `Dataset5/` | Wastewater network — source 5 (full zone) |
| `IGN/` | Road network from IGN (Institut national de l'information géographique) |
| `OSM/` | Road network from OpenStreetMap |
| `Prades/Dataset1/` | Subset of Dataset1 cropped to the Prades area |
| `Prades/Dataset5/` | Subset of Dataset5 cropped to the Prades area |

The goal is to align nodes from one network to their counterparts in the other (e.g., `Dataset1` ↔ `Dataset5`, or `IGN` ↔ `OSM`).

### Where to place the files

After downloading, extract the contents and place the dataset folders inside `data/raw/`:

```
data/raw/
├── Dataset1/
│   ├── Nodes.geojson
│   └── Pipes.geojson
├── Dataset5/
│   ├── Nodes.geojson
│   └── Pipes.geojson
├── IGN/
│   ├── Nodes.geojson
│   └── Pipes.geojson
├── OSM/
│   ├── Nodes.geojson
│   └── Pipes.geojson
└── Prades/
    ├── Dataset1/
    │   ├── Nodes.geojson
    │   └── Pipes.geojson
    └── Dataset5/
        ├── Nodes.geojson
        └── Pipes.geojson
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
python3.11 -m venv .venv_struc2vec
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

random_walks:
  num_walks: 10          # number of walks per node
  walk_length: 80        # steps per walk
  cena:
    q: 0.5               # network-switching probability

embeddings:
  vector_size: 128       # Word2Vec embedding dimension
  window: 5
  epochs: 5

alignment:
  alpha: 1               # embedding weight
  beta: 0                # spatial weight (should equal 1 - alpha)
```

DeepWalk and Struc2Vec have their own config files (`config/deepwalk.yaml` and `config/struc2vec.yaml`) with similar structure.

---

## Usage

### Run Full Pipeline (CENA or SANA)

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

> **Note:** All scripts must be run from the project root directory (not from `scripts/`), as paths are resolved relative to it.

---

## Pipeline Steps

### Step 1 — Prepare Data (`01_prepare_data.py`)

Loads `Nodes.geojson` and `Pipes.geojson` for each dataset and builds a NetworkX graph.

**Outputs** (saved to `data/processed/<dataset>/`):
- `Graph.pkl` — serialized NetworkX graph with node attributes and edge geometries
- `<dataset>.edgelist` — plain text edge list (for DeepWalk / Struc2Vec)
- `<dataset>_coords.csv` — node coordinates (longitude, latitude)

---

### Step 2 — Spatial Candidates (`02_spatial_candidates.py`)

Finds candidate node pairs between two networks whose spatial distance is within the configured `radius` (default: 20 m), using a KD-tree index.

**Outputs** (saved to `data/processed/<dataset1>/`):
- `spatial_candidates_to_<dataset2>_<radius>m.pkl` — dict mapping each node in Dataset1 to its candidate nodes in Dataset2, with distances

---

### Step 3 — Structural Similarity (`03_structural_similarity.py`) *(CENA only)*

For each candidate pair, computes a structural similarity score based on the k-hop neighborhood topology (depth `K`, decay `alpha`).

**Outputs** (saved to `data/processed/<dataset1>/`):
- `structural_similarities_to_<dataset2>_<radius>m_K<K>.pkl` — dict of structural similarity scores per candidate pair

---

### Step 4 — Build Compound Graph (`04_build_compound_graph.py`)

Merges the two input graphs and the candidate edges into a single compound graph. Cross-edges are weighted by spatial distance (SANA) or structural similarity (CENA).

**Outputs** (saved to `data/processed/<dataset1>/`):
- `compound_graph_cena_<dataset2>_<radius>m_K<K>.pkl` — compound graph (CENA)
- `compound_graph_gaussian_<dataset2>_<radius>m_sigma<sigma>.pkl` — compound graph (SANA)
- `compound_graph_cross_edges_<dataset2>.gpkg` — cross-edges as a GeoPackage (for GIS visualization)

---

### Step 5 — Generate Random Walks (`05_generate_random_walks.py`)

Runs biased random walks on the compound graph. In CENA mode, walks can switch between the two networks with probability `q`; higher-weight cross-edges are favored.

**Outputs** (saved to `data/processed/<dataset1>/`):
- `random_walks_<method>_<dataset2>.pkl` — list of node-sequence walks

**Outputs** (saved to `outputs/`):
- `random_walks_<method>_<dataset1>_<dataset2>.png` — walk statistics plot

---

### Step 6 — Generate Embeddings (`06_generate_embeddings.py`)

Trains a Word2Vec Skip-gram model on the random walks to produce a vector embedding for each node in the compound graph.

**Outputs** (saved to `data/processed/<dataset1>/`):
- `embeddings_<method>_<dataset2>.pkl` — node embedding vectors (numpy arrays)
- `word2vec_<method>_<dataset2>.model` — saved Word2Vec model

**Outputs** (saved to `outputs/`):
- `training_loss_<method>_<dataset1>_<dataset2>.png` — training loss curve

---

### Step 7 — Embedding Analysis (`07_embedding_analysis.py`)

Analyses the quality of the learned embeddings by computing pairwise cosine similarities between candidate pairs and producing diagnostic plots.

**Outputs** (saved to `outputs/`):
- `plot1_similarity_dist_*.png` — distribution of cosine similarities for true vs. false pairs
- `plot2_distance_vs_similarity_*.png` — spatial distance vs. embedding similarity scatter
- `plot3_discrimination_*.png` — discrimination curve (true vs. false pair separability)
- `plot4_rank_distribution_*.png` — rank of the correct match among candidates
- `plot5_topk_accuracy_*.png` — top-k retrieval accuracy
- `plot6_embedding_norms_*.png` — distribution of embedding vector norms
- `plot7_similarity_by_distance_*.png` — similarity binned by spatial distance
- `plot8_nearest_candidate_sim_*.png` — similarity of nearest spatial candidate
- `plot9_topk_summary_*.png` — top-k summary across all pairs

---

### Step 8 — Hungarian Alignment (`08_hungarian_alignment.py`)

Solves the optimal one-to-one node assignment using the Hungarian algorithm. The cost matrix is built from embedding similarity (weight `alpha`) and optional spatial proximity (weight `beta`).

**Outputs** (saved to `outputs/`):
- `alignment_hungarian_<method>_<dataset1>_<dataset2>.gpkg` — final node-pair assignments as a GeoPackage with match geometries (lines connecting matched nodes), viewable in QGIS or any GIS tool

---

### Step 9 — Evaluate Alignment (`09_evaluate_alignment.py`)

Compares the predicted alignment against ground-truth anchors (node pairs within 0.1 m of each other) and reports performance metrics.

**Outputs** (saved to `outputs/`):
- `alignment_evaluation_<method>_<dataset1>_<dataset2>.png` — evaluation summary plot (Precision, Recall, F1)
- Console output with Precision / Recall / F1 scores and error breakdown

---

## Evaluation

Step 9 compares predictions against ground-truth anchors (nodes within `0.1 m` of each other) and reports:

- **Precision** — fraction of predicted matches that are correct
- **Recall** — fraction of true matches that were found
- **F1-score** — harmonic mean of Precision and Recall

Results and error layers are also exported to `outputs/` as GeoPackage files (`.gpkg`) for visual inspection in QGIS or similar GIS tools.

---

## Outputs Summary

| Location | Content |
|----------|---------|
| `data/processed/` | Intermediate pickles (graphs, candidates, embeddings, walks) |
| `outputs/*.gpkg` | Final alignment and error layers — open in QGIS |
| `outputs/*.png` | Diagnostic and evaluation plots |