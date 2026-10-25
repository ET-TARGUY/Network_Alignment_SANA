# SANA: Spatial-Aware Network Alignment

SANA aligns two georeferenced networks (wastewater or road) describing the same
infrastructure. It builds a compound graph whose cross-edges link spatially close
nodes with Gaussian weights, learns node embeddings with random walks and
Skip-Gram, and computes a one-to-one alignment with the Hungarian algorithm.

The repository also contains the baselines (nearest neighbour, CENA, DeepWalk,
struc2vec), the controlled perturbation protocol with provenance ground truth,
and the results files behind every table of the paper.

---

## Repository structure

```
config/        experiment configurations (YAML)
scripts/       pipeline steps, baselines, protocol drivers, analysis
src/           library code (loading, compound graph, walks, embeddings,
               matching, perturbation families); src/struc2vec_lib is third-party
results/       result tables (CSV/TeX) reported in the paper
data/raw/      input datasets (downloaded separately, see below)
data/processed/  generated intermediate files
```

## Data

The datasets are available on Figshare:
**https://doi.org/10.6084/m9.figshare.31628335**

Place each dataset folder in `data/raw/`, each containing `Nodes.geojson`
and `Pipes.geojson`:

```
data/raw/
├── Dataset1/ … Dataset5/    wastewater networks (Montpellier)
├── IGN/                     road network (IGN)
└── OSM/                     road network (OpenStreetMap)
```

## Installation

Two environments are used, because struc2vec needs older `networkx`/`gensim`
versions than the rest of the code.

```bash
# Main environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# struc2vec environment (Python 3.9), used only to train struc2vec
python3.9 -m venv venv_struc2vec
venv_struc2vec/bin/pip install -r requirements_struc2vec.txt
```

All commands below are run from the repository root with the main
environment active. Scripts read `config/default.yaml` unless the
`SANA_CONFIG` environment variable points to another configuration.

---

## Reproducing the paper

### 1. Prepare the networks

```bash
SANA_CONFIG=config/all_pairs.yaml python scripts/01_prepare_data.py    # Dataset1–5
SANA_CONFIG=config/roads_tol1.yaml python scripts/01_prepare_data.py   # IGN, OSM
python scripts/01b_carve_zone.py                                        # zone Z1
```

### 2. Describe the sources (Tables `datasets`, `topology`)

```bash
python scripts/characterise_sources.py   # -> results/characterisation_*
python scripts/topology_metrics.py        # -> results/topology_metrics.csv, topology_table.tex
```

### 3. Identity control (Table `identity`)

```bash
python scripts/01c_perturb.py             # builds data/processed/Z1__identity_L0_s01
python scripts/step0_identity.py          # runs every method, prints the table
python scripts/check_swaps.py data/processed/Z1__identity_L0_s01 SANA_a1   # error analysis
```

### 4. Controlled degradation (Tables `degradation`, `ablation`)

The sweep builds every perturbed copy of Z1 (jitter, warp, densify, composite;
five seeds), runs SANA, CENA and the nearest-neighbour baseline over the
parameter grids, and selects σ and α by two-fold spatial cross-fitting.

```bash
python scripts/run_sweep.py --dry-run     # list the runs
python scripts/run_sweep.py --jobs 4      # -> results/sweep_raw.csv, sweep_selected.csv

# struc2vec: tune until_layer, then DeepWalk and struc2vec on every variant
python scripts/main_struc2vec.py --config config/struc2vec_tuning.yaml
python scripts/main_posthoc.py --config config/posthoc.yaml --method deepwalk  --variants "data/processed/Z1__*"
python scripts/main_posthoc.py --config config/posthoc.yaml --method struc2vec --variants "data/processed/Z1__*"
```

### 5. Real source pairs (Table `pairs`, road pairs)

```bash
# ten wastewater pairs, proximity ground truth
python scripts/step2_real_pairs.py                                   # -> results/real_pairs.csv
python scripts/main_posthoc.py --config config/posthoc_real.yaml --method deepwalk  --base config/all_pairs.yaml
python scripts/main_posthoc.py --config config/posthoc_real.yaml --method struc2vec --base config/all_pairs.yaml

# IGN–OSM road pair, ground truth at 1 m and 2 m
python scripts/step2_real_pairs.py --base config/roads_tol1.yaml --out results/roads_tol1.csv
python scripts/step2_real_pairs.py --base config/roads_tol2.yaml --out results/roads_tol2.csv
```

Each script documents its options at the top of the file (`--dry-run`,
`--only`, `--resume`, `--jobs`, …).

### Single run

To run the full pipeline once on the pair and zone set in a configuration:

```bash
python scripts/main_SANA_CENA.py          # steps 01–09, method from compound_graph.method
python scripts/main_nearest_neighbor.py   # nearest-neighbour baseline
```

---

## Pipeline steps

| Step | Script | Role |
|------|--------|------|
| 01  | `01_prepare_data.py` | GeoJSON → graphs |
| 01b | `01b_carve_zone.py` | carve an evaluation zone with a topological halo |
| 01c | `01c_perturb.py` | build a perturbed copy with its provenance ground truth |
| 02  | `02_spatial_candidates.py` | spatial candidates within the radius |
| 03  | `03_structural_similarity.py` | structural similarity (CENA only) |
| 04  | `04_build_compound_graph.py` | compound graph with weighted cross-edges |
| 05  | `05_generate_random_walks.py` | random walks |
| 06  | `06_generate_embeddings.py` | Skip-Gram embeddings |
| 07  | `07_embedding_analysis.py` | embedding diagnostics |
| 08  | `08_hungarian_alignment.py` | one-to-one assignment |
| 09  | `09_evaluate_alignment.py` | precision, recall = TP/(TP+FN), F1 |

## Results files

| Paper | File(s) in `results/` | Produced by |
|-------|-----------------------|-------------|
| Network characteristics | `characterisation_*.csv/.tex` | `characterise_sources.py` |
| Topological homogeneity | `topology_metrics.csv`, `topology_table.tex` | `topology_metrics.py` |
| Controlled degradation, parameter selection | `sweep_raw.csv`, `sweep_selected.csv` | `run_sweep.py` |
| DeepWalk / struc2vec under the protocol | `posthoc_*`, `struc2vec_tuning_raw.csv` | `main_posthoc.py`, `main_struc2vec.py` |
| Wastewater source pairs | `real_pairs.csv`, `real_*` | `step2_real_pairs.py`, `main_posthoc.py` |
| Road pairs | `roads_tol1.csv`, `roads_tol2.csv` | `step2_real_pairs.py` |

## Third-party code

`src/struc2vec_lib/` is the reference implementation of struc2vec
(https://github.com/leoribeiro/struc2vec), used as a baseline and distributed
under its own license. See `src/struc2vec_lib/README.md`.
