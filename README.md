# SANA: Spatial-Aware Network Alignment

SANA is a Python framework for aligning spatial infrastructure networks (wastewater, road, OSM/IGN). It integrates spatial context into the alignment pipeline to overcome the limitations of purely structural methods, achieving over 95% F1-score on large-scale real-world datasets.

---

## Installation

**For SANA and DeepWalk:**
```bash
pip install -r requirements_SANA_DeepWalk.txt
```

**For Struc2vec:**
```bash
pip install -r requirements_struc2vec.txt
```

---

## Methodology

1. **Data Preparation** – loads GeoJSON networks into NetworkX graphs with coordinates
2. **Spatial Candidates** – uses KD-Tree to find cross-network node candidates within a spatial radius
3. **Structural Similarity** – computes k-hop neighborhood similarity (used by CENA method)
4. **Compound Graph** – merges both networks into a single graph with cross-edges weighted by spatial (Gaussian) or structural (CENA) similarity
5. **Random Walks** – generates node sequences preserving local graph structure
6. **Skip-Gram Embeddings** – learns node representations in a shared vector space via Word2Vec
7. **Hungarian Alignment** – finds optimal one-to-one node matching using combined embedding + spatial similarity
8. **Evaluation** – computes Precision, Recall, and F1-score against ground truth anchors

---

## Pipeline Usage

Run each step in order from the project root:

```bash
# Step 1 – Prepare data (choose dataset type)
python scripts/01_prepare_data_wastewater.py
python scripts/01_prepare_data_road.py
python scripts/01_prepare_data_prades.py

# Step 2 – Generate spatial candidates
python scripts/02_spatial_candidates.py

# Step 3 – Compute structural similarity (for CENA method)
python scripts/03_structural_similarity.py

# Step 4 – Build compound graph
python scripts/04_build_compound_graph.py

# Step 5 – Generate random walks
python scripts/05_generate_random_walks.py

# Step 6 – Train embeddings
python scripts/06_generate_embeddings.py

# Step 7 – Analyze embeddings (optional)
python scripts/07_embedding_analysis.py

# Step 8 – Run Hungarian alignment
python scripts/08_hungarian_alignment.py

# Step 9 – Evaluate alignment
python scripts/09_evaluate_alignment.py
```

---

## Configuration

All pipeline parameters are controlled via `config/default.yaml`:

```yaml
zone: "full"               # "full" or "Prades"
dataset_pairs:
  - ["Dataset1", "Dataset5"]

spatial:
  radius: 20               # Spatial candidate search radius (meters)

compound_graph:
  method: "gaussian"       # "gaussian" (spatial) or "cena" (structural)
  gaussian:
    sigma: 5               # Gaussian bandwidth (meters)

random_walks:
  num_walks: 10
  walk_length: 80

embeddings:
  vector_size: 128
  window: 5
  epochs: 5
  sg: 1                    # 1 = Skip-gram, 0 = CBOW

alignment:
  alpha: 1                 # Embedding weight (beta = 1 - alpha = spatial weight)
  sigma: 5
```

Baseline-specific configs: `config/deepwalk.yaml`, `config/struc2vec.yaml`

---

## Baselines

Two baseline methods are provided for comparison:

```bash
# DeepWalk – independent embeddings per network, post-hoc alignment
python scripts/deepwalk_baseline.py

# Struc2vec – pure structural identity, no spatial filtering
python scripts/struc2vec_baseline.py
```

---

## Ablation Study

Study the impact of the spatial radius parameter on alignment performance:

```bash
python scripts/ablation_radius.py
```

---

## Input Format

- **Nodes**: GeoJSON file with point geometries
- **Pipes/Edges**: GeoJSON file with connectivity information
- Expected structure per dataset:
  ```
  data/raw/DatasetX/
  ├── Nodes.geojson
  └── Pipes.geojson
  ```

## Output

- Aligned anchor pairs with confidence scores (saved as `.pkl` and `.gpkg`)
- Evaluation metrics: Precision, Recall, F1-score
- Error analysis layers exportable to GeoPackage for GIS visualization

---

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| Dataset1 – Dataset5 | Wastewater | Montpellier wastewater networks (tens of thousands of nodes) |
| Prades | Wastewater | Sub-zone of Montpellier |
| IGN vs OSM | Road | French national geographic institute vs OpenStreetMap |

---

## Results

| Dataset | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Montpellier WW (large) | >95% | >95% | >95% |
| Road (IGN vs OSM) | TBD | TBD | TBD |

---

## Project Structure

```
Network_Alignment/
├── config/
│   ├── default.yaml          # Main pipeline configuration
│   ├── deepwalk.yaml         # DeepWalk baseline config
│   └── struc2vec.yaml        # Struc2vec baseline config
├── data/
│   ├── raw/                  # Input GeoJSON files (DatasetX/Nodes.geojson, Pipes.geojson)
│   └── processed/            # Pickled graphs and intermediate results
├── outputs/                  # Alignment results and GeoPackage exports
├── scripts/
│   ├── 01_prepare_data_wastewater.py
│   ├── 01_prepare_data_road.py
│   ├── 01_prepare_data_prades.py
│   ├── 02_spatial_candidates.py
│   ├── 03_structural_similarity.py
│   ├── 04_build_compound_graph.py
│   ├── 05_generate_random_walks.py
│   ├── 06_generate_embeddings.py
│   ├── 07_embedding_analysis.py
│   ├── 08_hungarian_alignment.py
│   ├── 09_evaluate_alignment.py
│   ├── deepwalk_baseline.py
│   ├── struc2vec_baseline.py
│   └── ablation_radius.py
├── src/
│   ├── data/
│   │   ├── loaders.py        # GeoJSON → NetworkX graph loading
│   │   └── preprocessors.py
│   ├── graph/
│   │   ├── compound.py       # Compound graph construction (Gaussian & CENA)
│   │   └── random_walks.py   # Random walk generation
│   ├── embeddings/
│   │   └── word2vec_embedder.py  # Skip-Gram embedding training
│   ├── matching/
│   │   ├── spatial.py        # Spatial candidate filtering
│   │   └── structural.py     # Structural similarity (k-hop)
│   └── struc2vec_lib/        # Struc2vec library
├── requirements_SANA_DeepWalk.txt
├── requirements_struc2vec.txt
└── README.md
```

---

## Citation

If you use SANA in your research, please cite:

```bibtex
@phdthesis{sana2025,
  author    = {Omar},
  title     = {Possibilistic Conditioning and Graph-Based Representation of Wastewater Networks},
  school    = {Université d'Artois / Sidi Mohamed Ben Abdellah University},
  year      = {2025}
}
```

---

## Contact

For questions or collaborations, feel free to reach out via your institutional email (IUSTI, CNRS – Aix-Marseille Université).
