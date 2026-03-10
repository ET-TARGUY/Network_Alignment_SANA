"""
Step 1: Load all datasets and prepare graphs.

Supports two modes (set via params.yaml):
  - "wastewater": loads dataset pairs from GeoJSON
  - "road":       loads OSM and IGN datasets from GeoJSON
"""

import sys
import os
import pickle
from pathlib import Path

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loaders import load_graph_from_geojson, print_graph_statistics


# ============================================
# LOAD PARAMS
# ============================================

with open("config/default.yaml", "r") as f:
    params = yaml.safe_load(f)

MODE               = params["mode"]                      # "wastewater" or "road"
ZONE               = params.get("zone", "full")          # "full" or "Prades"
RAW_DATA_DIR       = Path(params["paths"]["data_raw"])
PROCESSED_DATA_DIR = Path(params["paths"]["data_processed"])
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

if MODE == "wastewater" and ZONE == "Prades":
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "Prades"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
# Adjust raw data dir based on zone
if MODE == "wastewater" and ZONE == "Prades":
    RAW_DATA_DIR = RAW_DATA_DIR / "Prades"

# Collect unique dataset names from dataset_pairs (both modes)
DATASETS = list(dict.fromkeys(
    ds for pair in params["dataset_pairs"] for ds in pair
))



# ============================================
# MAIN
# ============================================

print("=" * 60)
print(f"STEP 1: PREPARE DATA — MODE: {MODE.upper()} | ZONE: {ZONE.upper()}")
print("=" * 60)

# Load all datasets
graphs = {}

for i, dataset_name in enumerate(DATASETS, 1):
    print(f"\n[{i}/{len(DATASETS)}] Loading {dataset_name}...")

    nodes_path = RAW_DATA_DIR / dataset_name / "Nodes.geojson"
    pipes_path = RAW_DATA_DIR / dataset_name / "Pipes.geojson"

    if not nodes_path.exists():
        print(f"  ⚠️  Nodes file not found: {nodes_path}")
        continue
    if not pipes_path.exists():
        print(f"  ⚠️  Pipes file not found: {pipes_path}")
        continue

    try:
        data_format = "road" if MODE == "road" else None
        G = load_graph_from_geojson(str(nodes_path), str(pipes_path),
                                    **({"data_format": data_format} if data_format else {}))
        graphs[dataset_name] = G
        print_graph_statistics(G, graph_name=dataset_name)

        dataset_folder = PROCESSED_DATA_DIR / dataset_name
        dataset_folder.mkdir(parents=True, exist_ok=True)
        output_path = dataset_folder / "Graph.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
        print(f"  ✓ Saved to: {output_path}")

    except Exception as e:
        print(f"  ❌ Error loading {dataset_name}: {e}")
        continue

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nSuccessfully loaded {len(graphs)}/{len(DATASETS)} datasets:")
for dataset_name, G in graphs.items():
    print(f"  ✓ {dataset_name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

if len(graphs) < len(DATASETS):
    print(f"\n⚠️  Failed to load {len(DATASETS) - len(graphs)} dataset(s)")

print("\n" + "=" * 60)
print("✓ DATA PREPARATION COMPLETE")
print("=" * 60)
