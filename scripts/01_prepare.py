"""
Step 1: Load all datasets and prepare graphs.

Supports two modes (set via params.yaml):
  - "wastewater": loads dataset pairs from GeoJSON
  - "road":       converts IGN shapefile to GeoJSON first, then loads OSM/IGN
"""

import sys
import os
import json
import pickle
from pathlib import Path

import yaml
import geopandas as gpd

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

# Adjust raw data dir based on zone
if ZONE == "Prades":
    RAW_DATA_DIR = RAW_DATA_DIR / "Prades"

# Collect unique dataset names from dataset_pairs (both modes)
DATASETS = list(dict.fromkeys(
    ds for pair in params["dataset_pairs"] for ds in pair
))


# ============================================
# ROAD MODE: CONVERT IGN SHAPEFILE TO GEOJSON
# ============================================

def shapefile_to_geojson(shp_path, output_dir, precision=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(shp_path)
    print(f"  Loaded {len(gdf):,} edges from shapefile")

    coord_to_id = {}
    nodes = []
    node_counter = 0

    def get_or_create_node(x, y):
        nonlocal node_counter
        key = (round(x, precision), round(y, precision))
        if key not in coord_to_id:
            coord_to_id[key] = node_counter
            nodes.append({"nodeID": node_counter, "x": key[0], "y": key[1]})
            node_counter += 1
        return coord_to_id[key]

    edges = []
    for idx, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        u = get_or_create_node(coords[0][0], coords[0][1])
        v = get_or_create_node(coords[-1][0], coords[-1][1])
        if u != v:
            edges.append({
                "nodeID": idx, "u": u, "v": v,
                "ID": row["ID"], "NATURE": row["NATURE"],
                "IMPORTANCE": row["IMPORTANCE"],
                "SENS": row["SENS"],
                "VIT_MOY_VL": row["VIT_MOY_VL"],
                "NB_VOIES": row["NB_VOIES"],
                "LARGEUR": row["LARGEUR"],
                "length": row.geometry.length
            })

    nodes_geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
        "features": [
            {"type": "Feature", "properties": n,
             "geometry": {"type": "Point", "coordinates": [n["x"], n["y"]]}}
            for n in nodes
        ]
    }
    pipes_geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
        "features": [
            {"type": "Feature", "properties": e, "geometry": None}
            for e in edges
        ]
    }

    with open(output_dir / "Nodes.geojson", "w") as f:
        json.dump(nodes_geojson, f)
    with open(output_dir / "Pipes.geojson", "w") as f:
        json.dump(pipes_geojson, f)

    print(f"  ✓ Nodes: {len(nodes):,} | Edges: {len(edges):,}")


# ============================================
# MAIN
# ============================================

print("=" * 60)
print(f"STEP 1: PREPARE DATA — MODE: {MODE.upper()} | ZONE: {ZONE.upper()}")
print("=" * 60)

# Road mode: convert IGN shapefile first
if MODE == "road":
    ign_shp = RAW_DATA_DIR / "IGN" / "TRONCON_DE_ROUTE.shp"
    if ign_shp.exists():
        print("\nConverting IGN shapefile to GeoJSON...")
        shapefile_to_geojson(ign_shp, RAW_DATA_DIR / "IGN")
    else:
        print(f"⚠️  IGN shapefile not found: {ign_shp}")

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
