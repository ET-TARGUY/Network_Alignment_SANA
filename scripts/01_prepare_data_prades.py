"""
Step 1: Load all datasets and prepare graphs.

This script loads all 5 datasets (Dataset1 through Dataset5),
creates NetworkX graphs with coordinates, and saves them to
the processed data directory. It also prepares edgelists and
coordinate files for Struc2vec.
"""
import sys
import os
import pickle
import csv
from pathlib import Path
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loaders import load_graph_from_geojson, print_graph_statistics


# ============================================
# CONFIGURATION
# ============================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
TEMP_DIR = Path("temp")

# Create output directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Dataset names
DATASETS = ["Dataset1", "Dataset2", "Dataset3", "Dataset4", "Dataset5"]

# ============================================
# LOAD ALL DATASETS
# ============================================

print("="*60)
print("STEP 1: PREPARE DATA - LOAD ALL DATASETS")
print("="*60)

graphs = {}

for i, dataset_name in enumerate(DATASETS, 1):
    print(f"\n[{i}/{len(DATASETS)}] Loading {dataset_name}...")
    
    # Paths to GeoJSON files
    nodes_path = RAW_DATA_DIR / dataset_name / "Nodes.geojson"
    pipes_path = RAW_DATA_DIR / dataset_name / "Pipes.geojson"
    
    # Check if files exist
    if not nodes_path.exists():
        print(f"  ⚠️ Nodes file not found: {nodes_path}")
        continue
    if not pipes_path.exists():
        print(f"  ⚠️ Pipes file not found: {pipes_path}")
        continue
    
    # Load graph
    try:
        G = load_graph_from_geojson(str(nodes_path), str(pipes_path))
        graphs[dataset_name] = G
        
        # Print statistics
        print_graph_statistics(G, graph_name=dataset_name)
        
        # Save to processed directory in dataset folder
        dataset_folder = PROCESSED_DATA_DIR / dataset_name
        dataset_folder.mkdir(parents=True, exist_ok=True)
        output_path = dataset_folder / "Graph.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"  ✓ Saved to: {output_path}")
        
    except Exception as e:
        print(f"  ❌ Error loading {dataset_name}: {e}")
        continue

# ============================================
# SUMMARY OF LOADED DATASETS
# ============================================

print("\n" + "="*60)
print("SUMMARY - LOADED DATASETS")
print("="*60)

print(f"\nSuccessfully loaded {len(graphs)}/{len(DATASETS)} datasets:")
for dataset_name, G in graphs.items():
    print(f"  ✓ {dataset_name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

if len(graphs) < len(DATASETS):
    print(f"\n⚠️ Failed to load {len(DATASETS) - len(graphs)} dataset(s)")

# ============================================
# PREPARE EDGELISTS AND COORDINATES FOR STRUC2VEC (FULL GRAPHS)
# ============================================

print("\n" + "="*60)
print("STEP 2: PREPARING FULL GRAPH DATA FOR STRUC2VEC")
print("="*60)

edgelists_saved = []
coords_saved = []

for dataset_name, G in graphs.items():
    print(f"\n  Processing {dataset_name}...")
    
    # Save edgelist (full graph)
    edgelist_path = TEMP_DIR / f"{dataset_name.lower()}.edgelist"
    nx.write_edgelist(G, edgelist_path, data=False)
    print(f"    ✓ Saved edgelist: {edgelist_path}")
    print(f"      {G.number_of_edges():,} edges")
    edgelists_saved.append(edgelist_path)
    
    # Save coordinates as CSV (full graph)
    coords_path = TEMP_DIR / f"{dataset_name.lower()}_coords.csv"
    with open(coords_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'x', 'y'])  # Header
        
        for node in G.nodes():
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
            writer.writerow([node, x, y])
    
    print(f"    ✓ Saved coordinates: {coords_path}")
    print(f"      {G.number_of_nodes():,} nodes")
    coords_saved.append(coords_path)

print(f"\n✓ Saved {len(edgelists_saved)} edgelists to {TEMP_DIR}/")
print(f"✓ Saved {len(coords_saved)} coordinate files to {TEMP_DIR}/")
print(f"\n  Files:")
for edgelist, coords in zip(edgelists_saved, coords_saved):
    print(f"    - {edgelist.name}")
    print(f"    - {coords.name}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("✓ DATA PREPARATION COMPLETE")
print("="*60)
print(f"\nProcessed data saved to:")
print(f"  - Graph pickles: {PROCESSED_DATA_DIR}/")
print(f"  - Struc2vec files (full graphs): {TEMP_DIR}/")
print("="*60)