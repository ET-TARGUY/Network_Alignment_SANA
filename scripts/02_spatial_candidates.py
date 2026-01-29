"""
Step 2: Find spatial candidates for dataset pairs.

This script:
1. Loads configuration to get dataset pairs
2. Loads graphs for each pair
3. Finds spatial candidates within radius
4. Saves candidates for each pair
"""
import sys
import os
import pickle
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loaders import get_coordinates
from src.matching.spatial import find_spatial_candidates, print_candidate_statistics


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 2: FIND SPATIAL CANDIDATES")
print("="*60)

# Load config
config_path = Path("config/default.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get parameters
dataset_pairs = config['dataset_pairs']
zone = config['zone']
radius = config['spatial']['radius']

PROCESSED_DATA_DIR = Path("data/processed")

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Radius: {radius}m")
print(f"  Dataset pairs: {dataset_pairs}")

# Determine base directory
if zone == "Prades":
    base_dir = PROCESSED_DATA_DIR / "Prades"
else:
    base_dir = PROCESSED_DATA_DIR

# ============================================
# PROCESS EACH DATASET PAIR
# ============================================

for pair_idx, (dataset1_name, dataset2_name) in enumerate(dataset_pairs, 1):
    print(f"\n{'='*60}")
    print(f"PAIR {pair_idx}/{len(dataset_pairs)}: {dataset1_name} ↔ {dataset2_name}")
    print(f"{'='*60}")
    
    # Load graphs
    graph1_path = base_dir / dataset1_name / "Graph.pkl"
    graph2_path = base_dir / dataset2_name / "Graph.pkl"
    
    if not graph1_path.exists():
        print(f"  ❌ Graph not found: {graph1_path}")
        continue
    if not graph2_path.exists():
        print(f"  ❌ Graph not found: {graph2_path}")
        continue
    
    print(f"\nLoading graphs...")
    with open(graph1_path, 'rb') as f:
        G1 = pickle.load(f)
    with open(graph2_path, 'rb') as f:
        G2 = pickle.load(f)
    
    print(f"  ✓ {dataset1_name}: {G1.number_of_nodes():,} nodes")
    print(f"  ✓ {dataset2_name}: {G2.number_of_nodes():,} nodes")
    
    # Extract coordinates
    print(f"\nExtracting coordinates...")
    coords_G1 = get_coordinates(G1)
    coords_G2 = get_coordinates(G2)
    
    print(f"  {dataset1_name}: {len(coords_G1):,} nodes with coordinates")
    print(f"  {dataset2_name}: {len(coords_G2):,} nodes with coordinates")
    
    # Find spatial candidates
    print(f"\nFinding spatial candidates (radius={radius}m)...")
    (candidates_G1_to_G2, distances_G1_to_G2,
     candidates_G2_to_G1, distances_G2_to_G1) = find_spatial_candidates(
        coords_G1, coords_G2, radius=radius
    )
    
    # Print statistics
    print_candidate_statistics(
        candidates_G1_to_G2, distances_G1_to_G2,
        dataset1_name, dataset2_name
    )
    print_candidate_statistics(
        candidates_G2_to_G1, distances_G2_to_G1,
        dataset2_name, dataset1_name
    )
    
    # Save results
    print(f"\nSaving candidates...")
    
    # Save for dataset 1
    output_dir_1 = base_dir / dataset1_name
    output_dir_1.mkdir(parents=True, exist_ok=True)
    output_path_1 = output_dir_1 / f"spatial_candidates_to_{dataset2_name}_{radius}m.pkl"
    
    with open(output_path_1, 'wb') as f:
        pickle.dump({
            'candidates': candidates_G1_to_G2,
            'distances': distances_G1_to_G2,
            'source': dataset1_name,
            'target': dataset2_name,
            'radius': radius
        }, f)
    print(f"  ✓ Saved: {output_path_1}")
    
    # Save for dataset 2
    output_dir_2 = base_dir / dataset2_name
    output_dir_2.mkdir(parents=True, exist_ok=True)
    output_path_2 = output_dir_2 / f"spatial_candidates_to_{dataset1_name}_{radius}m.pkl"
    
    with open(output_path_2, 'wb') as f:
        pickle.dump({
            'candidates': candidates_G2_to_G1,
            'distances': distances_G2_to_G1,
            'source': dataset2_name,
            'target': dataset1_name,
            'radius': radius
        }, f)
    print(f"  ✓ Saved: {output_path_2}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nProcessed {len(dataset_pairs)} dataset pair(s):")
for dataset1, dataset2 in dataset_pairs:
    print(f"  ✓ {dataset1} ↔ {dataset2}")

print("\n" + "="*60)
print("✓ SPATIAL CANDIDATE FINDING COMPLETE")
print("="*60)