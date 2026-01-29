"""
Step 3: Compute structural similarities for ALL cross-graph pairs (no spatial filter).

This script:
1. Loads configuration
2. Loads graphs
3. Computes structural similarity using CENA k-hop degree sequences
   for all (u in G1, v in G2) and (v in G2, u in G1)
4. Saves structural similarities for each pair
"""
import sys
import os
import pickle
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.matching.structural import (
    compute_structural_similarity,
    print_similarity_statistics
)


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 3: COMPUTE STRUCTURAL SIMILARITIES")
print("="*60)

# Load config
config_path = Path("config/default.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get parameters
dataset_pairs = config['dataset_pairs']
zone = config['zone']
radius = config['spatial']['radius']      # still loaded, but unused here
K = config['structural']['K']
alpha = config['structural']['alpha']

PROCESSED_DATA_DIR = Path("data/processed")

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Spatial radius (unused for CENA): {radius}m")
print(f"  K-hop depth: {K}")
print(f"  Alpha: {alpha}")
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
    
    # ============================================
    # Load Graphs
    # ============================================
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
    
    print(f"  ✓ {dataset1_name}: {G1.number_of_nodes():,} nodes, {G1.number_of_edges():,} edges")
    print(f"  ✓ {dataset2_name}: {G2.number_of_nodes():,} nodes, {G2.number_of_edges():,} edges")
    
    # ============================================
    # Build FULL candidate sets (NO spatial filter)
    # ============================================
    print("\nBuilding full cross-graph candidate sets (no spatial filter)...")

    nodes_G1 = list(G1.nodes())
    nodes_G2 = list(G2.nodes())

    # For each node in G1, all nodes in G2 are candidates
    candidates_G1_to_G2 = {
        u: nodes_G2[:]  # shallow copy, for clarity
        for u in nodes_G1
    }

    # For each node in G2, all nodes in G1 are candidates
    candidates_G2_to_G1 = {
        v: nodes_G1[:]
        for v in nodes_G2
    }

    total_pairs_1 = sum(len(cands) for cands in candidates_G1_to_G2.values())
    total_pairs_2 = sum(len(cands) for cands in candidates_G2_to_G1.values())

    print(f"  ✓ {dataset1_name} → {dataset2_name}: {total_pairs_1:,} candidate pairs (FULL)")
    print(f"  ✓ {dataset2_name} → {dataset1_name}: {total_pairs_2:,} candidate pairs (FULL)")
    
    # ============================================
    # Compute Structural Similarities
    # ============================================
    
    # Direction 1: G1 → G2
    print(f"\n{'='*60}")
    print(f"COMPUTING: {dataset1_name} → {dataset2_name}")
    print(f"{'='*60}")
    
    similarities_G1_to_G2 = compute_structural_similarity(
        G1, G2, candidates_G1_to_G2, K=K, alpha=alpha
    )
    
    print_similarity_statistics(
        similarities_G1_to_G2, dataset1_name, dataset2_name
    )
    
    # Direction 2: G2 → G1
    print(f"\n{'='*60}")
    print(f"COMPUTING: {dataset2_name} → {dataset1_name}")
    print(f"{'='*60}")
    
    similarities_G2_to_G1 = compute_structural_similarity(
        G2, G1, candidates_G2_to_G1, K=K, alpha=alpha
    )
    
    print_similarity_statistics(
        similarities_G2_to_G1, dataset2_name, dataset1_name
    )
    
    # ============================================
    # Save Results
    # ============================================
    
    print(f"\nSaving structural similarities...")
    
    # Save for dataset 1
    output_dir_1 = base_dir / dataset1_name
    # To this (match the expected pattern):
    output_path_1 = output_dir_1 / f"structural_similarities_to_{dataset2_name}_{radius}m_K{K}.pkl"

    
    # Line ~117:
    with open(output_path_1, 'wb') as f:
        pickle.dump({
            'similarities': similarities_G1_to_G2,
            'source': dataset1_name,
            'target': dataset2_name,
            'radius': 'FULL',  # Changed from None
            'K': K,
            'alpha': alpha,
            'method': 'CENA_FULL'  # Add this
        }, f)
    print(f"  ✓ Saved: {output_path_1}")
    
    # Save for dataset 2
    output_dir_2 = base_dir / dataset2_name
    output_path_2 = output_dir_2 / f"structural_similarities_to_{dataset1_name}_{radius}m_K{K}.pkl"
    
    with open(output_path_2, 'wb') as f:
        pickle.dump({
            'similarities': similarities_G2_to_G1,
            'source': dataset2_name,
            'target': dataset1_name,
            'radius': None,
            'K': K,
            'alpha': alpha
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
print("✓ STRUCTURAL SIMILARITY COMPUTATION COMPLETE (FULL CENA)")
print("="*60)
