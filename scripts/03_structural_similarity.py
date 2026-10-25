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
    print_similarity_statistics,
    top_structural_candidates,
)


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 3: COMPUTE STRUCTURAL SIMILARITIES")
print("="*60)

# Load config
config_path = Path(os.environ.get("SANA_CONFIG", "config/default.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get parameters
dataset_pairs = config['dataset_pairs']
zone = config['zone']
radius = config['spatial']['radius']
K = config['structural']['K']
alpha = config['structural']['alpha']

PROCESSED_DATA_DIR = Path("data/processed")

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Candidates: {config['structural'].get('candidates', 'radius')} (radius {radius}m)")
print(f"  K-hop depth: {K}")
print(f"  Alpha: {alpha}")
print(f"  Dataset pairs: {dataset_pairs}")

# Determine base directory
if str(zone).lower() == "full":
    base_dir = PROCESSED_DATA_DIR
else:
    base_dir = PROCESSED_DATA_DIR / zone
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
    # Candidate sets
    # structural.candidates = "radius" (default): only the pairs inside the
    # spatial candidate radius, read from step 02 - the same candidate set as
    # SANA and NN. "all": every cross-graph pair, as in the original CENA
    # (|V| x |V'| pairs; ~18 h per direction on Z1).
    # ============================================
    CAND_MODE = config['structural'].get('candidates', 'all')
    TOP_M = int(config['structural'].get('top_m', 50))
    nodes_G1 = list(G1.nodes())
    nodes_G2 = list(G2.nodes())

    if CAND_MODE == 'all':
        # No spatial signal: every pair is compared, then each node keeps its
        # TOP_M structurally closest partners. Comparing all pairs one by one
        # costs hours; the vectorised form below gives identical numbers.
        print(f"\nComparing all cross-graph pairs, keeping the top {TOP_M} per node...")
        candidates_G1_to_G2, similarities_G1_to_G2, candidates_G2_to_G1, similarities_G2_to_G1 = \
            top_structural_candidates(G1, G2, K=K, alpha=alpha, top_m=TOP_M)
    elif CAND_MODE == 'radius':
        print(f"\nLoading spatial candidates (r = {radius} m) from step 02...")
        with open(base_dir / dataset1_name / f"spatial_candidates_to_{dataset2_name}_{radius}m.pkl", 'rb') as f:
            candidates_G1_to_G2 = pickle.load(f)['candidates']
        with open(base_dir / dataset2_name / f"spatial_candidates_to_{dataset1_name}_{radius}m.pkl", 'rb') as f:
            candidates_G2_to_G1 = pickle.load(f)['candidates']
        similarities_G1_to_G2 = compute_structural_similarity(G1, G2, candidates_G1_to_G2, K=K, alpha=alpha)
        similarities_G2_to_G1 = compute_structural_similarity(G2, G1, candidates_G2_to_G1, K=K, alpha=alpha)
    else:
        raise ValueError(f"structural.candidates must be 'all' or 'radius', got {CAND_MODE!r}")

    total_pairs_1 = sum(len(c) for c in candidates_G1_to_G2.values())
    total_pairs_2 = sum(len(c) for c in candidates_G2_to_G1.values())
    print(f"  ✓ {dataset1_name} → {dataset2_name}: {total_pairs_1:,} candidate pairs ({CAND_MODE})")
    print(f"  ✓ {dataset2_name} → {dataset1_name}: {total_pairs_2:,} candidate pairs ({CAND_MODE})")

    print_similarity_statistics(similarities_G1_to_G2, dataset1_name, dataset2_name)
    print_similarity_statistics(similarities_G2_to_G1, dataset2_name, dataset1_name)

    print(f"\nSaving structural similarities...")
    
    # Save for dataset 1
    output_dir_1 = base_dir / dataset1_name
    # To this (match the expected pattern):
    output_path_1 = output_dir_1 / f"structural_similarities_to_{dataset2_name}_{radius}m_K{K}.pkl"

    
    # Line ~117:
    with open(output_path_1, 'wb') as f:
        pickle.dump({
            'similarities': similarities_G1_to_G2,
            'candidates': candidates_G1_to_G2,
            'source': dataset1_name,
            'target': dataset2_name,
            'radius': radius if CAND_MODE == 'radius' else 'FULL',
            'K': K,
            'alpha': alpha,
            'method': 'CENA_' + CAND_MODE.upper(), 'top_m': TOP_M
        }, f)
    print(f"  ✓ Saved: {output_path_1}")
    
    # Save for dataset 2
    output_dir_2 = base_dir / dataset2_name
    output_path_2 = output_dir_2 / f"structural_similarities_to_{dataset1_name}_{radius}m_K{K}.pkl"
    
    with open(output_path_2, 'wb') as f:
        pickle.dump({
            'similarities': similarities_G2_to_G1,
            'candidates': candidates_G2_to_G1,
            'source': dataset2_name,
            'target': dataset1_name,
            'radius': radius if CAND_MODE == 'radius' else 'FULL',
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
