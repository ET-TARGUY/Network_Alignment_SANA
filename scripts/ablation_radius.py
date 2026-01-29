#!/usr/bin/env python3
"""
Ablation study: Impact of spatial radius on alignment performance
"""

import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree



print("✓ Imports successful")


def load_config(config_path="config/default.yaml"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    return config



def compute_ground_truth(G, G_prime, threshold=0.1):
    """
    Compute ground truth correspondences (distance < threshold)
    Returns: list of tuples (node_u, node_v, distance)
    """
    print(f"\nComputing ground truth (distance < {threshold}m)...")
    ground_truth = []
    
    nodes_G = list(G.nodes())
    nodes_G_prime = list(G_prime.nodes())
    
    # Build coordinate arrays
    coords_G = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nodes_G])
    coords_G_prime = np.array([[G_prime.nodes[n]['x'], G_prime.nodes[n]['y']] for n in nodes_G_prime])
    
    # Build KDTree for G_prime
    tree = cKDTree(coords_G_prime)
    
    # Query for each node in G
    for i, node_u in enumerate(tqdm(nodes_G, desc="Computing ground truth")):
        indices = tree.query_ball_point(coords_G[i], r=threshold)
        for j in indices:
            node_v = nodes_G_prime[j]
            distance = np.linalg.norm(coords_G[i] - coords_G_prime[j])
            ground_truth.append((node_u, node_v, distance))
    
    print(f"✓ Found {len(ground_truth)} ground truth pairs")
    return ground_truth


def compute_spatial_candidates_all(G, G_prime, max_radius=40):
    """
    Compute ALL spatial candidates within max_radius using KDTree
    Returns: list of tuples (node_u, node_v, distance)
    """
    print(f"\nComputing spatial candidates (max radius = {max_radius}m)...")
    candidates = []
    
    nodes_G = list(G.nodes())
    nodes_G_prime = list(G_prime.nodes())
    
    # Build coordinate arrays
    coords_G = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nodes_G])
    coords_G_prime = np.array([[G_prime.nodes[n]['x'], G_prime.nodes[n]['y']] for n in nodes_G_prime])
    
    # Build KDTree for G_prime
    tree = cKDTree(coords_G_prime)
    
    # Query for each node in G
    for i, node_u in enumerate(tqdm(nodes_G, desc="Finding candidates")):
        indices = tree.query_ball_point(coords_G[i], r=max_radius)
        for j in indices:
            node_v = nodes_G_prime[j]
            distance = np.linalg.norm(coords_G[i] - coords_G_prime[j])
            candidates.append((node_u, node_v, distance))
    
    print(f"✓ Found {len(candidates):,} candidate pairs within {max_radius}m")
    return candidates


def analyze_radius_coverage(G, G_prime, all_candidates, ground_truth, radii):
    """
    Analyze coverage metrics for different radii
    Returns: DataFrame with results
    """
    print("\n" + "="*60)
    print("ANALYZING COVERAGE FOR DIFFERENT RADII")
    print("="*60)
    
    results = []
    total_nodes_G = G.number_of_nodes()
    total_gt = len(ground_truth)
    
    for r in radii:
        print(f"\nRadius = {r}m:")
        
        # Filter candidates for this radius
        candidates_r = [(u, v, d) for u, v, d in all_candidates if d <= r]
        
        # Node coverage: nodes with at least one candidate
        nodes_with_candidates = len(set(u for u, v, d in candidates_r))
        node_coverage = 100 * nodes_with_candidates / total_nodes_G
        
        # Ground truth coverage: GT pairs within radius
        gt_within_r = [(u, v, d) for u, v, d in ground_truth if d <= r]
        gt_coverage = 100 * len(gt_within_r) / total_gt
        
        # Average candidates per node
        avg_candidates = len(candidates_r) / total_nodes_G
        
        print(f"  Node coverage: {node_coverage:.2f}%")
        print(f"  GT coverage: {gt_coverage:.2f}%")
        print(f"  Avg candidates: {avg_candidates:.1f}")
        print(f"  Total candidates: {len(candidates_r):,}")
        
        results.append({
            'radius': r,
            'node_coverage': node_coverage,
            'gt_coverage': gt_coverage,
            'avg_candidates': avg_candidates,
            'total_candidates': len(candidates_r)
        })
    
    return pd.DataFrame(results)

def load_graphs(config):
    """Load processed graphs"""
    processed_dir = Path(config['paths']['data_processed'])
    
    # Correct paths
    d1_path = processed_dir / "Dataset1" / "Graph.pkl"
    d5_path = processed_dir / "Dataset5" / "Graph.pkl"
    
    with open(d1_path, 'rb') as f:
        G = pickle.load(f)
    with open(d5_path, 'rb') as f:
        G_prime = pickle.load(f)
    
    print(f"✓ Loaded Dataset1: {G.number_of_nodes()} nodes")
    print(f"✓ Loaded Dataset5: {G_prime.number_of_nodes()} nodes")
    
    return G, G_prime
if __name__ == "__main__":
    print("="*60)
    print("ABLATION STUDY: SPATIAL RADIUS")
    print("="*60)
    
    config = load_config()
    G, G_prime = load_graphs(config)
    
    # Compute ground truth
    ground_truth = compute_ground_truth(G, G_prime, threshold=0.1)
    
    # Compute all candidates for max radius
    max_radius = 40
    try:
        all_candidates = compute_spatial_candidates_all(G, G_prime, max_radius)
        print(f"✓ Candidates computed: {len(all_candidates)}")
    except Exception as e:
        print(f"❌ Error computing candidates: {e}")
        import traceback
        traceback.print_exc()


    # Analyze different radii
    radii = [5, 10, 15, 20, 25, 30, 40]
    coverage_df = analyze_radius_coverage(G, G_prime, all_candidates, ground_truth, radii)
    
    print("\n" + "="*60)
    print("COVERAGE SUMMARY")
    print("="*60)
    print(coverage_df.to_string(index=False))