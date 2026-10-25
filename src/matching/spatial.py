"""Spatial candidate finding using KDTree."""
import numpy as np
from scipy.spatial import KDTree


def find_spatial_candidates(coords_G1, coords_G2, radius=20):
    """
    Find candidate matches within spatial radius using KDTree.
    
    Args:
        coords_G1: dict {node_id: (x, y)} for graph 1
        coords_G2: dict {node_id: (x, y)} for graph 2
        radius: search radius in meters
    
    Returns:
        candidates_G1_to_G2: dict {node_g1: [list of node_g2 candidates]}
        distances_G1_to_G2: dict {node_g1: [list of distances]}
        candidates_G2_to_G1: dict {node_g2: [list of node_g1 candidates]}
        distances_G2_to_G1: dict {node_g2: [list of distances]}
    """
    # Convert to arrays
    nodes_G1 = list(coords_G1.keys())
    nodes_G2 = list(coords_G2.keys())
    
    coords_array_G1 = np.array([coords_G1[n] for n in nodes_G1])
    coords_array_G2 = np.array([coords_G2[n] for n in nodes_G2])
    
    print(f"\nBuilding spatial indices...")
    tree_G1 = KDTree(coords_array_G1)
    tree_G2 = KDTree(coords_array_G2)
    
    # ============================================
    # Direction 1: G1 → G2
    # ============================================
    print(f"Finding candidates G1 → G2 (radius={radius}m)...")
    candidates_G1_to_G2 = {}
    distances_G1_to_G2 = {}
    
    for i, node_g1 in enumerate(nodes_G1):
        coord = coords_array_G1[i]
        indices = tree_G2.query_ball_point(coord, radius)
        
        candidates_G1_to_G2[node_g1] = [nodes_G2[j] for j in indices]
        distances_G1_to_G2[node_g1] = [
            np.linalg.norm(coord - coords_array_G2[j]) for j in indices
        ]
    
    # ============================================
    # Direction 2: G2 → G1
    # ============================================
    print(f"Finding candidates G2 → G1 (radius={radius}m)...")
    candidates_G2_to_G1 = {}
    distances_G2_to_G1 = {}
    
    for i, node_g2 in enumerate(nodes_G2):
        coord = coords_array_G2[i]
        indices = tree_G1.query_ball_point(coord, radius)
        
        candidates_G2_to_G1[node_g2] = [nodes_G1[j] for j in indices]
        distances_G2_to_G1[node_g2] = [
            np.linalg.norm(coord - coords_array_G1[j]) for j in indices
        ]
    
    return (candidates_G1_to_G2, distances_G1_to_G2, 
            candidates_G2_to_G1, distances_G2_to_G1)


def print_candidate_statistics(candidates, distances, graph_name_from, graph_name_to):
    """
    Print statistics about spatial candidates.
    
    Args:
        candidates: dict {node: [list of candidates]}
        distances: dict {node: [list of distances]}
        graph_name_from: source graph name
        graph_name_to: target graph name
    """
    print(f"\n{'='*60}")
    print(f"CANDIDATES: {graph_name_from} → {graph_name_to}")
    print(f"{'='*60}")
    
    # Count candidates per node
    candidate_counts = [len(cands) for cands in candidates.values()]
    
    if not candidate_counts:
        print("  ⚠️ No candidates found!")
        return
    
    # Distribution
    no_candidates = sum(1 for c in candidate_counts if c == 0)
    one_candidate = sum(1 for c in candidate_counts if c == 1)
    few_candidates = sum(1 for c in candidate_counts if 2 <= c <= 5)
    many_candidates = sum(1 for c in candidate_counts if c > 5)
    
    total_nodes = len(candidates)
    
    print(f"\nCandidate Distribution:")
    print(f"  Nodes with 0 candidates: {no_candidates:,} ({100*no_candidates/total_nodes:.1f}%)")
    print(f"  Nodes with 1 candidate:  {one_candidate:,} ({100*one_candidate/total_nodes:.1f}%)")
    print(f"  Nodes with 2-5:          {few_candidates:,} ({100*few_candidates/total_nodes:.1f}%)")
    print(f"  Nodes with >5:           {many_candidates:,} ({100*many_candidates/total_nodes:.1f}%)")
    
    print(f"\nCandidate Statistics:")
    print(f"  Mean candidates per node: {np.mean(candidate_counts):.2f}")
    print(f"  Median: {np.median(candidate_counts):.0f}")
    print(f"  Max: {np.max(candidate_counts)}")
    
    # Distance statistics (only for nodes with candidates)
    all_distances = [d for dists in distances.values() for d in dists if len(dists) > 0]
    
    if all_distances:
        print(f"\nDistance Statistics:")
        print(f"  Mean: {np.mean(all_distances):.2f}m")
        print(f"  Median: {np.median(all_distances):.2f}m")
        print(f"  Min: {np.min(all_distances):.2f}m")
        print(f"  Max: {np.max(all_distances):.2f}m")