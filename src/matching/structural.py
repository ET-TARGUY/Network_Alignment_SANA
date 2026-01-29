"""Structural similarity computation using CENA k-hop degree sequences."""
import numpy as np
import networkx as nx
from tqdm import tqdm


def compute_rectified_degree(degree, G_edges, G_nodes, G_prime_edges, G_prime_nodes):
    """
    Rectify degree to maintain consistency across networks.
    d̃ = d * sqrt(|E'|/|V'| / |E|/|V|)
    """
    ratio = (G_prime_edges / G_prime_nodes) / (G_edges / G_nodes)
    return degree * np.sqrt(ratio)


def get_k_hop_degree_sequence(G, node, k, rectify_factor=1.0):
    """
    Get rectified degree sequence of k-hop neighbors.
    Returns sorted list of rectified degrees.
    
    k=0: just the node itself
    k=1: 1-hop neighbors
    k=2: 2-hop neighbors, etc.
    """
    if k == 0:
        # 0-hop is the node itself
        degree = G.degree(node)
        return [degree * rectify_factor]
    
    # Get k-hop neighbors using BFS
    visited = {node}
    current_level = {node}
    
    for _ in range(k):
        next_level = set()
        for n in current_level:
            for neighbor in G.neighbors(n):
                if neighbor not in visited:
                    next_level.add(neighbor)
                    visited.add(neighbor)
        current_level = next_level
        
        if not current_level:
            break
    
    # Get degrees of k-hop neighbors
    degrees = [G.degree(n) * rectify_factor for n in current_level]
    return sorted(degrees) if degrees else []


def compute_structural_distance(G1, G2, u, v, K=2):
    """
    Compute structural distance f(u,v) between nodes from different graphs.
    
    f(u,v) = Σ_{k=0}^K dist(s_k(u), s_k(v))
    
    where dist is the min/max degree difference (Eq. 2 in CENA paper)
    
    Args:
        G1: NetworkX graph 1
        G2: NetworkX graph 2
        u: node in G1
        v: node in G2
        K: k-hop neighborhood depth
    
    Returns:
        float: structural distance (lower = more similar)
    """
    # Safety check
    if len(G1.edges()) == 0 or len(G2.edges()) == 0:
        return float('inf')
    
    # Compute rectification factors
    rectify_G1 = np.sqrt((len(G2.edges()) / len(G2.nodes())) / (len(G1.edges()) / len(G1.nodes())))
    rectify_G2 = 1.0 / rectify_G1
    
    total_distance = 0.0
    
    for k in range(K + 1):
        # Get k-hop degree sequences
        s_u = get_k_hop_degree_sequence(G1, u, k, rectify_G1)
        s_v = get_k_hop_degree_sequence(G2, v, k, rectify_G2)
        
        # Handle empty sequences
        if not s_u or not s_v:
            continue
        
        # Compute distance using min and max degrees (Eq. 2)
        min_diff = abs(np.log(min(s_u) + 1) - np.log(min(s_v) + 1))
        max_diff = abs(np.log(max(s_u) + 1) - np.log(max(s_v) + 1))
        
        total_distance += min_diff + max_diff
    
    return total_distance


def compute_structural_similarity(G1, G2, candidates, K=2, alpha=1.0):
    """
    Compute structural similarities for all candidate pairs.
    
    Converts distance to similarity: sim = exp(-alpha * distance)
    
    Args:
        G1: NetworkX graph 1
        G2: NetworkX graph 2
        candidates: dict {node_g1: [list of node_g2 candidates]}
        K: k-hop neighborhood depth
        alpha: decay parameter for distance-to-similarity conversion
    
    Returns:
        similarities: dict {(node_g1, node_g2): similarity_score}
    """
    similarities = {}
    
    # Count total pairs
    total_pairs = sum(len(cands) for cands in candidates.values())
    
    print(f"\nComputing structural similarities for {total_pairs:,} candidate pairs...")
    print(f"  K-hop depth: {K}")
    print(f"  Alpha: {alpha}")
    
    # Progress bar
    with tqdm(total=total_pairs, desc="  Progress") as pbar:
        for node_g1, cands in candidates.items():
            if node_g1 not in G1:
                continue
            
            for node_g2 in cands:
                if node_g2 not in G2:
                    continue
                
                # Compute structural distance
                distance = compute_structural_distance(G1, G2, node_g1, node_g2, K=K)
                
                # Convert to similarity
                similarity = np.exp(-alpha * distance)
                
                similarities[(node_g1, node_g2)] = similarity
                
                pbar.update(1)
    
    return similarities


def print_similarity_statistics(similarities, graph_name_from, graph_name_to):
    """
    Print statistics about structural similarities.
    
    Args:
        similarities: dict {(node_g1, node_g2): similarity}
        graph_name_from: source graph name
        graph_name_to: target graph name
    """
    if not similarities:
        print(f"\n⚠️ No similarities computed for {graph_name_from} → {graph_name_to}")
        return
    
    sim_values = list(similarities.values())
    
    print(f"\n{'='*60}")
    print(f"STRUCTURAL SIMILARITIES: {graph_name_from} → {graph_name_to}")
    print(f"{'='*60}")
    
    print(f"\nSimilarity Statistics:")
    print(f"  Total pairs: {len(similarities):,}")
    print(f"  Mean: {np.mean(sim_values):.4f}")
    print(f"  Median: {np.median(sim_values):.4f}")
    print(f"  Std: {np.std(sim_values):.4f}")
    print(f"  Min: {np.min(sim_values):.4f}")
    print(f"  Max: {np.max(sim_values):.4f}")
    
    # Distribution
    high_sim = sum(1 for s in sim_values if s >= 0.7)
    medium_sim = sum(1 for s in sim_values if 0.3 <= s < 0.7)
    low_sim = sum(1 for s in sim_values if s < 0.3)
    
    print(f"\nSimilarity Distribution:")
    print(f"  High (≥0.7):   {high_sim:,} ({100*high_sim/len(sim_values):.1f}%)")
    print(f"  Medium (0.3-0.7): {medium_sim:,} ({100*medium_sim/len(sim_values):.1f}%)")
    print(f"  Low (<0.3):    {low_sim:,} ({100*low_sim/len(sim_values):.1f}%)")