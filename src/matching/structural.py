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

# ----------------------------------------------------------------------
# Vectorised all-pairs variant
# ----------------------------------------------------------------------
# compute_structural_distance() above only ever uses the MIN and MAX of each
# k-hop degree sequence, so a node is fully described by 2(K+1) numbers:
#     [log(min_k + 1), log(max_k + 1)] for k = 0..K
# and the distance between two nodes is the L1 distance between those vectors.
# Recomputing the BFS per pair costs ~18 h per direction on a 7,800-node zone;
# summarising each node once and then comparing vectors takes seconds, and gives
# exactly the same numbers.

def structural_features(G, K, rectify=1.0):
    """Return (nodes, F) with F[i] the 2(K+1) summary of node i (log min/max
    rectified degree per hop). Hops with no neighbour repeat the previous hop,
    which is what skipping an empty sequence amounts to in the pairwise code."""
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    deg = np.array([G.degree(n) for n in nodes], dtype=float) * rectify
    F = np.zeros((len(nodes), 2 * (K + 1)))
    F[:, 0] = np.log(deg + 1.0)
    F[:, 1] = F[:, 0]

    nbrs = [np.array([idx[m] for m in G.neighbors(n)], dtype=np.int64) for n in nodes]
    for i, n in enumerate(nodes):
        visited = {i}
        current = {i}
        for k in range(1, K + 1):
            nxt = set()
            for u in current:
                for v in nbrs[u]:
                    v = int(v)
                    if v not in visited:
                        visited.add(v)
                        nxt.add(v)
            if not nxt:                       # no k-hop ring: carry the last one
                F[i, 2 * k] = F[i, 2 * (k - 1)]
                F[i, 2 * k + 1] = F[i, 2 * (k - 1) + 1]
                current = nxt
                continue
            d = deg[list(nxt)]
            F[i, 2 * k] = np.log(d.min() + 1.0)
            F[i, 2 * k + 1] = np.log(d.max() + 1.0)
            current = nxt
    return nodes, F


def top_structural_candidates(G1, G2, K=2, alpha=1.0, top_m=50, chunk=512, verbose=True):
    """For every node of G1, its `top_m` structurally closest nodes of G2 (and
    the reverse). No coordinates are used. Returns
    (candidates_G1_to_G2, similarities_G1_to_G2, candidates_G2_to_G1, similarities_G2_to_G1)
    with similarities keyed by (node_g1, node_g2) as in compute_structural_similarity."""
    r1 = np.sqrt((len(G2.edges()) / len(G2.nodes())) / (len(G1.edges()) / len(G1.nodes())))
    n1, F1 = structural_features(G1, K, r1)
    n2, F2 = structural_features(G2, K, 1.0 / r1)
    if verbose:
        print(f"  summaries: {len(n1):,} x {F1.shape[1]} and {len(n2):,} x {F2.shape[1]}")

    m = min(top_m, len(n2))
    m_rev = min(top_m, len(n1))
    cand12, sim12 = {}, {}
    best_rev_d = np.full((len(n2), m_rev), np.inf)   # for the reverse direction
    best_rev_i = np.zeros((len(n2), m_rev), dtype=np.int64)

    for start in range(0, len(n1), chunk):
        block = F1[start:start + chunk]
        D = np.abs(block[:, None, :] - F2[None, :, :]).sum(axis=2)    # L1, exact
        part = np.argpartition(D, m - 1, axis=1)[:, :m]
        for bi in range(D.shape[0]):
            i = start + bi
            js = part[bi][np.argsort(D[bi, part[bi]])]
            cand12[n1[i]] = [n2[j] for j in js]
            for j in js:
                sim12[(n1[i], n2[j])] = float(np.exp(-alpha * D[bi, j]))
        # keep the reverse direction's best rows without recomputing D
        merged_d = np.hstack([best_rev_d, D.T])
        merged_i = np.hstack([best_rev_i, np.tile(np.arange(start, start + D.shape[0]),
                                                  (len(n2), 1))])
        keep = np.argpartition(merged_d, m_rev - 1, axis=1)[:, :m_rev]
        rows = np.arange(len(n2))[:, None]
        best_rev_d = merged_d[rows, keep]
        best_rev_i = merged_i[rows, keep]
        if verbose and (start // chunk) % 4 == 0:
            print(f"    {min(start + chunk, len(n1)):,}/{len(n1):,} nodes", flush=True)

    cand21, sim21 = {}, {}
    for j in range(len(n2)):
        order = np.argsort(best_rev_d[j])
        cand21[n2[j]] = [n1[best_rev_i[j, o]] for o in order]
        for o in order:
            sim21[(n2[j], n1[best_rev_i[j, o]])] = float(np.exp(-alpha * best_rev_d[j, o]))
    return cand12, sim12, cand21, sim21
