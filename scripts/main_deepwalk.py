#!/usr/bin/env python3
"""
DeepWalk baseline for network alignment.
Tests independent embeddings with post-hoc alignment.
"""

import sys
import yaml
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

print("✓ Imports successful")


def load_config(config_path="config/deepwalk.yaml"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    return config


def load_graphs_from_edgelist(config):
    """Load graphs from edgelists and coordinates from CSV"""
    import csv
    
    processed_dir = Path(config['paths']['data_processed'])
    dataset_pair = config['dataset_pairs'][0]
    zone = config.get('zone', 'full')
        


    print(f"Zone: {zone}")
    
    # Determine file naming based on zone
    if zone == "prades":
        suffix = "_prades"
    elif zone == "full":
        suffix = ""
    else:
        raise ValueError(f"Unknown zone: {zone}. Expected 'prades' or 'full'")
    
    suffix = f"_{zone.lower()}" if zone.lower() != "full" else ""

    edgelist1 = processed_dir / dataset_pair[0] / f"{dataset_pair[0]}{suffix}.edgelist"
    edgelist2 = processed_dir / dataset_pair[1] / f"{dataset_pair[1]}{suffix}.edgelist"

 
    print(f"Loading {dataset_pair[0]} from {edgelist1}")
    G = nx.read_edgelist(str(edgelist1))
    
    print(f"Loading {dataset_pair[1]} from {edgelist2}")
    G_prime = nx.read_edgelist(str(edgelist2))
    
    # Load coordinates from CSV
    coords1 = processed_dir / dataset_pair[0] / f"{dataset_pair[0]}{suffix}_coords.csv"
    coords2 = processed_dir / dataset_pair[1] / f"{dataset_pair[1]}{suffix}_coords.csv"
    
    
    print(f"Loading coordinates from CSV files...")
    
    # Load coords for first dataset
    with open(coords1, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['node_id']
            if node_id in G.nodes():
                G.nodes[node_id]['x'] = float(row['x'])
                G.nodes[node_id]['y'] = float(row['y'])
    
    # Load coords for second dataset
    with open(coords2, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['node_id']
            if node_id in G_prime.nodes():
                G_prime.nodes[node_id]['x'] = float(row['x'])
                G_prime.nodes[node_id]['y'] = float(row['y'])
    
    print(f"✓ Loaded {dataset_pair[0]}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"✓ Loaded {dataset_pair[1]}: {G_prime.number_of_nodes()} nodes, {G_prime.number_of_edges()} edges")
    
    return G, G_prime, dataset_pair


def hungarian_alignment(similarity_matrix):
    """Apply Hungarian algorithm to find optimal alignment"""
    print("\nApplying Hungarian algorithm...")
    
    # Convert similarity to cost (Hungarian minimizes)
    cost_matrix = 1 - similarity_matrix
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print(f"  ✓ Found {len(row_ind)} alignments")
    
    return row_ind, col_ind


def evaluate_alignment(row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime):
    """
    Evaluate alignment against ground truth.
    Ground truth: nodes with distance < 0.1m
    Uses KDTree for efficient spatial queries.
    """
    from scipy.spatial import cKDTree
    
    print("\nEvaluating alignment...")
    
    # Build alignment dictionary
    alignment = {}
    for i, j in zip(row_ind, col_ind):
        node_u = nodes_G[i]
        node_v = nodes_G_prime[j]
        alignment[node_u] = node_v
    
    # Compute ground truth using KDTree (MUCH faster!)
    print("  Computing ground truth (distance < 0.1m) using KDTree...")
    
    # Build coordinate arrays
    coords_G = np.array([[G.nodes[node]['x'], G.nodes[node]['y']] for node in nodes_G])
    coords_G_prime = np.array([[G_prime.nodes[node]['x'], G_prime.nodes[node]['y']] for node in nodes_G_prime])
    
    # Build KDTree for G_prime
    tree = cKDTree(coords_G_prime)
    
    # Query for all nodes in G within 0.1m
    ground_truth = set()
    distances, indices = tree.query(coords_G, distance_upper_bound=0.1)
    
    for i, (dist, j) in enumerate(zip(distances, indices)):
        if dist < 0.1:  # Found a match
            node_u = nodes_G[i]
            node_v = nodes_G_prime[j]
            ground_truth.add((node_u, node_v))
    
    print(f"  Ground truth pairs: {len(ground_truth)}")
    
    # Evaluate
    true_positives = 0
    for node_u, node_v in alignment.items():
        if (node_u, node_v) in ground_truth:
            true_positives += 1
    
    precision = true_positives / len(alignment) if len(alignment) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Results:")
    print(f"    Precision: {precision*100:.2f}%")
    print(f"    Recall: {recall*100:.2f}%")
    print(f"    F1-Score: {f1*100:.2f}%")
    print(f"    True Positives: {true_positives}/{len(alignment)}")
    
    return precision, recall, f1


def run_deepwalk(G, config):
    """Run DeepWalk on a graph"""
    from gensim.models import Word2Vec
    
    # Generate random walks
    walks = []
    for node in G.nodes():
        for _ in range(config['deepwalk']['num_walks']):
            walk = [node]
            for _ in range(config['deepwalk']['walk_length'] - 1):
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if len(neighbors) > 0:
                    walk.append(np.random.choice(neighbors))
            walks.append([str(n) for n in walk])
    
    # Train Word2Vec
    model = Word2Vec(walks, vector_size=config['deepwalk']['dimensions'], 
                     window=config['deepwalk']['window_size'], 
                     min_count=0, sg=1, workers=config['deepwalk']['workers'])
    
    # Extract embeddings
    embeddings = {node: model.wv[str(node)] for node in G.nodes()}
    return embeddings


def procrustes_alignment_supervised(emb_G, emb_G_prime, nodes_G, nodes_G_prime, G, G_prime, threshold=0.1):
    """
    Align embeddings using Procrustes with ground truth anchors (SUPERVISED).
    
    Parameters:
    -----------
    emb_G, emb_G_prime : dict
        Node embeddings for each graph
    nodes_G, nodes_G_prime : list
        Node lists for each graph
    G, G_prime : networkx.Graph
        Graphs with coordinate information
    threshold : float
        Distance threshold for ground truth anchors (default: 0.1m)
    
    Returns:
    --------
    emb_G_aligned : np.array
        Aligned embeddings for G
    anchor_count : int
        Number of anchors used
    """
    from scipy.linalg import orthogonal_procrustes
    from scipy.spatial import cKDTree
    
    print(f"\n  Finding ground truth anchors (threshold={threshold}m)...")
    
    # Build coordinate arrays
    coords_G = np.array([[G.nodes[node]['x'], G.nodes[node]['y']] for node in nodes_G])
    coords_G_prime = np.array([[G_prime.nodes[node]['x'], G_prime.nodes[node]['y']] for node in nodes_G_prime])
    
    # Build KDTree for efficient spatial search
    tree = cKDTree(coords_G_prime)
    
    # Find ground truth anchors
    anchor_pairs = []
    distances, indices = tree.query(coords_G, distance_upper_bound=threshold)
    
    for i, (dist, j) in enumerate(zip(distances, indices)):
        if dist < threshold:  # Found a match
            anchor_pairs.append((i, j))
    
    print(f"  Found {len(anchor_pairs)} ground truth anchor pairs")
    
    if len(anchor_pairs) < 10:
        print("  ⚠️ Too few anchors for reliable Procrustes alignment!")
        return None, 0
    
    # Extract anchor embeddings
    anchor_indices_G = [pair[0] for pair in anchor_pairs]
    anchor_indices_G_prime = [pair[1] for pair in anchor_pairs]
    
    # Convert embeddings dict to matrices
    emb_matrix_G = np.array([emb_G[nodes_G[i]] for i in range(len(nodes_G))])
    emb_matrix_G_prime = np.array([emb_G_prime[nodes_G_prime[j]] for j in range(len(nodes_G_prime))])
    
    anchor_emb_G = emb_matrix_G[anchor_indices_G]
    anchor_emb_G_prime = emb_matrix_G_prime[anchor_indices_G_prime]
    
    # Learn orthogonal transformation using Procrustes
    print(f"  Learning Procrustes transformation...")
    R, _ = orthogonal_procrustes(anchor_emb_G, anchor_emb_G_prime)
    
    # Transform G embeddings to G_prime's space
    emb_G_aligned = emb_matrix_G @ R
    
    print(f"  ✓ Embeddings aligned using {len(anchor_pairs)} anchors")
    
    return emb_G_aligned, len(anchor_pairs)


if __name__ == "__main__":
    config = load_config()
    G, G_prime, dataset_pair = load_graphs_from_edgelist(config)
    
    print("\n" + "="*60)
    print("RUNNING DEEPWALK")
    print("="*60)
    
    print(f"\n[1/2] Training DeepWalk on {dataset_pair[0]}...")
    embeddings_G = run_deepwalk(G, config)

    print(f"\n[2/2] Training DeepWalk on {dataset_pair[1]}...")
    embeddings_G_prime = run_deepwalk(G_prime, config)
        
    print("\n" + "="*60)
    print("COMPUTING SIMILARITIES")
    print("="*60)
    
    # Convert embeddings dict to matrix
    print("\nPreparing embedding matrices...")
    nodes_G = sorted(G.nodes())
    nodes_G_prime = sorted(G_prime.nodes())
    
    emb_matrix_G = np.array([embeddings_G[node] for node in nodes_G])
    emb_matrix_G_prime = np.array([embeddings_G_prime[node] for node in nodes_G_prime])
    
    print(f"  {dataset_pair[0]} matrix: {emb_matrix_G.shape}")
    print(f"  {dataset_pair[1]} matrix: {emb_matrix_G_prime.shape}")
    
    # Method 1: Direct comparison
    print("\n[1/2] Direct cosine similarity...")
    similarity_direct = cosine_similarity(emb_matrix_G, emb_matrix_G_prime)
    print(f"  Similarity matrix: {similarity_direct.shape}")
    print(f"  Mean similarity: {similarity_direct.mean():.4f}")
    
    # Method 2: Normalized
    print("\n[2/2] Normalized cosine similarity...")
    emb_G_norm = normalize(emb_matrix_G, axis=1)
    emb_G_prime_norm = normalize(emb_matrix_G_prime, axis=1)
    similarity_normalized = cosine_similarity(emb_G_norm, emb_G_prime_norm)
    print(f"  Similarity matrix: {similarity_normalized.shape}")
    print(f"  Mean similarity: {similarity_normalized.mean():.4f}")
    
    print("\n✓ Similarities computed!")

    # After computing similarities, add:
    
    print("\n" + "="*60)
    print("ALIGNMENT AND EVALUATION")
    print("="*60)
    
    # Evaluate Direct method
    print("\n[Method 1: Direct Comparison]")
    row_ind, col_ind = hungarian_alignment(similarity_direct)
    prec_direct, rec_direct, f1_direct = evaluate_alignment(
        row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime
    )
    
    # Evaluate Normalized method
    print("\n[Method 2: Normalized Comparison]")
    row_ind, col_ind = hungarian_alignment(similarity_normalized)
    prec_norm, rec_norm, f1_norm = evaluate_alignment(
        row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime
    )
    
    # Evaluate Procrustes method (SUPERVISED - uses ground truth)
    print("\n[Method 3: Procrustes Alignment (SUPERVISED - uses ground truth)]")
    emb_G_aligned, anchor_count = procrustes_alignment_supervised(
        embeddings_G, embeddings_G_prime, nodes_G, nodes_G_prime, G, G_prime, threshold=0.1
    )
    
    if emb_G_aligned is not None:
        similarity_procrustes = cosine_similarity(emb_G_aligned, emb_matrix_G_prime)
        print(f"  Similarity matrix: {similarity_procrustes.shape}")
        print(f"  Mean similarity: {similarity_procrustes.mean():.4f}")
        
        row_ind, col_ind = hungarian_alignment(similarity_procrustes)
        prec_procrustes, rec_procrustes, f1_procrustes = evaluate_alignment(
            row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime
        )
    else:
        print("  ⚠️ Procrustes alignment failed")
        f1_procrustes = 0.0
        anchor_count = 0
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nDirect (Unsupervised):     F1={f1_direct*100:.2f}%")
    print(f"Normalized (Unsupervised): F1={f1_norm*100:.2f}%")
    print(f"Procrustes (SUPERVISED):   F1={f1_procrustes*100:.2f}% (used {anchor_count} ground truth anchors)")
    print("\n⚠️ Note: Procrustes uses ground truth for alignment (supervised)")