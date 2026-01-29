#!/usr/bin/env python3
"""
Struc2vec baseline for network alignment.
Tests pure structural identity without spatial filtering.
"""

import sys
sys.path.append('src/struc2vec_lib')  # Add struc2vec to path

import yaml
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import subprocess
import os

print("✓ Imports successful")


def load_config(config_path="config/struc2vec.yaml"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    return config


def load_graphs_from_edgelist(config):
    """Load graphs from edgelists and coordinates from CSV"""
    import networkx as nx
    import csv
    
    temp_dir = Path("temp")
    dataset_pair = config['dataset_pairs'][0]
    
    # Load from edgelists
    edgelist1 = temp_dir / f"{dataset_pair[0].lower()}_prades.edgelist"
    edgelist2 = temp_dir / f"{dataset_pair[1].lower()}_prades.edgelist"
    
    print(f"Loading {dataset_pair[0]} from {edgelist1}")
    G = nx.read_edgelist(str(edgelist1))
    
    print(f"Loading {dataset_pair[1]} from {edgelist2}")
    G_prime = nx.read_edgelist(str(edgelist2))
    
    # Load coordinates from CSV
    coords1 = temp_dir / f"{dataset_pair[0].lower()}_prades_coords.csv"
    coords2 = temp_dir / f"{dataset_pair[1].lower()}_prades_coords.csv"
    
    print(f"Loading coordinates from CSV files...")
    
    # Load coords for Dataset1
    with open(coords1, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['node_id']
            if node_id in G.nodes():
                G.nodes[node_id]['x'] = float(row['x'])
                G.nodes[node_id]['y'] = float(row['y'])
    
    # Load coords for Dataset5
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
    """
    print("\nEvaluating alignment...")
    
    # Build alignment dictionary
    alignment = {}
    for i, j in zip(row_ind, col_ind):
        node_u = nodes_G[i]
        node_v = nodes_G_prime[j]
        alignment[node_u] = node_v
    
    # Compute ground truth (distance < 0.1m)
    print("  Computing ground truth (distance < 0.1m)...")
    ground_truth = set()
    
    for node_u in nodes_G:
        pos_u = np.array([G.nodes[node_u]['x'], G.nodes[node_u]['y']])
        
        for node_v in nodes_G_prime:
            pos_v = np.array([G_prime.nodes[node_v]['x'], G_prime.nodes[node_v]['y']])
            distance = np.linalg.norm(pos_u - pos_v)
            
            if distance < 0.1:
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

def run_struc2vec(G, output_name, config):
    """
    Run struc2vec on a graph by:
    1. Saving graph as edgelist
    2. Running struc2vec via subprocess
    3. Reading back embeddings
    """
    import subprocess
    import sys
    
    temp_dir = Path(config['paths']['temp'])
    temp_dir.mkdir(exist_ok=True)
    
    # Paths
    input_edgelist = temp_dir / f"{output_name}_input.edgelist"
    output_emb = temp_dir / f"{output_name}_output.emb"
    
    # Save graph as edgelist (struc2vec format: node1 node2)
    print(f"  Saving edgelist: {input_edgelist}")
    nx.write_edgelist(G, str(input_edgelist), data=False)
    
    # Build struc2vec command
    struc2vec_params = config['struc2vec']
    
    # Use current Python interpreter (from virtual environment)
    python_executable = sys.executable
    
    cmd = [
        python_executable,  # Changed from "python" to sys.executable
        "src/struc2vec_lib/main.py",
        "--input", str(input_edgelist),
        "--output", str(output_emb),
        "--num-walks", str(struc2vec_params['num_walks']),
        "--walk-length", str(struc2vec_params['walk_length']),
        "--dimensions", str(struc2vec_params['dimensions']),
        "--window-size", str(struc2vec_params['window_size']),
        "--workers", str(struc2vec_params['workers']),
        "--OPT1", str(struc2vec_params['OPT1']),
        "--OPT2", str(struc2vec_params['OPT2']),
        "--OPT3", str(struc2vec_params['OPT3']),
        "--until-layer", str(struc2vec_params['until_layer'])
    ]
    
    print(f"  Running struc2vec...")
    print(f"    Using Python: {python_executable}")
    print(f"    Command: {' '.join(cmd)}")
    
    # Run struc2vec
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Struc2vec completed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Struc2vec failed!")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"  ❌ Error running struc2vec: {e}")
        return None
    
    # Read embeddings
    print(f"  Reading embeddings from: {output_emb}")
    
    if not output_emb.exists():
        print(f"  ❌ Output file not found: {output_emb}")
        return None
    
    embeddings = read_struc2vec_embeddings(output_emb)
    
    print(f"  ✓ Loaded {len(embeddings)} node embeddings")
    
    return embeddings
def read_struc2vec_embeddings(emb_file):
    """
    Read struc2vec embeddings from output file.
    
    Format:
    First line: num_nodes dimensions
    Following lines: node_id dim1 dim2 ... dimN
    """
    embeddings = {}
    
    with open(emb_file, 'r') as f:
        # Skip first line (header)
        header = f.readline().strip().split()
        num_nodes = int(header[0])
        dimensions = int(header[1])
        
        # Read embeddings
        for line in f:
            parts = line.strip().split()
            node_id = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings[node_id] = embedding
    
    print(f"    Dimensions: {dimensions}, Nodes: {len(embeddings)}")
    
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
    print("RUNNING STRUC2VEC")
    print("="*60)
    
    print(f"\n[1/2] Training struc2vec on {dataset_pair[0]}...")
    embeddings_G = run_struc2vec(G, "dataset1", config)
    
    print(f"\n[2/2] Training struc2vec on {dataset_pair[1]}...")
    embeddings_G_prime = run_struc2vec(G_prime, "dataset5", config)
    
    print("\n" + "="*60)
    print("COMPUTING SIMILARITIES")
    print("="*60)
    
    # Convert embeddings dict to matrix
    print("\nPreparing embedding matrices...")
    nodes_G = sorted(G.nodes())
    nodes_G_prime = sorted(G_prime.nodes())
    
    emb_matrix_G = np.array([embeddings_G[node] for node in nodes_G])
    emb_matrix_G_prime = np.array([embeddings_G_prime[node] for node in nodes_G_prime])
    
    print(f"  Dataset1 matrix: {emb_matrix_G.shape}")
    print(f"  Dataset5 matrix: {emb_matrix_G_prime.shape}")
    
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
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nDirect:     F1={f1_direct*100:.2f}%")
    print(f"Normalized: F1={f1_norm*100:.2f}%")




    # Add this after the "Normalized" evaluation and before "FINAL RESULTS"

    # Evaluate Procrustes method (SUPERVISED - uses ground truth)
    print("\n[Method 3: Procrustes Alignment (SUPERVISED - uses ground truth)]")
    emb_G_aligned, anchor_count = procrustes_alignment_supervised(
        embeddings_G, embeddings_G_prime, nodes_G, nodes_G_prime, G, G_prime, threshold=0.1
    )
    
    if emb_G_aligned is not None:
        emb_matrix_G_prime = np.array([embeddings_G_prime[node] for node in nodes_G_prime])
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
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nDirect (Unsupervised):     F1={f1_direct*100:.2f}%")
    print(f"Normalized (Unsupervised): F1={f1_norm*100:.2f}%")
    print(f"Procrustes (SUPERVISED):   F1={f1_procrustes*100:.2f}% (used {anchor_count} ground truth anchors)")
    print("\n⚠️ Note: Procrustes uses ground truth for alignment (supervised)")