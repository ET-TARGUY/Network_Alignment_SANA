#!/usr/bin/env python3
"""
Struc2vec baseline for network alignment.
Tests pure structural identity without spatial filtering.
"""

import sys
sys.path.append('src/struc2vec_lib')

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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    return config


def load_graphs_from_edgelist(config):
    import csv

    raw_dir  = Path(config['paths']['data_raw'])
    zone     = config.get('zone', 'full')
    dataset_pair = config['dataset_pairs'][0]

    base_dir = raw_dir / "Prades" if zone == "Prades" else raw_dir

    edgelist1 = base_dir / dataset_pair[0] / f"{dataset_pair[0].lower()}_prades.edgelist"
    edgelist2 = base_dir / dataset_pair[1] / f"{dataset_pair[1].lower()}_prades.edgelist"
    coords1   = base_dir / dataset_pair[0] / f"{dataset_pair[0].lower()}_prades_coords.csv"
    coords2   = base_dir / dataset_pair[1] / f"{dataset_pair[1].lower()}_prades_coords.csv"

    print(f"Loading {dataset_pair[0]} from {edgelist1}")
    G = nx.read_edgelist(str(edgelist1))

    print(f"Loading {dataset_pair[1]} from {edgelist2}")
    G_prime = nx.read_edgelist(str(edgelist2))

    print(f"Loading coordinates from CSV files...")

    with open(coords1, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['node_id']
            if node_id in G.nodes():
                G.nodes[node_id]['x'] = float(row['x'])
                G.nodes[node_id]['y'] = float(row['y'])

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


def run_struc2vec(G, output_name, config):
    out_dir = Path(config['paths']['outputs'])
    out_dir.mkdir(parents=True, exist_ok=True)

    input_edgelist = out_dir / f"{output_name}_input.edgelist"
    output_emb     = out_dir / f"{output_name}_output.emb"

    print(f"  Saving edgelist: {input_edgelist}")
    nx.write_edgelist(G, str(input_edgelist), data=False)

    struc2vec_params = config['struc2vec']
    python_executable = sys.executable

    cmd = [
        python_executable,
        "src/struc2vec_lib/main.py",
        "--input",       str(input_edgelist),
        "--output",      str(output_emb),
        "--num-walks",   str(struc2vec_params['num_walks']),
        "--walk-length", str(struc2vec_params['walk_length']),
        "--dimensions",  str(struc2vec_params['dimensions']),
        "--window-size", str(struc2vec_params['window_size']),
        "--workers",     str(struc2vec_params['workers']),
        "--OPT1",        str(struc2vec_params['OPT1']),
        "--OPT2",        str(struc2vec_params['OPT2']),
        "--OPT3",        str(struc2vec_params['OPT3']),
        "--until-layer", str(struc2vec_params['until_layer'])
    ]

    print(f"  Running struc2vec...")
    print(f"    Using Python: {python_executable}")
    print(f"    Command: {' '.join(cmd)}")

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

    print(f"  Reading embeddings from: {output_emb}")

    if not output_emb.exists():
        print(f"  ❌ Output file not found: {output_emb}")
        return None

    embeddings = read_struc2vec_embeddings(output_emb)
    print(f"  ✓ Loaded {len(embeddings)} node embeddings")

    return embeddings


def read_struc2vec_embeddings(emb_file):
    embeddings = {}
    with open(emb_file, 'r') as f:
        header = f.readline().strip().split()
        num_nodes  = int(header[0])
        dimensions = int(header[1])
        for line in f:
            parts = line.strip().split()
            node_id = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings[node_id] = embedding
    print(f"    Dimensions: {dimensions}, Nodes: {len(embeddings)}")
    return embeddings


def hungarian_alignment(similarity_matrix):
    print("\nApplying Hungarian algorithm...")
    cost_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(f"  ✓ Found {len(row_ind)} alignments")
    return row_ind, col_ind


def evaluate_alignment(row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime):
    print("\nEvaluating alignment...")

    alignment = {}
    for i, j in zip(row_ind, col_ind):
        alignment[nodes_G[i]] = nodes_G_prime[j]

    print("  Computing ground truth (distance < 0.1m)...")
    ground_truth = set()
    for node_u in nodes_G:
        pos_u = np.array([G.nodes[node_u]['x'], G.nodes[node_u]['y']])
        for node_v in nodes_G_prime:
            pos_v = np.array([G_prime.nodes[node_v]['x'], G_prime.nodes[node_v]['y']])
            if np.linalg.norm(pos_u - pos_v) < 0.1:
                ground_truth.add((node_u, node_v))

    print(f"  Ground truth pairs: {len(ground_truth)}")

    true_positives = sum(1 for u, v in alignment.items() if (u, v) in ground_truth)
    precision = true_positives / len(alignment) if alignment else 0
    recall    = true_positives / len(ground_truth) if ground_truth else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Results:")
    print(f"    Precision: {precision*100:.2f}%")
    print(f"    Recall:    {recall*100:.2f}%")
    print(f"    F1-Score:  {f1*100:.2f}%")
    print(f"    True Positives: {true_positives}/{len(alignment)}")

    return precision, recall, f1


def procrustes_alignment_supervised(emb_G, emb_G_prime, nodes_G, nodes_G_prime, G, G_prime, threshold=0.1):
    from scipy.linalg import orthogonal_procrustes
    from scipy.spatial import cKDTree

    print(f"\n  Finding ground truth anchors (threshold={threshold}m)...")

    coords_G       = np.array([[G.nodes[n]['x'],       G.nodes[n]['y']]       for n in nodes_G])
    coords_G_prime = np.array([[G_prime.nodes[n]['x'], G_prime.nodes[n]['y']] for n in nodes_G_prime])

    tree = cKDTree(coords_G_prime)
    distances, indices = tree.query(coords_G, distance_upper_bound=threshold)

    anchor_pairs = [(i, j) for i, (d, j) in enumerate(zip(distances, indices)) if d < threshold]
    print(f"  Found {len(anchor_pairs)} ground truth anchor pairs")

    if len(anchor_pairs) < 10:
        print("  ⚠️ Too few anchors for reliable Procrustes alignment!")
        return None, 0

    anchor_i = [p[0] for p in anchor_pairs]
    anchor_j = [p[1] for p in anchor_pairs]

    emb_matrix_G       = np.array([emb_G[nodes_G[i]]             for i in range(len(nodes_G))])
    emb_matrix_G_prime = np.array([emb_G_prime[nodes_G_prime[j]] for j in range(len(nodes_G_prime))])

    R, _ = orthogonal_procrustes(emb_matrix_G[anchor_i], emb_matrix_G_prime[anchor_j])
    emb_G_aligned = emb_matrix_G @ R

    print(f"  Learning Procrustes transformation...")
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

    print("\nPreparing embedding matrices...")
    nodes_G       = sorted(G.nodes())
    nodes_G_prime = sorted(G_prime.nodes())

    emb_matrix_G       = np.array([embeddings_G[node]       for node in nodes_G])
    emb_matrix_G_prime = np.array([embeddings_G_prime[node] for node in nodes_G_prime])

    print(f"  {dataset_pair[0]} matrix: {emb_matrix_G.shape}")
    print(f"  {dataset_pair[1]} matrix: {emb_matrix_G_prime.shape}")

    print("\n[1/2] Direct cosine similarity...")
    similarity_direct = cosine_similarity(emb_matrix_G, emb_matrix_G_prime)
    print(f"  Similarity matrix: {similarity_direct.shape}")
    print(f"  Mean similarity: {similarity_direct.mean():.4f}")

    print("\n[2/2] Normalized cosine similarity...")
    similarity_normalized = cosine_similarity(normalize(emb_matrix_G, axis=1),
                                               normalize(emb_matrix_G_prime, axis=1))
    print(f"  Similarity matrix: {similarity_normalized.shape}")
    print(f"  Mean similarity: {similarity_normalized.mean():.4f}")

    print("\n✓ Similarities computed!")

    print("\n" + "="*60)
    print("ALIGNMENT AND EVALUATION")
    print("="*60)

    print("\n[Method 1: Direct Comparison]")
    row_ind, col_ind = hungarian_alignment(similarity_direct)
    _, _, f1_direct = evaluate_alignment(row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime)

    print("\n[Method 2: Normalized Comparison]")
    row_ind, col_ind = hungarian_alignment(similarity_normalized)
    _, _, f1_norm = evaluate_alignment(row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nDirect:     F1={f1_direct*100:.2f}%")
    print(f"Normalized: F1={f1_norm*100:.2f}%")

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
        _, _, f1_procrustes = evaluate_alignment(row_ind, col_ind, nodes_G, nodes_G_prime, G, G_prime)
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