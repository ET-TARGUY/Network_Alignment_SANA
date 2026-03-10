"""
Step 8: Final alignment using Hungarian algorithm.

This script:
1. Loads embeddings and spatial candidates
2. Computes combined similarity (embedding + spatial)
3. Applies Hungarian algorithm for optimal matching
4. Saves alignment results and exports to GeoPackage
"""
import sys
import os
import pickle
import yaml
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
from shapely.geometry import LineString

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ============================================
# HELPER FUNCTIONS
# ============================================

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def compute_spatial_similarity(distance, sigma=5):
    """Compute Gaussian spatial similarity."""
    return np.exp(-(distance**2) / (2 * sigma**2))


def compute_combined_similarity(embedding_sim, spatial_sim, alpha=0.7):
    """
    Compute combined similarity.
    
    combined = alpha * embedding_sim + (1 - alpha) * spatial_sim
    
    Args:
        embedding_sim: embedding similarity [0, 1]
        spatial_sim: spatial similarity [0, 1]
        alpha: weight for embedding (beta = 1 - alpha for spatial)
    """
    return alpha * embedding_sim + (1 - alpha) * spatial_sim


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 8: HUNGARIAN ALIGNMENT")
print("="*60)

# Load config
config_path = Path("config/default.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get parameters
dataset_pairs = config['dataset_pairs']
zone = config['zone']
radius = config['spatial']['radius']
method = config['compound_graph']['method']

# Get alignment parameters from config
ALPHA = config['alignment']['alpha']
BETA = config['alignment']['beta']
SIGMA = config['alignment']['sigma']
MIN_SIMILARITY = config['alignment']['min_similarity']

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR = Path(config['paths']['outputs'])
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Method: {method}")
print(f"  Alpha (embedding weight): {ALPHA}")
print(f"  Beta (spatial weight): {BETA}")
print(f"  Sigma (Gaussian): {SIGMA}")
print(f"  Min similarity threshold: {MIN_SIMILARITY}")

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
    
    if not graph1_path.exists() or not graph2_path.exists():
        print(f"  ❌ Graphs not found")
        continue
    
    print(f"\nLoading graphs...")
    with open(graph1_path, 'rb') as f:
        Graph_1 = pickle.load(f)
    with open(graph2_path, 'rb') as f:
        Graph_2 = pickle.load(f)
    
    print(f"  ✓ {dataset1_name}: {Graph_1.number_of_nodes():,} nodes")
    print(f"  ✓ {dataset2_name}: {Graph_2.number_of_nodes():,} nodes")
    
    # ============================================
    # Load Embeddings
    # ============================================
    
    embeddings_filename = f"embeddings_{method}_{dataset2_name}.pkl"
    embeddings_path = base_dir / dataset1_name / embeddings_filename
    
    if not embeddings_path.exists():
        print(f"  ❌ Embeddings not found: {embeddings_path}")
        continue
    
    print(f"\nLoading embeddings...")
    with open(embeddings_path, 'rb') as f:
        emb_data = pickle.load(f)
        embeddings_G1 = emb_data['embeddings_G1']
        embeddings_G2 = emb_data['embeddings_G2']
    
    print(f"  ✓ G1 embeddings: {len(embeddings_G1):,}")
    print(f"  ✓ G2 embeddings: {len(embeddings_G2):,}")
    
    # ============================================
    # Load Candidates (Method-Dependent)
    # ============================================

    if method == "SANA":
        # Gaussian: Use spatial candidates (filtered by radius)
        candidates_filename = f"spatial_candidates_to_{dataset2_name}_{radius}m.pkl"
        candidates_path = base_dir / dataset1_name / candidates_filename
        
        if not candidates_path.exists():
            print(f"  ❌ Spatial candidates not found: {candidates_path}")
            print(f"  Run script 02_spatial_candidates.py first!")
            continue
        
        print(f"\nLoading spatial candidates...")
        with open(candidates_path, 'rb') as f:
            spatial_data = pickle.load(f)
            candidates_G1_to_G2 = spatial_data['candidates']
            distances_G1_to_G2 = spatial_data['distances']
        
        print(f"  ✓ Loaded spatial candidates")

    elif method == "CENA":
        # CENA: Use ALL nodes (no spatial filtering)
        print(f"\nBuilding FULL candidate sets (CENA - no spatial filter)...")
        
        nodes_G1 = list(Graph_1.nodes())
        nodes_G2 = list(Graph_2.nodes())
        
        # Every G1 node has ALL G2 nodes as candidates
        candidates_G1_to_G2 = {
            node_g1: nodes_G2[:] for node_g1 in nodes_G1
        }
        
        # For distances: compute all pairwise distances
        print(f"  Computing pairwise distances for all pairs...")
        distances_G1_to_G2 = {}
        for node_g1 in nodes_G1:
            x1, y1 = Graph_1.nodes[node_g1]['x'], Graph_1.nodes[node_g1]['y']
            dists = []
            for node_g2 in nodes_G2:
                x2, y2 = Graph_2.nodes[node_g2]['x'], Graph_2.nodes[node_g2]['y']
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                dists.append(dist)
            distances_G1_to_G2[node_g1] = dists
        
        total_pairs = len(nodes_G1) * len(nodes_G2)
        print(f"  ✓ Built FULL candidates: {total_pairs:,} pairs")

    else:
        print(f"  ❌ Unknown method: {method}")
        continue
    
    # ============================================
    # Compute Combined Similarities
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"COMPUTING COMBINED SIMILARITIES")
    print(f"{'='*60}")
    
    print(f"\nComputing combined similarities for all candidate pairs...")
    
    # Store all similarities
    all_similarities = {}
    
    for node_g1 in tqdm(candidates_G1_to_G2.keys(), desc="Computing similarities"):
        if node_g1 not in embeddings_G1:
            continue
        
        emb_g1 = embeddings_G1[node_g1]
        candidates = candidates_G1_to_G2[node_g1]
        distances = distances_G1_to_G2[node_g1]
        
        for candidate_g2, distance in zip(candidates, distances):
            if candidate_g2 not in embeddings_G2:
                continue
            
            # Compute embedding similarity
            emb_g2 = embeddings_G2[candidate_g2]
            embedding_sim = cosine_similarity(emb_g1, emb_g2)
            
            # Compute spatial similarity
            spatial_sim = compute_spatial_similarity(distance, sigma=SIGMA)
            
            # Compute combined similarity
            combined_sim = compute_combined_similarity(embedding_sim, spatial_sim, alpha=ALPHA)
            
            # Store
            all_similarities[(node_g1, candidate_g2)] = {
                'embedding_sim': embedding_sim,
                'spatial_sim': spatial_sim,
                'combined_sim': combined_sim,
                'distance': distance
            }
    
    print(f"\n✓ Computed {len(all_similarities):,} similarity scores")
    
    # Filter by threshold
    filtered_similarities = {
        pair: sims for pair, sims in all_similarities.items()
        if sims['combined_sim'] >= MIN_SIMILARITY
    }
    
    print(f"✓ After threshold filter (≥{MIN_SIMILARITY}): {len(filtered_similarities):,} pairs")
    
    # ============================================
    # Build Cost Matrix for Hungarian Algorithm
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"HUNGARIAN ALGORITHM")
    print(f"{'='*60}")
    
    print(f"\nBuilding cost matrix...")
    
    # Get unique nodes
    nodes_g1 = sorted(set(pair[0] for pair in filtered_similarities.keys()))
    nodes_g2 = sorted(set(pair[1] for pair in filtered_similarities.keys()))
    
    print(f"  G1 nodes with candidates: {len(nodes_g1):,}")
    print(f"  G2 nodes as candidates: {len(nodes_g2):,}")
    
    # Create cost matrix (Hungarian minimizes, so use negative similarity)
    n1 = len(nodes_g1)
    n2 = len(nodes_g2)
    
    # Make square matrix by padding with large costs
    n = max(n1, n2)
    cost_matrix = np.full((n, n), 1e10)  # Large cost for non-existent pairs
    
    # Fill in actual costs (negative similarity for minimization)
    for i, node_g1 in enumerate(nodes_g1):
        for j, node_g2 in enumerate(nodes_g2):
            pair = (node_g1, node_g2)
            if pair in filtered_similarities:
                # Use negative similarity as cost (to maximize similarity)
                cost_matrix[i, j] = -filtered_similarities[pair]['combined_sim']
    
    print(f"\n✓ Cost matrix shape: {cost_matrix.shape}")
    
    # ============================================
    # Run Hungarian Algorithm
    # ============================================
    
    print(f"\nRunning Hungarian algorithm...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Extract valid matches (ignore padded rows/cols)
    hungarian_matches = []
    
    for i, j in zip(row_ind, col_ind):
        if i < n1 and j < n2:
            node_g1 = nodes_g1[i]
            node_g2 = nodes_g2[j]
            
            # Check if this is a real match (not padding)
            pair = (node_g1, node_g2)
            if pair in filtered_similarities:
                sims = filtered_similarities[pair]
                
                # Get coordinates
                x_g1 = Graph_1.nodes[node_g1]['x']
                y_g1 = Graph_1.nodes[node_g1]['y']
                x_g2 = Graph_2.nodes[node_g2]['x']
                y_g2 = Graph_2.nodes[node_g2]['y']
                
                hungarian_matches.append({
                    'g1_id': node_g1,
                    'g2_id': node_g2,
                    'x_g1': x_g1,
                    'y_g1': y_g1,
                    'x_g2': x_g2,
                    'y_g2': y_g2,
                    'distance_m': sims['distance'],
                    'embedding_sim': sims['embedding_sim'],
                    'spatial_sim': sims['spatial_sim'],
                    'combined_sim': sims['combined_sim']
                })
    
    print(f"\n✓ Hungarian algorithm complete")
    print(f"  Total matches: {len(hungarian_matches):,}")
    
    # ============================================
    # Statistics
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ALIGNMENT STATISTICS")
    print(f"{'='*60}")
    
    distances = [m['distance_m'] for m in hungarian_matches]
    combined_sims = [m['combined_sim'] for m in hungarian_matches]
    embedding_sims = [m['embedding_sim'] for m in hungarian_matches]
    spatial_sims = [m['spatial_sim'] for m in hungarian_matches]
    
    print(f"\nMatch Statistics:")
    print(f"  Total matches: {len(hungarian_matches):,}")
    print(f"  Coverage G1: {100*len(hungarian_matches)/Graph_1.number_of_nodes():.1f}%")
    print(f"  Coverage G2: {100*len(set(m['g2_id'] for m in hungarian_matches))/Graph_2.number_of_nodes():.1f}%")
    
    print(f"\nDistance Statistics:")
    print(f"  Mean: {np.mean(distances):.2f}m")
    print(f"  Median: {np.median(distances):.2f}m")
    print(f"  Max: {np.max(distances):.2f}m")
    
    print(f"\nSimilarity Statistics:")
    print(f"  Combined - Mean: {np.mean(combined_sims):.4f}, Median: {np.median(combined_sims):.4f}")
    print(f"  Embedding - Mean: {np.mean(embedding_sims):.4f}, Median: {np.median(embedding_sims):.4f}")
    print(f"  Spatial - Mean: {np.mean(spatial_sims):.4f}, Median: {np.median(spatial_sims):.4f}")
    
    # ============================================
    # Save to Pickle
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"SAVING ALIGNMENT RESULTS")
    print(f"{'='*60}")
    
    output_dir = base_dir / dataset1_name
    
    # Save to pickle
    alignment_filename = f"alignment_hungarian_{method}_{dataset2_name}.pkl"
    alignment_path = output_dir / alignment_filename
    
    with open(alignment_path, 'wb') as f:
        pickle.dump({
            'matches': hungarian_matches,
            'method': 'hungarian_algorithm',
            'parameters': {
                'min_similarity': MIN_SIMILARITY,
                'alpha': ALPHA,
                'beta': BETA,
                'sigma': SIGMA
            },
            'statistics': {
                'total_matches': len(hungarian_matches),
                'mean_combined_sim': np.mean(combined_sims),
                'mean_distance': np.mean(distances),
                'median_distance': np.median(distances)
            },
            'dataset1': dataset1_name,
            'dataset2': dataset2_name
        }, f)
    
    print(f"\n✓ Saved pickle: {alignment_path}")
    print(f"  Matches: {len(hungarian_matches):,}")
    
    # ============================================
    # Export to GeoPackage
    # ============================================
    
    print(f"\nExporting to GeoPackage...")
    
    # Create LineString features
    features = []
    for match in tqdm(hungarian_matches, desc="Creating geometries"):
        line = LineString([
            (match['x_g1'], match['y_g1']),
            (match['x_g2'], match['y_g2'])
        ])
        
        features.append({
            'g1_id': match['g1_id'],
            'g2_id': match['g2_id'],
            'distance_m': match['distance_m'],
            'embedding_sim': match['embedding_sim'],
            'spatial_sim': match['spatial_sim'],
            'combined_sim': match['combined_sim'],
            'geometry': line
        })
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features, crs='EPSG:2154')
    
    # Save to GeoPackage
    gpkg_filename = f"alignment_hungarian_{method}_{dataset1_name}_{dataset2_name}.gpkg"
    gpkg_path = OUTPUTS_DIR / gpkg_filename
    
    gdf.to_file(gpkg_path, layer='matches', driver='GPKG')
    
    print(f"✓ Saved GeoPackage: {gpkg_path}")
    print(f"  Layer: 'matches' ({len(gdf):,} features)")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nAligned {len(dataset_pairs)} dataset pair(s) using Hungarian algorithm")

print("\n" + "="*60)
print("✓ HUNGARIAN ALIGNMENT COMPLETE")
print("="*60)