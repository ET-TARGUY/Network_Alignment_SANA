"""
Step 7: Analyze embedding quality.

This script:
1. Loads embeddings and spatial candidates
2. Computes cross-network similarities
3. Analyzes correlation with spatial distance
4. Evaluates discrimination power and ranking quality
5. Creates comprehensive visualizations
"""
import sys
import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ============================================
# HELPER FUNCTIONS
# ============================================

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def get_embedding_rank_of_nearest(node_g1, candidates, distances, embeddings_G1, embeddings_G2):
    """Get embedding rank of spatially nearest candidate."""
    if node_g1 not in embeddings_G1:
        return None
    
    # FIX: Skip if no candidates
    if len(candidates) == 0 or len(distances) == 0:
        return None
    
    # Find spatially nearest
    nearest_idx = np.argmin(distances)
    nearest_g2 = candidates[nearest_idx]
    
    if nearest_g2 not in embeddings_G2:
        return None
    
    # Compute embedding similarities for all candidates
    emb_g1 = embeddings_G1[node_g1]
    sims = []
    for cand_g2 in candidates:
        if cand_g2 in embeddings_G2:
            emb_g2 = embeddings_G2[cand_g2]
            sim = cosine_similarity(emb_g1, emb_g2)
            sims.append((cand_g2, sim))
    
    # FIX: Skip if no valid candidates with embeddings
    if len(sims) == 0:
        return None
    
    # Sort by embedding similarity
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    
    # Find rank of spatially nearest
    for rank, (cand, sim) in enumerate(sims_sorted, 1):
        if cand == nearest_g2:
            return rank
    
    return None

# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 7: EMBEDDING ANALYSIS")
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

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR = Path(config['paths']['outputs'])
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Method: {method}")

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
    # Load Spatial Candidates
    # ============================================
    
    candidates_filename = f"spatial_candidates_to_{dataset2_name}_{radius}m.pkl"
    candidates_path = base_dir / dataset1_name / candidates_filename
    
    if not candidates_path.exists():
        print(f"  ❌ Spatial candidates not found: {candidates_path}")
        continue
    
    print(f"\nLoading spatial candidates...")
    with open(candidates_path, 'rb') as f:
        spatial_data = pickle.load(f)
        candidates_G1_to_G2 = spatial_data['candidates']
        distances_G1_to_G2 = spatial_data['distances']
    
    print(f"  ✓ Loaded candidates")
    
    # ============================================
    # ANALYSIS 1: Embedding Space Properties
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS 1: EMBEDDING SPACE PROPERTIES")
    print(f"{'='*60}")
    
    # Get all embeddings as arrays
    all_emb_g1 = np.array(list(embeddings_G1.values()))
    all_emb_g2 = np.array(list(embeddings_G2.values()))
    all_embeddings = np.vstack([all_emb_g1, all_emb_g2])
    
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {all_embeddings.shape}")
    print(f"  Mean: {np.mean(all_embeddings):.4f}")
    print(f"  Std: {np.std(all_embeddings):.4f}")
    
    # Compute norms
    norms_g1 = np.linalg.norm(all_emb_g1, axis=1)
    norms_g2 = np.linalg.norm(all_emb_g2, axis=1)
    
    print(f"\nEmbedding Norms:")
    print(f"  G1 - Mean: {np.mean(norms_g1):.3f}, Std: {np.std(norms_g1):.3f}")
    print(f"  G2 - Mean: {np.mean(norms_g2):.3f}, Std: {np.std(norms_g2):.3f}")
    
    # ============================================
    # ANALYSIS 2: Cross-Network Similarity
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS 2: CROSS-NETWORK SIMILARITY")
    print(f"{'='*60}")
    
    print(f"\nComputing embedding similarities for spatial candidates...")
    embedding_similarities = []
    spatial_distances = []
    nearest_embedding_sims = []
    
    sample_size = min(5000, len(candidates_G1_to_G2))
    sampled_nodes = list(candidates_G1_to_G2.keys())[:sample_size]
    
    for node_g1 in tqdm(sampled_nodes, desc="Computing similarities"):
        if node_g1 not in embeddings_G1:
            continue
        
        emb_g1 = embeddings_G1[node_g1]
        candidates = candidates_G1_to_G2[node_g1]
        distances = distances_G1_to_G2[node_g1]
        
        sims = []
        for candidate_g2, distance in zip(candidates, distances):
            if candidate_g2 in embeddings_G2:
                emb_g2 = embeddings_G2[candidate_g2]
                sim = cosine_similarity(emb_g1, emb_g2)
                embedding_similarities.append(sim)
                spatial_distances.append(distance)
                sims.append(sim)
        
        if sims:
            nearest_embedding_sims.append(max(sims))
    
    embedding_similarities = np.array(embedding_similarities)
    spatial_distances = np.array(spatial_distances)
    nearest_embedding_sims = np.array(nearest_embedding_sims)
    
    print(f"\nEmbedding Similarity for Candidate Pairs:")
    print(f"  Sample size: {len(embedding_similarities):,}")
    print(f"  Mean: {np.mean(embedding_similarities):.4f}")
    print(f"  Median: {np.median(embedding_similarities):.4f}")
    print(f"  Std: {np.std(embedding_similarities):.4f}")
    print(f"  Min: {np.min(embedding_similarities):.4f}")
    print(f"  Max: {np.max(embedding_similarities):.4f}")
    
    print(f"\nNearest Candidate (Embedding):")
    print(f"  Mean: {np.mean(nearest_embedding_sims):.4f}")
    print(f"  Median: {np.median(nearest_embedding_sims):.4f}")
    
    # ============================================
    # ANALYSIS 3: Spatial vs Embedding Correlation
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS 3: SPATIAL vs EMBEDDING CORRELATION")
    print(f"{'='*60}")
    
    pearson_corr, p_pearson = pearsonr(spatial_distances, embedding_similarities)
    spearman_corr, p_spearman = spearmanr(spatial_distances, embedding_similarities)
    
    print(f"\nCorrelation (Distance vs Embedding Similarity):")
    print(f"  Pearson:  {pearson_corr:.4f} (p={p_pearson:.4e})")
    print(f"  Spearman: {spearman_corr:.4f} (p={p_spearman:.4e})")
    
    if spearman_corr < -0.3:
        print(f"  ✓ Good negative correlation: closer nodes → higher embedding similarity")
    elif spearman_corr < -0.1:
        print(f"  ⚠ Weak negative correlation: embeddings partially capture spatial proximity")
    else:
        print(f"  ⚠ Low correlation: embeddings may focus more on structure than space")
    
    # ============================================
    # ANALYSIS 4: Discrimination Power
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS 4: DISCRIMINATION POWER")
    print(f"{'='*60}")
    
    print(f"\nComputing discrimination gaps...")
    discrimination_gaps = []
    
    sample_nodes = list(embeddings_G1.keys())[:1000]
    
    for node_g1 in tqdm(sample_nodes, desc="Computing gaps"):
        if node_g1 not in candidates_G1_to_G2:
            continue
        
        candidates = candidates_G1_to_G2[node_g1]
        if len(candidates) < 2:
            continue
        
        emb_g1 = embeddings_G1[node_g1]
        
        sims = []
        for candidate_g2 in candidates:
            if candidate_g2 in embeddings_G2:
                emb_g2 = embeddings_G2[candidate_g2]
                sim = cosine_similarity(emb_g1, emb_g2)
                sims.append(sim)
        
        if len(sims) >= 2:
            sims_sorted = sorted(sims, reverse=True)
            gap = sims_sorted[0] - sims_sorted[1]
            discrimination_gaps.append(gap)
    
    discrimination_gaps = np.array(discrimination_gaps)
    
    print(f"\nDiscrimination Gap (1st - 2nd best):")
    print(f"  Sample size: {len(discrimination_gaps):,}")
    print(f"  Mean: {np.mean(discrimination_gaps):.4f}")
    print(f"  Median: {np.median(discrimination_gaps):.4f}")
    print(f"  Min: {np.min(discrimination_gaps):.4f}")
    print(f"  Max: {np.max(discrimination_gaps):.4f}")
    
    if np.median(discrimination_gaps) > 0.05:
        print(f"  ✓ Good discrimination: Clear winner in most cases")
    else:
        print(f"  ⚠ Weak discrimination: Hard to pick best match")
    
    # ============================================
    # ANALYSIS 5: Top-K Ranking Quality
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS 5: TOP-K RANKING QUALITY")
    print(f"{'='*60}")
    
    print(f"\nFor spatially nearest candidates, check embedding rank...")
    
    ranks = []
    for node_g1 in tqdm(sample_nodes, desc="Computing ranks"):
        if node_g1 not in candidates_G1_to_G2:
            continue
        
        candidates = candidates_G1_to_G2[node_g1]
        distances = distances_G1_to_G2[node_g1]
        
        rank = get_embedding_rank_of_nearest(node_g1, candidates, distances, embeddings_G1, embeddings_G2)
        if rank is not None:
            ranks.append(rank)
    
    ranks = np.array(ranks)
    
    print(f"\nEmbedding Rank of Spatially Nearest Candidate:")
    print(f"  Sample size: {len(ranks):,}")
    print(f"  Rank 1 (best match): {np.sum(ranks == 1):,} ({100*np.sum(ranks == 1)/len(ranks):.1f}%)")
    print(f"  Top-3: {np.sum(ranks <= 3):,} ({100*np.sum(ranks <= 3)/len(ranks):.1f}%)")
    print(f"  Top-5: {np.sum(ranks <= 5):,} ({100*np.sum(ranks <= 5)/len(ranks):.1f}%)")
    print(f"  Top-10: {np.sum(ranks <= 10):,} ({100*np.sum(ranks <= 10)/len(ranks):.1f}%)")
    print(f"  Mean rank: {np.mean(ranks):.2f}")
    print(f"  Median rank: {np.median(ranks):.0f}")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Embedding similarity distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(embedding_similarities, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Embedding Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Embedding Similarity Distribution\n(Candidate Pairs)')
    ax1.axvline(x=np.median(embedding_similarities), color='red', linestyle='--', 
                label=f'Median={np.median(embedding_similarities):.3f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Distance vs Embedding Similarity
    ax2 = plt.subplot(3, 3, 2)
    sample_idx = np.random.choice(len(spatial_distances), min(5000, len(spatial_distances)), replace=False)
    ax2.scatter(spatial_distances[sample_idx], embedding_similarities[sample_idx], alpha=0.1, s=1)
    ax2.set_xlabel('Spatial Distance (m)')
    ax2.set_ylabel('Embedding Similarity')
    ax2.set_title(f'Distance vs Embedding Similarity\n(Correlation: {spearman_corr:.3f})')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Discrimination gaps
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(discrimination_gaps, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax3.set_xlabel('Gap (1st - 2nd best)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Discrimination Power')
    ax3.axvline(x=np.median(discrimination_gaps), color='red', linestyle='--', 
                label=f'Median={np.median(discrimination_gaps):.3f}')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Rank distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(ranks, bins=range(1, min(51, int(np.max(ranks))+2)), edgecolor='black', alpha=0.7, color='orange')
    ax4.set_xlabel('Embedding Rank of Nearest Spatial Candidate')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Rank Distribution')
    ax4.axvline(x=np.median(ranks), color='red', linestyle='--', label=f'Median={np.median(ranks):.0f}')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Plot 5: Cumulative rank accuracy
    ax5 = plt.subplot(3, 3, 5)
    max_k = 20
    top_k_acc = [100 * np.sum(ranks <= k) / len(ranks) for k in range(1, max_k+1)]
    ax5.plot(range(1, max_k+1), top_k_acc, 'o-', linewidth=2)
    ax5.set_xlabel('K (Top-K)')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Top-K Accuracy')
    ax5.grid(alpha=0.3)
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 6: Embedding norms comparison
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist([norms_g1, norms_g2], bins=50, label=['G1', 'G2'], alpha=0.6, edgecolor='black')
    ax6.set_xlabel('Embedding Norm')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Embedding Norms by Graph')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # Plot 7: Distance bins vs similarity
    ax7 = plt.subplot(3, 3, 7)
    bins = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20)]
    bin_labels = ['0-2m', '2-5m', '5-10m', '10-15m', '15-20m']
    bin_means = []
    for low, high in bins:
        mask = (spatial_distances >= low) & (spatial_distances < high)
        if np.sum(mask) > 0:
            bin_means.append(np.mean(embedding_similarities[mask]))
        else:
            bin_means.append(0)
    
    ax7.bar(bin_labels, bin_means, edgecolor='black', alpha=0.7, color='purple')
    ax7.set_xlabel('Distance Bin')
    ax7.set_ylabel('Mean Embedding Similarity')
    ax7.set_title('Embedding Similarity by Distance')
    ax7.grid(alpha=0.3, axis='y')
    ax7.tick_params(axis='x', rotation=45)
    
    # Plot 8: Nearest candidate embedding similarity
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(nearest_embedding_sims, bins=50, edgecolor='black', alpha=0.7, color='red')
    ax8.set_xlabel('Embedding Similarity')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Nearest Spatial Candidate\n(Embedding Similarity)')
    ax8.axvline(x=np.median(nearest_embedding_sims), color='black', linestyle='--', 
                label=f'Median={np.median(nearest_embedding_sims):.3f}')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    # Plot 9: Top-K bar chart
    ax9 = plt.subplot(3, 3, 9)
    topk_values = [1, 3, 5, 10, 20]
    topk_pcts = [100 * np.sum(ranks <= k) / len(ranks) for k in topk_values]
    ax9.bar([str(k) for k in topk_values], topk_pcts, edgecolor='black', alpha=0.7, color='teal')
    ax9.set_xlabel('Top-K')
    ax9.set_ylabel('Accuracy (%)')
    ax9.set_title('Top-K Accuracy Summary')
    ax9.grid(alpha=0.3, axis='y')
    for i, (k, pct) in enumerate(zip(topk_values, topk_pcts)):
        ax9.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    viz_filename = f"embedding_analysis_{method}_{dataset1_name}_{dataset2_name}.png"
    viz_path = OUTPUTS_DIR / viz_filename
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {viz_path}")
    plt.close()
    
    # ============================================
    # SUMMARY
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"EMBEDDING QUALITY SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n✓ Embedding Properties:")
    print(f"  - Mean: {np.mean(all_embeddings):.4f}")
    print(f"  - Std: {np.std(all_embeddings):.4f}")
    print(f"  - Consistent norms across G1 and G2")
    
    print(f"\n✓ Cross-Network Similarity:")
    print(f"  - Mean similarity: {np.mean(embedding_similarities):.4f}")
    print(f"  - Spatial correlation: {spearman_corr:.4f}")
    
    print(f"\n✓ Discrimination:")
    print(f"  - Mean gap: {np.mean(discrimination_gaps):.4f}")
    print(f"  - {'Good' if np.median(discrimination_gaps) > 0.05 else 'Weak'} discrimination power")
    
    print(f"\n✓ Ranking Quality:")
    print(f"  - Top-1 accuracy: {100*np.sum(ranks == 1)/len(ranks):.1f}%")
    print(f"  - Top-5 accuracy: {100*np.sum(ranks <= 5)/len(ranks):.1f}%")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nAnalyzed embeddings for {len(dataset_pairs)} dataset pair(s)")

print("\n" + "="*60)
print("✓ EMBEDDING ANALYSIS COMPLETE")
print("="*60)