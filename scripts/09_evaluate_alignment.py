"""
Step 9: Evaluate alignment against ground truth.

This script:
1. Generates ground truth anchors (spatial tolerance)
2. Compares predictions vs ground truth
3. Computes Precision, Recall, F1-score
4. Analyzes errors
5. Exports error analysis layers to GeoPackage
"""
import sys
import os
import pickle
import yaml
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm
from shapely.geometry import LineString, Point

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 9: EVALUATE ALIGNMENT")
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
sigma = config['compound_graph']['gaussian']['sigma']


# Evaluation parameters
ANCHOR_TOLERANCE = 0.1  # meters (nodes at nearly same position)

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR = Path(config['paths']['outputs'])
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Method: {method}")
print(f"  Anchor tolerance: {ANCHOR_TOLERANCE}m")
print(f"  Radius: {radius}m")
print(f"  sigma: {sigma}m")
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
    # Load Alignment Results
    # ============================================
    
    alignment_filename = f"alignment_hungarian_{method}_{dataset2_name}.pkl"
    alignment_path = base_dir / dataset1_name / alignment_filename
    
    if not alignment_path.exists():
        print(f"  ❌ Alignment not found: {alignment_path}")
        print(f"  Run script 08_hungarian_alignment.py first!")
        continue
    
    print(f"\nLoading alignment results...")
    with open(alignment_path, 'rb') as f:
        alignment_data = pickle.load(f)
        predicted_matches = alignment_data['matches']
    
    print(f"  ✓ Loaded {len(predicted_matches):,} predicted matches")
    
    # ============================================
    # Generate Ground Truth Anchors
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"GENERATING GROUND TRUTH ANCHORS")
    print(f"{'='*60}")
    
    print(f"Spatial tolerance: {ANCHOR_TOLERANCE}m")
    print(f"Building spatial index...")
    
    # Get all node coordinates
    coords_G1 = {node: (Graph_1.nodes[node]['x'], Graph_1.nodes[node]['y']) 
                 for node in Graph_1.nodes()}
    coords_G2 = {node: (Graph_2.nodes[node]['x'], Graph_2.nodes[node]['y']) 
                 for node in Graph_2.nodes()}
    
    # Build KDTree for G2
    nodes_G2_list = list(coords_G2.keys())
    coords_G2_array = np.array([coords_G2[n] for n in nodes_G2_list])
    tree_G2 = KDTree(coords_G2_array)
    
    # Find ground truth anchors
    print(f"Finding ground truth anchors...")
    
    ground_truth_anchors = {}
    
    for node_g1, coord_g1 in tqdm(coords_G1.items(), desc="Finding anchors"):
        # Find G2 nodes within tolerance
        indices = tree_G2.query_ball_point(coord_g1, ANCHOR_TOLERANCE)
        
        if len(indices) == 1:
            # Unique match within tolerance
            node_g2 = nodes_G2_list[indices[0]]
            distance = np.linalg.norm(np.array(coord_g1) - np.array(coords_G2[node_g2]))
            ground_truth_anchors[node_g1] = {
                'g2_id': node_g2,
                'distance': distance
            }
    
    print(f"\n✓ Found {len(ground_truth_anchors):,} ground truth anchors")
    print(f"  ({100*len(ground_truth_anchors)/len(coords_G1):.1f}% of G1 nodes)")
    
    # ============================================
    # Evaluate Alignment
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"ALIGNMENT EVALUATION")
    print(f"{'='*60}")
    
    # Convert predicted matches to dict for easy lookup
    predicted_dict = {m['g1_id']: m['g2_id'] for m in predicted_matches}
    
    # True Positives: predicted matches that match ground truth
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Lists for detailed analysis
    correct_matches = []
    incorrect_matches = []
    missed_anchors = []
    
    for node_g1, gt_match in tqdm(ground_truth_anchors.items(), desc="Evaluating"):
        gt_g2 = gt_match['g2_id']
        
        if node_g1 in predicted_dict:
            pred_g2 = predicted_dict[node_g1]
            
            if pred_g2 == gt_g2:
                true_positives += 1
                correct_matches.append(node_g1)
            else:
                false_positives += 1
                # Get distance of incorrect match
                coord_g1 = coords_G1[node_g1]
                coord_pred_g2 = coords_G2[pred_g2]
                error_dist = np.linalg.norm(np.array(coord_g1) - np.array(coord_pred_g2))
                incorrect_matches.append({
                    'g1_id': node_g1,
                    'predicted_g2': pred_g2,
                    'true_g2': gt_g2,
                    'error_distance': error_dist
                })
        else:
            false_negatives += 1
            missed_anchors.append(node_g1)
    
    # Matches that are not in ground truth (predicted but no anchor)
    extra_predictions = sum(1 for g1 in predicted_dict.keys() 
                           if g1 not in ground_truth_anchors)
    
    # ============================================
    # Compute Metrics
    # ============================================
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(ground_truth_anchors) if len(ground_truth_anchors) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS")
    print(f"{'='*60}")
    
    print(f"\nGround Truth:")
    print(f"  Total anchors: {len(ground_truth_anchors):,}")
    
    print(f"\nPredictions:")
    print(f"  Total predicted matches: {len(predicted_matches):,}")
    print(f"  Matches with ground truth: {len([g1 for g1 in predicted_dict if g1 in ground_truth_anchors]):,}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {true_positives:,}")
    print(f"  False Positives (FP): {false_positives:,}")
    print(f"  False Negatives (FN): {false_negatives:,}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision:.4f} ({100*precision:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({100*recall:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f} ({100*f1_score:.2f}%)")
    
    print(f"\nAdditional Info:")
    print(f"  Correct matches: {true_positives:,}")
    print(f"  Incorrect matches: {false_positives:,}")
    print(f"  Missed anchors: {false_negatives:,}")
    print(f"  Extra predictions (no ground truth): {extra_predictions:,}")
    
    # ============================================
    # Error Analysis
    # ============================================
    
    if incorrect_matches:
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*60}")
        
        error_distances = [m['error_distance'] for m in incorrect_matches]
        
        print(f"\nIncorrect Match Distances:")
        print(f"  Mean error: {np.mean(error_distances):.2f}m")
        print(f"  Median error: {np.median(error_distances):.2f}m")
        print(f"  Max error: {np.max(error_distances):.2f}m")
        
        # Distance bins
        bins = [(0, 5), (5, 10), (10, 20), (20, float('inf'))]
        print(f"\nError Distance Distribution:")
        for low, high in bins:
            count = sum(1 for d in error_distances if low <= d < high)
            if count > 0:
                pct = 100 * count / len(error_distances)
                print(f"  {low}-{high if high != float('inf') else '+'}m: {count} errors ({pct:.1f}%)")
    
    # ============================================
    # Visualization
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Metrics bar chart
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1_score]
    colors = ['green' if v > 0.9 else 'orange' if v > 0.7 else 'red' for v in values]
    
    axes[0].bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Alignment Quality Metrics')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    axes[0].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # Plot 2: Confusion matrix
    confusion_data = [true_positives, false_positives, false_negatives]
    confusion_labels = [f'TP\n{true_positives:,}', f'FP\n{false_positives:,}', f'FN\n{false_negatives:,}']
    confusion_colors = ['green', 'red', 'orange']
    
    axes[1].bar(range(3), confusion_data, color=confusion_colors, edgecolor='black', alpha=0.7)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(confusion_labels)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confusion Matrix')
    axes[1].grid(alpha=0.3, axis='y')
    
    # Plot 3: Error distance distribution (if errors exist)
    if incorrect_matches:
        axes[2].hist(error_distances, bins=30, edgecolor='black', alpha=0.7, color='red')
        axes[2].set_xlabel('Error Distance (m)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Incorrect Match Error Distances')
        axes[2].axvline(np.median(error_distances), color='blue', linestyle='--', 
                       label=f'Median={np.median(error_distances):.2f}m')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No incorrect matches!', 
                    ha='center', va='center', fontsize=20, color='green')
        axes[2].set_title('Error Analysis')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_filename = f"alignment_evaluation_{method}_{dataset1_name}_{dataset2_name}.png"
    viz_path = OUTPUTS_DIR / viz_filename
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {viz_path}")
    plt.close()
    
    # ============================================
    # Save Evaluation Results
    # ============================================
    
    print(f"\nSaving evaluation results...")
    
    output_dir = base_dir / dataset1_name
    eval_filename = f"alignment_evaluation_{method}_{dataset2_name}.pkl"
    eval_path = output_dir / eval_filename
    
    evaluation_results = {
        'ground_truth_anchors': len(ground_truth_anchors),
        'predicted_matches': len(predicted_matches),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'correct_matches': correct_matches,
        'incorrect_matches': incorrect_matches,
        'missed_anchors': missed_anchors,
        'ground_truth_dict': ground_truth_anchors
    }
    
    with open(eval_path, 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print(f"✓ Saved: {eval_path}")
    

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nEvaluated alignment for {len(dataset_pairs)} dataset pair(s)")

print("\n" + "="*60)
print("✓ EVALUATION COMPLETE")
print("="*60)