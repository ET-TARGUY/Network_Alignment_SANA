"""
Step 5: Generate random walks on compound graph.

This script:
1. Loads configuration
2. Loads compound graph
3. Generates weighted random walks
4. Analyzes and visualizes walk statistics
5. Saves walks for embedding generation
"""
import sys
import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.graph.random_walks import (
    generate_random_walks,
    walks_to_sentences,
    analyze_walk_statistics,
    generate_random_walks_cena  # NEW
)




# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 5: GENERATE RANDOM WALKS")
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

num_walks = config['random_walks']['num_walks']
walk_length = config['random_walks']['walk_length']
seed = config['random_walks']['seed']

# Set random seed
np.random.seed(seed)

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR = Path(config['paths']['outputs'])
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Method: {method}")
print(f"  Num walks per node: {num_walks}")
print(f"  Walk length: {walk_length}")
print(f"  Random seed: {seed}")
print(f"  Radius: {radius}")


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
    # Load Compound Graph
    # ============================================
    
    # Determine compound graph filename based on method
    if method == "gaussian":
        sigma = config['compound_graph']['gaussian']['sigma']
        compound_filename = f"compound_graph_gaussian_{dataset2_name}_{radius}m_sigma{sigma}.pkl"
    elif method == "cena":
        K = config['structural']['K']
        compound_filename = f"compound_graph_cena_{dataset2_name}_{radius}m_K{K}.pkl"
    else:
        print(f"  ❌ Unknown method: {method}")
        continue
    
    compound_path = base_dir / dataset1_name / compound_filename
    
    if not compound_path.exists():
        print(f"  ❌ Compound graph not found: {compound_path}")
        print(f"  Run script 04_build_compound_graph.py first!")
        continue
    
    print(f"\nLoading compound graph...")
    with open(compound_path, 'rb') as f:
        data = pickle.load(f)
        
        # Debug: check what we loaded
        print(f"DEBUG: data type = {type(data)}")
        if isinstance(data, dict):
            print(f"DEBUG: data keys = {data.keys()}")
            G_compound = data['graph']
            
            # NEW DEBUG: Check what G_compound actually is
            print(f"DEBUG: G_compound type = {type(G_compound)}")
            print(f"DEBUG: G_compound = {G_compound}")
            
            # If it's a tuple, maybe the graph is the first element?
            if isinstance(G_compound, tuple):
                print(f"DEBUG: It's a tuple with {len(G_compound)} elements")
                print(f"DEBUG: First element type = {type(G_compound[0])}")
                G_compound = G_compound[0]  # Try extracting first element
        else:
            print(f"DEBUG: data is not a dict, treating as graph directly")
            G_compound = data

    print(f"  ✓ Loaded: {G_compound.number_of_nodes():,} nodes, {G_compound.number_of_edges():,} edges")
    
    # ============================================
    # Generate Random Walks
    # ============================================
    
    # ============================================
    # Generate Random Walks
    # ============================================
    if method == "cena":
        # CENA biased random walks
        q = config['random_walks']['cena']['q']
        
        walks, walk_stats = generate_random_walks_cena(
            G_compound,
            num_walks=num_walks,
            walk_length=walk_length,
            q=q,
            verbose=True
        )
    else:
        # Gaussian (standard weighted random walks)
        walks, walk_stats = generate_random_walks(
            G_compound,
            num_walks=num_walks,
            walk_length=walk_length,
            verbose=True
        )

    
    # Convert to string sentences
    print(f"\nConverting walks to string sentences...")
    walks_str = walks_to_sentences(walks)
    print(f"  ✓ Converted {len(walks_str):,} walks")
    
    # ============================================
    # Detailed Walk Analysis
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"DETAILED WALK STATISTICS")
    print(f"{'='*60}")
    
    detailed_stats = analyze_walk_statistics(walks_str)
    
    print(f"\nStep Distribution:")
    print(f"  Total steps: {detailed_stats['total_steps']:,}")
    print(f"  Within G1: {detailed_stats['within_G1_steps']:,} ({detailed_stats['within_G1_pct']:.1f}%)")
    print(f"  Within G2: {detailed_stats['within_G2_steps']:,} ({detailed_stats['within_G2_pct']:.1f}%)")
    print(f"  Cross-graph: {detailed_stats['cross_steps']:,} ({detailed_stats['cross_pct']:.1f}%)")
    
    # ============================================
    # Visualization
    # ============================================
    
    print(f"\nCreating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Walk length distribution
    walk_lengths = [len(w) for w in walks_str]
    axes[0].hist(walk_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Walk Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Walk Lengths')
    axes[0].axvline(x=walk_length, color='red', linestyle='--', label=f'Target: {walk_length}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Cross-graph transitions
    cross_transitions = []
    for walk in walks_str:
        transitions = sum(1 for i in range(len(walk)-1) 
                         if walk[i].split('_')[0] != walk[i+1].split('_')[0])
        cross_transitions.append(transitions)
    
    axes[1].hist(cross_transitions, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Number of Cross-Graph Transitions')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cross-Graph Transitions per Walk')
    axes[1].axvline(np.mean(cross_transitions), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(cross_transitions):.1f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Step type distribution (pie chart)
    step_types = ['Within G1', 'Within G2', 'Cross-graph']
    step_counts = [
        detailed_stats['within_G1_steps'],
        detailed_stats['within_G2_steps'],
        detailed_stats['cross_steps']
    ]
    colors = ['blue', 'orange', 'green']
    
    axes[2].pie(step_counts, labels=step_types, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[2].set_title('Walk Step Distribution')
    
    plt.tight_layout()
    
    # Save visualization
    viz_filename = f"random_walks_{method}_{dataset1_name}_{dataset2_name}.png"
    viz_path = OUTPUTS_DIR / viz_filename
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {viz_path}")
    plt.close()
    
    # ============================================
    # Save Walks
    # ============================================
    
    print(f"\nSaving random walks...")
    
    output_dir = base_dir / dataset1_name
    walks_filename = f"random_walks_{method}_{dataset2_name}.pkl"
    walks_path = output_dir / walks_filename
    
    with open(walks_path, 'wb') as f:
        pickle.dump({
            'walks': walks,
            'walks_str': walks_str,
            'stats': walk_stats,
            'detailed_stats': detailed_stats,
            'config': {
                'num_walks': num_walks,
                'walk_length': walk_length,
                'seed': seed,
                'method': method
            }
        }, f)
    
    print(f"  ✓ Saved: {walks_path}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nGenerated random walks for {len(dataset_pairs)} dataset pair(s):")
for dataset1, dataset2 in dataset_pairs:
    print(f"  ✓ {dataset1} ↔ {dataset2}")

print("\n" + "="*60)
print("✓ RANDOM WALK GENERATION COMPLETE")
print("="*60)