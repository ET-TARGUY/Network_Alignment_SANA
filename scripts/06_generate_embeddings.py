"""
Step 6: Generate embeddings using Skip-gram.

This script:
1. Loads configuration
2. Loads random walks
3. Trains Skip-gram model
4. Extracts embeddings
5. Analyzes and saves results
"""
import sys
import os
import pickle
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings.word2vec_embedder import (
    train_skipgram_model,
    extract_embeddings,
    analyze_embeddings,
    plot_training_loss,
    show_similar_examples
)


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 6: GENERATE EMBEDDINGS")
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

# Embedding parameters
emb_config = config['embeddings']

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR = Path(config['paths']['outputs'])
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Method: {method}")
print(f"  Dataset pairs: {dataset_pairs}")

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
    # Load Random Walks
    # ============================================
    
    walks_filename = f"random_walks_{method}_{dataset2_name}.pkl"
    walks_path = base_dir / dataset1_name / walks_filename
    
    if not walks_path.exists():
        print(f"  ❌ Random walks not found: {walks_path}")
        print(f"  Run script 05_generate_random_walks.py first!")
        continue
    
    print(f"\nLoading random walks...")
    with open(walks_path, 'rb') as f:
        data = pickle.load(f)
        walks_str = data['walks_str']
        walk_stats = data['stats']
    
    print(f"  ✓ Loaded {len(walks_str):,} walks")
    print(f"  Mean length: {walk_stats['mean_length']:.1f}")
    print(f"  Mean cross-transitions: {walk_stats['mean_cross_transitions']:.2f}")
    
    # ============================================
    # Train Skip-gram Model
    # ============================================
    
    model, training_losses = train_skipgram_model(
        walks_str,
        vector_size=emb_config['vector_size'],
        window=emb_config['window'],
        min_count=emb_config['min_count'],
        workers=emb_config['workers'],
        epochs=emb_config['epochs'],
        sg=emb_config['sg'],
        hs=emb_config['hs'],
        negative=emb_config['negative'],
        seed=emb_config['seed'],
        verbose=True
    )
    
    # Plot training loss
    loss_filename = f"training_loss_{method}_{dataset1_name}_{dataset2_name}.png"
    loss_path = OUTPUTS_DIR / loss_filename
    plot_training_loss(training_losses, loss_path)
    print(f"\n✓ Saved training loss plot: {loss_path}")
    
    # ============================================
    # Extract Embeddings
    # ============================================
    
    embeddings_G1, embeddings_G2, extract_stats = extract_embeddings(
        model, verbose=True
    )
    
    # ============================================
    # Analyze Embeddings
    # ============================================
    
    embedding_stats = analyze_embeddings(
        embeddings_G1, embeddings_G2, verbose=True
    )
    
    # ============================================
    # Show Examples
    # ============================================
    
    show_similar_examples(model, num_examples=3, topn=5)
    
    # ============================================
    # Save Model and Embeddings
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"SAVING MODEL AND EMBEDDINGS")
    print(f"{'='*60}")
    
    output_dir = base_dir / dataset1_name
    
    # Save Word2Vec model
    model_filename = f"skipgram_model_{method}_{dataset2_name}.model"
    model_path = output_dir / model_filename
    print(f"\nSaving Word2Vec model...")
    model.save(str(model_path))
    print(f"  ✓ Saved: {model_path}")
    
    # Save embeddings
    embeddings_filename = f"embeddings_{method}_{dataset2_name}.pkl"
    embeddings_path = output_dir / embeddings_filename
    print(f"\nSaving embeddings...")
    with open(embeddings_path, 'wb') as f:
        pickle.dump({
            'embeddings_G1': embeddings_G1,
            'embeddings_G2': embeddings_G2,
            'extract_stats': extract_stats,
            'embedding_stats': embedding_stats,
            'training_losses': training_losses,
            'config': emb_config
        }, f)
    print(f"  ✓ Saved: {embeddings_path}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nGenerated embeddings for {len(dataset_pairs)} dataset pair(s):")
for dataset1, dataset2 in dataset_pairs:
    print(f"  ✓ {dataset1} ↔ {dataset2}")

print("\n" + "="*60)
print("✓ EMBEDDING GENERATION COMPLETE")
print("="*60)