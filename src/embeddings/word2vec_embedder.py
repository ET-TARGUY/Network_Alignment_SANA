"""Word2Vec embedding generation using Skip-gram."""
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pyplot as plt


class EpochLogger(CallbackAny2Vec):
    """Callback to log training loss per epoch."""
    
    def __init__(self, epochs):
        self.epoch = 0
        self.epochs = epochs
        self.losses = []
        self.previous_loss = 0
        
    def on_epoch_end(self, model):
        cumulative_loss = model.get_latest_training_loss()
        epoch_loss = cumulative_loss - self.previous_loss
        self.losses.append(epoch_loss)
        print(f"  Epoch {self.epoch + 1}/{self.epochs} - Loss: {epoch_loss:,.0f}")
        self.previous_loss = cumulative_loss
        self.epoch += 1


def train_skipgram_model(walks, vector_size=128, window=5, min_count=1,
                         workers=20, epochs=5, sg=1, hs=0, negative=5,
                         seed=42, verbose=True):
    """
    Train Skip-gram model on random walks.
    
    Args:
        walks: list of list of str - Random walks as string sentences
        vector_size: int - Dimensionality of embeddings
        window: int - Context window size
        min_count: int - Minimum word frequency
        workers: int - Number of worker threads
        epochs: int - Number of training epochs
        sg: int - 1 for Skip-gram, 0 for CBOW
        hs: int - 1 for hierarchical softmax, 0 for negative sampling
        negative: int - Number of negative samples
        seed: int - Random seed
        verbose: bool - Print training progress
    
    Returns:
        model: trained Word2Vec model
        training_losses: list of epoch losses
    """
    params = {
        'vector_size': vector_size,
        'window': window,
        'min_count': min_count,
        'workers': workers,
        'epochs': epochs,
        'sg': sg,
        'hs': hs,
        'negative': negative,
        'seed': seed
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING SKIP-GRAM MODEL (Word2Vec)")
        print(f"{'='*60}")
        print(f"\nParameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print(f"\nTraining model...")
        print(f"  This may take several minutes...")
    
    # Setup callback
    epoch_logger = EpochLogger(epochs)
    
    # Train model
    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        hs=hs,
        negative=negative,
        seed=seed,
        callbacks=[epoch_logger],
        compute_loss=True
    )
    
    if verbose:
        print(f"\n✓ Training complete!")
        print(f"\nTraining summary:")
        print(f"  Vocabulary size: {len(model.wv):,}")
        print(f"  Total training loss: {model.get_latest_training_loss():,.0f}")
    
    return model, epoch_logger.losses


def extract_embeddings(model, verbose=True):
    """
    Extract embeddings from trained Word2Vec model.
    
    Args:
        model: trained Word2Vec model
        verbose: bool - Print extraction info
    
    Returns:
        embeddings_G1: dict {node_id: embedding vector}
        embeddings_G2: dict {node_id: embedding vector}
        stats: dict with statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXTRACTING EMBEDDINGS")
        print(f"{'='*60}")
    
    embeddings_G1 = {}
    embeddings_G2 = {}
    
    for word in model.wv.index_to_key:
        if word.startswith('G1_'):
            node_id = int(word.split('_')[1])
            embeddings_G1[node_id] = model.wv[word]
        elif word.startswith('G2_'):
            node_id = int(word.split('_')[1])
            embeddings_G2[node_id] = model.wv[word]
    
    # Count G1 vs G2 nodes
    g1_nodes = [w for w in model.wv.index_to_key if w.startswith('G1_')]
    g2_nodes = [w for w in model.wv.index_to_key if w.startswith('G2_')]
    
    stats = {
        'vocab_size': len(model.wv),
        'g1_nodes': len(g1_nodes),
        'g2_nodes': len(g2_nodes),
        'embedding_dim': model.wv.vector_size
    }
    
    if verbose:
        print(f"\n✓ Extracted embeddings")
        print(f"  Vocabulary size: {stats['vocab_size']:,}")
        print(f"  G1 nodes: {stats['g1_nodes']:,}")
        print(f"  G2 nodes: {stats['g2_nodes']:,}")
        print(f"  Embedding dimension: {stats['embedding_dim']}")
    
    return embeddings_G1, embeddings_G2, stats


def analyze_embeddings(embeddings_G1, embeddings_G2, verbose=True):
    """
    Analyze embedding properties.
    
    Args:
        embeddings_G1: dict of embeddings for G1
        embeddings_G2: dict of embeddings for G2
        verbose: bool - Print analysis
    
    Returns:
        stats: dict with embedding statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EMBEDDING ANALYSIS")
        print(f"{'='*60}")
    
    # Combine all embeddings
    all_embeddings = np.array(
        list(embeddings_G1.values()) + list(embeddings_G2.values())
    )
    
    # Compute norms
    norms = np.linalg.norm(all_embeddings, axis=1)
    
    stats = {
        'norm_mean': np.mean(norms),
        'norm_std': np.std(norms),
        'norm_min': np.min(norms),
        'norm_max': np.max(norms),
        'value_mean': np.mean(all_embeddings),
        'value_std': np.std(all_embeddings),
        'value_min': np.min(all_embeddings),
        'value_max': np.max(all_embeddings)
    }
    
    if verbose:
        print(f"\nEmbedding Norms:")
        print(f"  Mean: {stats['norm_mean']:.3f}")
        print(f"  Std: {stats['norm_std']:.3f}")
        print(f"  Min: {stats['norm_min']:.3f}")
        print(f"  Max: {stats['norm_max']:.3f}")
        
        print(f"\nEmbedding Values:")
        print(f"  Mean: {stats['value_mean']:.3f}")
        print(f"  Std: {stats['value_std']:.3f}")
        print(f"  Min: {stats['value_min']:.3f}")
        print(f"  Max: {stats['value_max']:.3f}")
    
    return stats


def plot_training_loss(losses, output_path):
    """
    Plot training loss curve.
    
    Args:
        losses: list of epoch losses
        output_path: path to save figure
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'o-', 
             linewidth=2, markersize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Skip-Gram Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def show_similar_examples(model, num_examples=3, topn=5):
    """
    Show example similar nodes.
    
    Args:
        model: trained Word2Vec model
        num_examples: number of example queries
        topn: number of similar nodes to show
    """
    print(f"\n{'='*60}")
    print(f"EXAMPLE: MOST SIMILAR NODES")
    print(f"{'='*60}")
    
    # Get G1 and G2 nodes
    g1_nodes = [w for w in model.wv.index_to_key if w.startswith('G1_')]
    g2_nodes = [w for w in model.wv.index_to_key if w.startswith('G2_')]
    
    for i in range(min(num_examples, len(g1_nodes))):
        sample_node = g1_nodes[i]
        print(f"\nQuery: {sample_node}")
        
        # Find most similar G2 nodes
        print(f"Most similar G2 nodes:")
        count = 0
        for word, similarity in model.wv.most_similar(sample_node, topn=100):
            if word.startswith('G2_'):
                print(f"  {word}: {similarity:.4f}")
                count += 1
                if count >= topn:
                    break