"""Random walk generation for compound graphs."""
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict


def weighted_random_walk(G, start_node, walk_length):
    """
    Generate a single weighted random walk starting from start_node.
    
    Args:
        G: networkx.Graph - The graph to walk on
        start_node: Starting node for the walk
        walk_length: int - Maximum length of the walk
    
    Returns:
        walk: list - Sequence of nodes in the walk
    """
    walk = [start_node]
    current = start_node
    
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(current))
        
        if len(neighbors) == 0:
            # Dead end - stop walk
            break
        
        # Get weights for all neighbors
        weights = np.array([G[current][neighbor].get('weight', 1.0) 
                           for neighbor in neighbors])
        
        # Normalize to probabilities
        probabilities = weights / weights.sum()
        
        # Sample next node by index, then get the actual node
        neighbor_idx = np.random.choice(len(neighbors), p=probabilities)
        next_node = neighbors[neighbor_idx]
        
        walk.append(next_node)
        current = next_node
    
    return walk


def generate_random_walks(G, num_walks=10, walk_length=80, verbose=True):
    """
    Generate random walks from all nodes in the graph.
    
    Args:
        G: networkx.Graph - The graph to walk on
        num_walks: int - Number of walks to generate per node
        walk_length: int - Length of each walk
        verbose: bool - Show progress bar
    
    Returns:
        walks: list of list - All generated walks
        stats: dict - Statistics about the walks
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING RANDOM WALKS")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  num_walks per node: {num_walks}")
        print(f"  walk_length: {walk_length}")
        print(f"  Total nodes: {G.number_of_nodes():,}")
        print(f"  Expected walks: ~{G.number_of_nodes() * num_walks:,}")
    
    all_walks = []
    walk_lengths = []
    cross_graph_transitions = []
    
    nodes = list(G.nodes())
    
    # Generate walks
    for node in tqdm(nodes, desc="Generating walks", disable=not verbose):
        for _ in range(num_walks):
            walk = weighted_random_walk(G, node, walk_length)
            all_walks.append(walk)
            walk_lengths.append(len(walk))
            
            # Count cross-graph transitions
            transitions = 0
            for i in range(len(walk) - 1):
                graph_current = G.nodes[walk[i]]['graph']
                graph_next = G.nodes[walk[i+1]]['graph']
                if graph_current != graph_next:
                    transitions += 1
            cross_graph_transitions.append(transitions)
    
    # Statistics
    walk_lengths = np.array(walk_lengths)
    cross_graph_transitions = np.array(cross_graph_transitions)
    
    stats = {
        'total_walks': len(all_walks),
        'mean_length': np.mean(walk_lengths),
        'median_length': np.median(walk_lengths),
        'min_length': np.min(walk_lengths),
        'max_length': np.max(walk_lengths),
        'mean_cross_transitions': np.mean(cross_graph_transitions),
        'median_cross_transitions': np.median(cross_graph_transitions),
        'walks_with_cross': np.sum(cross_graph_transitions > 0),
        'percentage_with_cross': 100 * np.sum(cross_graph_transitions > 0) / len(all_walks)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"WALK GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"\nGenerated walks: {stats['total_walks']:,}")
        print(f"\nWalk Lengths:")
        print(f"  Mean: {stats['mean_length']:.2f}")
        print(f"  Median: {stats['median_length']:.0f}")
        print(f"  Min: {stats['min_length']}")
        print(f"  Max: {stats['max_length']}")
        
        print(f"\nCross-Graph Transitions:")
        print(f"  Mean per walk: {stats['mean_cross_transitions']:.2f}")
        print(f"  Median per walk: {stats['median_cross_transitions']:.0f}")
        print(f"  Walks with ≥1 cross-transition: {stats['walks_with_cross']:,} ({stats['percentage_with_cross']:.1f}%)")
    
    return all_walks, stats


def walks_to_sentences(walks):
    """
    Convert walks (with tuple node IDs) to string sentences for Word2Vec.
    
    Args:
        walks: list of list of tuples - Walks with node IDs as tuples like ('G1', 123)
    
    Returns:
        sentences: list of list of str - Walks with node IDs as strings like 'G1_123'
    """
    sentences = []
    for walk in walks:
        sentence = [f"{graph}_{node_id}" for graph, node_id in walk]
        sentences.append(sentence)
    return sentences


def analyze_walk_statistics(walks_str):
    """
    Analyze detailed statistics about random walks.
    
    Args:
        walks_str: list of list of str - Walks with string node IDs
    
    Returns:
        stats: dict - Detailed walk statistics
    """
    total_steps = 0
    within_G1_steps = 0
    within_G2_steps = 0
    cross_steps = 0
    
    for walk in walks_str:
        for i in range(len(walk) - 1):
            total_steps += 1
            
            graph_current = walk[i].split('_')[0]
            graph_next = walk[i+1].split('_')[0]
            
            if graph_current == graph_next:
                if graph_current == 'G1':
                    within_G1_steps += 1
                else:
                    within_G2_steps += 1
            else:
                cross_steps += 1
    
    stats = {
        'total_steps': total_steps,
        'within_G1_steps': within_G1_steps,
        'within_G2_steps': within_G2_steps,
        'cross_steps': cross_steps,
        'within_G1_pct': 100 * within_G1_steps / total_steps if total_steps > 0 else 0,
        'within_G2_pct': 100 * within_G2_steps / total_steps if total_steps > 0 else 0,
        'cross_pct': 100 * cross_steps / total_steps if total_steps > 0 else 0
    }
    
    return stats


def biased_random_walk_cena(G, start_node, walk_length, q=0.5):
    """
    Generate a biased random walk for CENA (with network switching).
    
    Args:
        G: networkx.Graph - Compound graph with 'graph' and 'edge_type' attributes
        start_node: Starting node for the walk
        walk_length: int - Maximum length of the walk
        q: float - Probability of staying in current network (vs switching)
    
    Returns:
        walk: list - Sequence of nodes in the walk
    """
    walk = [start_node]
    current = start_node
    
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(current))
        
        if len(neighbors) == 0:
            break
        
        # Get current graph
        current_graph = G.nodes[current]['graph']
        
        # Separate neighbors by edge type
        same_graph_neighbors = []
        cross_graph_neighbors = []
        
        for neighbor in neighbors:
            edge_type = G[current][neighbor].get('edge_type', 'unknown')
            if edge_type.startswith('within_'):
                same_graph_neighbors.append(neighbor)
            elif edge_type == 'cross_graph':
                cross_graph_neighbors.append(neighbor)
        
        # Decide: stay in current network or switch?
        if np.random.random() < q:
            # Stay in current network (if possible)
            if len(same_graph_neighbors) > 0:
                # Uniform probability among same-graph neighbors
                next_node = same_graph_neighbors[np.random.randint(len(same_graph_neighbors))]
            elif len(cross_graph_neighbors) > 0:
                # Forced to switch if no same-graph neighbors
                weights = np.array([G[current][neighbor]['weight'] 
                                   for neighbor in cross_graph_neighbors])
                weights = weights / weights.sum()
                idx = np.random.choice(len(cross_graph_neighbors), p=weights)
                next_node = cross_graph_neighbors[idx]
            else:
                break
        else:
            # Switch networks (if possible)
            if len(cross_graph_neighbors) > 0:
                # Weighted probability based on structural similarity
                weights = np.array([G[current][neighbor]['weight'] 
                                   for neighbor in cross_graph_neighbors])
                weights = weights / weights.sum()
                idx = np.random.choice(len(cross_graph_neighbors), p=weights)
                next_node = cross_graph_neighbors[idx]
            elif len(same_graph_neighbors) > 0:
                # Forced to stay if no cross-graph neighbors
                next_node = same_graph_neighbors[np.random.randint(len(same_graph_neighbors))]
            else:
                break
        
        walk.append(next_node)
        current = next_node
    
    return walk


def generate_random_walks_cena(G, num_walks=10, walk_length=80, q=0.5, verbose=True):
    """
    Generate CENA biased random walks from all nodes.
    
    Args:
        G: networkx.Graph - The compound graph
        num_walks: int - Number of walks per node
        walk_length: int - Length of each walk
        q: float - Network switching probability (0.5 = equal probability)
        verbose: bool - Show progress
    
    Returns:
        walks: list of list - All generated walks
        stats: dict - Statistics about the walks
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING CENA BIASED RANDOM WALKS")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  num_walks per node: {num_walks}")
        print(f"  walk_length: {walk_length}")
        print(f"  q (stay in network): {q}")
        print(f"  Total nodes: {G.number_of_nodes():,}")
        print(f"  Expected walks: ~{G.number_of_nodes() * num_walks:,}")
    
    all_walks = []
    walk_lengths = []
    cross_graph_transitions = []
    
    nodes = list(G.nodes())
    
    # Generate walks
    for node in tqdm(nodes, desc="Generating CENA walks", disable=not verbose):
        for _ in range(num_walks):
            walk = biased_random_walk_cena(G, node, walk_length, q=q)
            all_walks.append(walk)
            walk_lengths.append(len(walk))
            
            # Count cross-graph transitions
            transitions = 0
            for i in range(len(walk) - 1):
                graph_current = G.nodes[walk[i]]['graph']
                graph_next = G.nodes[walk[i+1]]['graph']
                if graph_current != graph_next:
                    transitions += 1
            cross_graph_transitions.append(transitions)
    
    # Statistics
    walk_lengths = np.array(walk_lengths)
    cross_graph_transitions = np.array(cross_graph_transitions)
    
    stats = {
        'total_walks': len(all_walks),
        'mean_length': np.mean(walk_lengths),
        'median_length': np.median(walk_lengths),
        'min_length': np.min(walk_lengths),
        'max_length': np.max(walk_lengths),
        'mean_cross_transitions': np.mean(cross_graph_transitions),
        'median_cross_transitions': np.median(cross_graph_transitions),
        'walks_with_cross': np.sum(cross_graph_transitions > 0),
        'percentage_with_cross': 100 * np.sum(cross_graph_transitions > 0) / len(all_walks)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CENA WALK GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"\nGenerated walks: {stats['total_walks']:,}")
        print(f"\nWalk Lengths:")
        print(f"  Mean: {stats['mean_length']:.2f}")
        print(f"  Median: {stats['median_length']:.0f}")
        print(f"  Min: {stats['min_length']}")
        print(f"  Max: {stats['max_length']}")
        
        print(f"\nCross-Graph Transitions:")
        print(f"  Mean per walk: {stats['mean_cross_transitions']:.2f}")
        print(f"  Median per walk: {stats['median_cross_transitions']:.0f}")
        print(f"  Walks with ≥1 cross-transition: {stats['walks_with_cross']:,} ({stats['percentage_with_cross']:.1f}%)")
    
    return all_walks, stats