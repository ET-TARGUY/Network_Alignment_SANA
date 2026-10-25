"""Compound graph construction methods."""
import numpy as np
import networkx as nx
from tqdm import tqdm


def compute_spatial_similarity(distance, sigma=5, mode="gaussian"):
    """
    Convert distance to similarity using either:
    - Exponential decay
    - Gaussian (RBF) decay
    
    Args:
        distance: spatial distance in meters
        sigma: decay parameter
        mode: "exp" or "gaussian"
    
    Returns:
        similarity score [0, 1]
    """
    if mode == "exp":
        # e^(-d / σ)
        return np.exp(-distance / sigma)
    elif mode == "gaussian":
        # e^(-(d²) / (2σ²))
        return np.exp(-(distance**2) / (2 * sigma**2))
    else:
        raise ValueError("mode must be 'exp' or 'gaussian'")


def build_compound_graph_gaussian(G1, G2, candidates_dict, distances_dict,
                                   sigma=5, threshold=0.0):
    """
    Build compound graph using Gaussian spatial similarity.
    
    Cross-graph edge weight = exp(-(d²) / (2σ²))
    
    Args:
        G1, G2: NetworkX graphs
        candidates_dict: dict {node_g1: [candidates in G2]}
        distances_dict: dict {node_g1: [distances]}
        sigma: Gaussian bandwidth parameter
        threshold: minimum similarity to add edge
    
    Returns:
        G_compound: compound graph
        stats: statistics dictionary
    """
    print(f"\n{'='*60}")
    print(f"COMPOUND GRAPH: GAUSSIAN METHOD")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  σ = {sigma}")
    print(f"  Threshold = {threshold}")
    print(f"  Distance threshold ≈ {sigma * np.sqrt(-2 * np.log(threshold)):.1f}m")
    
    # Create empty compound graph
    G_compound = nx.Graph()
    
    # ========================================
    # Add nodes from G1
    # ========================================
    print(f"\nAdding nodes from G1...")
    for node in tqdm(G1.nodes(), desc="G1 nodes"):
        node_id = ('G1', node)
        G_compound.add_node(node_id, 
                          graph='G1', 
                          original_id=node,
                          x=G1.nodes[node]['x'],
                          y=G1.nodes[node]['y'])
        
        # Copy other attributes
        for attr, value in G1.nodes[node].items():
            if attr not in ['x', 'y']:
                G_compound.nodes[node_id][attr] = value
    
    # ========================================
    # Add nodes from G2
    # ========================================
    print(f"Adding nodes from G2...")
    for node in tqdm(G2.nodes(), desc="G2 nodes"):
        node_id = ('G2', node)
        G_compound.add_node(node_id,
                          graph='G2',
                          original_id=node,
                          x=G2.nodes[node]['x'],
                          y=G2.nodes[node]['y'])
        
        # Copy other attributes
        for attr, value in G2.nodes[node].items():
            if attr not in ['x', 'y']:
                G_compound.nodes[node_id][attr] = value
    
    # ========================================
    # Add within-G1 edges
    # ========================================
    print(f"Adding within-G1 edges...")
    for u, v in tqdm(G1.edges(), desc="G1 edges"):
        u_id = ('G1', u)
        v_id = ('G1', v)
        G_compound.add_edge(u_id, v_id, 
                          edge_type='within_G1',
                          weight=1.0)
        
        # Copy edge attributes
        for attr, value in G1.edges[u, v].items():
            G_compound.edges[u_id, v_id][attr] = value
    
    # ========================================
    # Add within-G2 edges
    # ========================================
    print(f"Adding within-G2 edges...")
    for u, v in tqdm(G2.edges(), desc="G2 edges"):
        u_id = ('G2', u)
        v_id = ('G2', v)
        G_compound.add_edge(u_id, v_id,
                          edge_type='within_G2',
                          weight=1.0)
        
        # Copy edge attributes
        for attr, value in G2.edges[u, v].items():
            G_compound.edges[u_id, v_id][attr] = value
    
    # ========================================
    # Add cross-graph edges (Gaussian)
    # ========================================
    print(f"\nAdding cross-graph edges (Gaussian similarity)...")
    
    cross_edges_added = 0
    total_candidates = 0
    filtered_out = 0
    similarity_values = []
    distance_values = []
    
    for node_G1, candidates in tqdm(candidates_dict.items(), desc="Cross-graph edges"):
        distances = distances_dict[node_G1]
        
        for candidate_G2, distance in zip(candidates, distances):
            total_candidates += 1
            
            # Compute Gaussian spatial similarity
            similarity = compute_spatial_similarity(distance, sigma, mode="gaussian")
            
            # Only add edge if above threshold
            if similarity >= threshold:
                u_id = ('G1', node_G1)
                v_id = ('G2', candidate_G2)
                
                G_compound.add_edge(u_id, v_id,
                                  edge_type='cross_graph',
                                  weight=similarity,
                                  distance=distance)
                
                cross_edges_added += 1
                similarity_values.append(similarity)
                distance_values.append(distance)
            else:
                filtered_out += 1
    
    # Print statistics
    _print_compound_graph_stats(G_compound, G1, G2, 
                                cross_edges_added, total_candidates, filtered_out,
                                similarity_values, distance_values)
    
    # Return statistics
    stats = {
        'method': 'gaussian',
        'sigma': sigma,
        'threshold': threshold,
        'total_candidates': total_candidates,
        'edges_added': cross_edges_added,
        'edges_filtered': filtered_out,
        'similarity_values': np.array(similarity_values),
        'distance_values': np.array(distance_values),
        'mean_similarity': np.mean(similarity_values) if similarity_values else 0,
        'median_similarity': np.median(similarity_values) if similarity_values else 0
    }
    
    return G_compound



def _print_compound_graph_stats(G_compound, G1, G2, 
                                
                                cross_edges_added, total_candidates, filtered_out,
                                similarity_values, distance_values):
    """Helper function to print compound graph statistics."""
    print(f"\n{'='*60}")
    print(f"COMPOUND GRAPH STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nNodes:")
    print(f"  Total nodes: {G_compound.number_of_nodes():,}")
    print(f"  Nodes from G1: {G1.number_of_nodes():,}")
    print(f"  Nodes from G2: {G2.number_of_nodes():,}")
    
    print(f"\nEdges:")
    print(f"  Total edges: {G_compound.number_of_edges():,}")
    print(f"  Within-G1 edges: {G1.number_of_edges():,}")
    print(f"  Within-G2 edges: {G2.number_of_edges():,}")
    print(f"  Cross-graph edges: {cross_edges_added:,}")
    
    print(f"\nCross-graph Edge Filtering:")
    print(f"  Total candidates: {total_candidates:,}")
    print(f"  Added (≥ threshold): {cross_edges_added:,} ({100*cross_edges_added/total_candidates:.1f}%)")
    print(f"  Filtered out: {filtered_out:,} ({100*filtered_out/total_candidates:.1f}%)")
    
    if similarity_values:
        similarity_values = np.array(similarity_values)
        
        print(f"\nCross-graph Edge Weights:")
        print(f"  Mean similarity: {np.mean(similarity_values):.3f}")
        print(f"  Median similarity: {np.median(similarity_values):.3f}")
        print(f"  Min similarity: {np.min(similarity_values):.3f}")
        print(f"  Max similarity: {np.max(similarity_values):.3f}")
    
    if distance_values is not None:
        distance_values = np.array(distance_values)
        print(f"\nCross-graph Edge Distances:")
        print(f"  Mean distance: {np.mean(distance_values):.2f}m")
        print(f"  Median distance: {np.median(distance_values):.2f}m")
        print(f"  Max distance: {np.max(distance_values):.2f}m")
    
    # Degree statistics
    degrees = dict(G_compound.degree())
    degrees_array = np.array(list(degrees.values()))
    
    print(f"\nDegree Statistics:")
    print(f"  Mean degree: {np.mean(degrees_array):.2f}")
    print(f"  Median degree: {np.median(degrees_array):.0f}")
    print(f"  Max degree: {np.max(degrees_array)}")
    
    # Connected components
    num_components = nx.number_connected_components(G_compound)
    print(f"\nConnectivity:")
    print(f"  Connected components: {num_components}")
    if num_components == 1:
        print(f"  ✓ Graph is fully connected!")
    else:
        component_sizes = [len(c) for c in nx.connected_components(G_compound)]
        print(f"  Largest component: {max(component_sizes):,} nodes")
        print(f"  Smallest component: {min(component_sizes)} nodes")




def build_compound_graph_cena_full(G1, G2, structural_similarities, threshold=0.0, verbose=True):
    """
    Build compound graph using CENA method with ALL cross-graph edges.
    
    CENA uses structural similarity for ALL node pairs across networks,
    not limited by spatial filtering.
    
    Args:
        G1: networkx.Graph - First graph
        G2: networkx.Graph - Second graph  
        structural_similarities: dict - {(node_g1, node_g2): similarity}
        threshold: float - Minimum similarity threshold
        verbose: bool - Print progress
    
    Returns:
        G_compound: networkx.Graph - Compound graph with all nodes and edges
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPOUND GRAPH: CENA METHOD (FULL)")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  Threshold = {threshold}")
        print(f"  Cross-graph edges: ALL node pairs")
    
    # Initialize compound graph
    G_compound = nx.Graph()
    
    # Add nodes from G1
    if verbose:
        print(f"\nAdding nodes from G1...")
    for node in tqdm(G1.nodes(), desc="G1 nodes", disable=not verbose):
        G_compound.add_node(
            ('G1', node),
            graph='G1',
            original_id=node,
            **G1.nodes[node]
        )
    
    # Add nodes from G2
    if verbose:
        print(f"Adding nodes from G2...")
    for node in tqdm(G2.nodes(), desc="G2 nodes", disable=not verbose):
        G_compound.add_node(
            ('G2', node),
            graph='G2',
            original_id=node,
            **G2.nodes[node]
        )
    
    # Add within-G1 edges
    if verbose:
        print(f"Adding within-G1 edges...")
    for u, v in tqdm(G1.edges(), desc="G1 edges", disable=not verbose):
        G_compound.add_edge(
            ('G1', u),
            ('G1', v),
            edge_type='within_G1',
            weight=1.0
        )
    
    # Add within-G2 edges
    if verbose:
        print(f"Adding within-G2 edges...")
    for u, v in tqdm(G2.edges(), desc="G2 edges", disable=not verbose):
        G_compound.add_edge(
            ('G2', u),
            ('G2', v),
            edge_type='within_G2',
            weight=1.0
        )
    
    # Add cross-graph edges based on structural similarity
    if verbose:
        print(f"\nAdding cross-graph edges (CENA - structural similarity)...")
    
    cross_edges_added = 0
    cross_edges_filtered = 0
    cross_edge_weights = []
    
    for (node_g1, node_g2), similarity in tqdm(structural_similarities.items(), 
                                                desc="Cross-graph edges", 
                                                disable=not verbose):
        if similarity >= threshold:
            G_compound.add_edge(
                ('G1', node_g1),
                ('G2', node_g2),
                edge_type='cross_graph',
                weight=similarity
            )
            cross_edges_added += 1
            cross_edge_weights.append(similarity)
        else:
            cross_edges_filtered += 1
    
    # Statistics
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPOUND GRAPH STATISTICS")
        print(f"{'='*60}")
        
        print(f"\nNodes:")
        print(f"  Total nodes: {G_compound.number_of_nodes():,}")
        print(f"  Nodes from G1: {G1.number_of_nodes():,}")
        print(f"  Nodes from G2: {G2.number_of_nodes():,}")
        
        print(f"\nEdges:")
        print(f"  Total edges: {G_compound.number_of_edges():,}")
        print(f"  Within-G1 edges: {G1.number_of_edges():,}")
        print(f"  Within-G2 edges: {G2.number_of_edges():,}")
        print(f"  Cross-graph edges: {cross_edges_added:,}")
        
        print(f"\nCross-graph Edge Filtering:")
        print(f"  Total candidates: {len(structural_similarities):,}")
        print(f"  Added (≥ threshold): {cross_edges_added:,} ({100*cross_edges_added/len(structural_similarities):.1f}%)")
        print(f"  Filtered out: {cross_edges_filtered:,} ({100*cross_edges_filtered/len(structural_similarities):.1f}%)")
        
        if cross_edge_weights:
            print(f"\nCross-graph Edge Weights (Structural Similarity):")
            print(f"  Mean similarity: {np.mean(cross_edge_weights):.3f}")
            print(f"  Median similarity: {np.median(cross_edge_weights):.3f}")
            print(f"  Min similarity: {np.min(cross_edge_weights):.3f}")
            print(f"  Max similarity: {np.max(cross_edge_weights):.3f}")
        
        # Degree statistics
        degrees = [G_compound.degree(n) for n in G_compound.nodes()]
        print(f"\nDegree Statistics:")
        print(f"  Mean degree: {np.mean(degrees):.2f}")
        print(f"  Median degree: {np.median(degrees):.0f}")
        print(f"  Max degree: {np.max(degrees)}")
        
        # Connectivity
        num_components = nx.number_connected_components(G_compound)
        print(f"\nConnectivity:")
        print(f"  Connected components: {num_components}")
        if num_components == 1:
            print(f"  ✓ Graph is fully connected!")
        else:
            largest_cc = max(nx.connected_components(G_compound), key=len)
            smallest_cc = min(nx.connected_components(G_compound), key=len)
            print(f"  Largest component: {len(largest_cc)} nodes")
            print(f"  Smallest component: {len(smallest_cc)} nodes")
    
    return G_compound



