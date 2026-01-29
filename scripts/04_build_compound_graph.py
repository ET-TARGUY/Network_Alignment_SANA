"""
Step 4: Build compound graph.

This script:
1. Loads graphs and spatial candidates
2. Builds compound graph using selected method:
   - Gaussian: spatial similarity (with spatial filtering)
   - CENA: structural similarity (ALL cross-graph edges)
3. Saves compound graph for random walk generation
"""
import sys
import os
import pickle
import yaml
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import numpy as np


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.graph.compound import (
    build_compound_graph_gaussian,
    build_compound_graph_cena_full
)


# ============================================
# LOAD CONFIGURATION
# ============================================

print("="*60)
print("STEP 4: BUILD COMPOUND GRAPH")
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

print(f"\nConfiguration:")
print(f"  Zone: {zone}")
print(f"  Method: {method}")
print(f"  Dataset pairs: {dataset_pairs}")
print(f"  Radius: {radius}")

# Determine base directory
if zone == "Prades":
    base_dir = PROCESSED_DATA_DIR / "Prades"
else:
    base_dir = PROCESSED_DATA_DIR

# ============================================
# PROCESS EACH DATASET PAIR
# ============================================
def export_cross_edges_to_geopackage(G_compound, Graph_1, Graph_2, output_path, dataset1_name, dataset2_name):
    """
    Export cross-edges from compound graph to GeoPackage for QGIS visualization.
    
    Parameters:
    -----------
    G_compound : networkx.Graph
        Compound graph containing both networks and cross-edges
    Graph_1, Graph_2 : networkx.Graph
        Original graphs with coordinate information
    output_path : Path
        Path to output GeoPackage file
    dataset1_name, dataset2_name : str
        Names of the datasets for labeling
    """
    print(f"\nExporting cross-edges to GeoPackage...")
    
    # Collect cross-edge data
    cross_edges_data = []
    
    cross_edge_count = 0
    
    for u, v, data in G_compound.edges(data=True):
        # Check if nodes are tuples (compound graph format)
        if isinstance(u, tuple) and isinstance(v, tuple):
            graph_u, node_u = u
            graph_v, node_v = v
            
            # Cross-edge: one node from G1 and one from G2
            if (graph_u == 'G1' and graph_v == 'G2') or (graph_u == 'G2' and graph_v == 'G1'):
                cross_edge_count += 1
                
                # Determine which node is from which graph
                if graph_u == 'G1':
                    node_g1 = node_u
                    node_g2 = node_v
                else:
                    node_g1 = node_v
                    node_g2 = node_u
                
                # Get coordinates from original graphs
                x1 = Graph_1.nodes[node_g1]['x']
                y1 = Graph_1.nodes[node_g1]['y']
                x2 = Graph_2.nodes[node_g2]['x']
                y2 = Graph_2.nodes[node_g2]['y']
                
                # Calculate distance
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Create LineString geometry
                line = LineString([(x1, y1), (x2, y2)])
                
                # Get edge weight if available
                weight = data.get('weight', 1.0)
                
                cross_edges_data.append({
                    'geometry': line,
                    'node_1': str(node_g1),
                    'node_2': str(node_g2),
                    'dataset_1': dataset1_name,
                    'dataset_2': dataset2_name,
                    'distance_m': distance,
                    'weight': weight
                })
    
    print(f"  Debug: Found {cross_edge_count} cross-edges")
    
    if len(cross_edges_data) == 0:
        print("  ⚠️ No cross-edges found!")
        return
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(cross_edges_data, crs="EPSG:2154")
    
    # Save to GeoPackage
    gdf.to_file(output_path, driver="GPKG", layer="cross_edges")
    
    print(f"  ✓ Exported {len(cross_edges_data):,} cross-edges")
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Statistics:")
    print(f"    Mean distance: {gdf['distance_m'].mean():.2f}m")
    print(f"    Max distance: {gdf['distance_m'].max():.2f}m")
    print(f"    Min distance: {gdf['distance_m'].min():.2f}m")
    print(f"    Mean weight: {gdf['weight'].mean():.4f}")

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
    # Load Spatial Candidates (for Gaussian method)
    # ============================================
    
    if method == "gaussian":
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
        
        print(f"  ✓ Loaded candidates")
    
    # ============================================
    # Build Compound Graph
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"BUILDING COMPOUND GRAPH: {method.upper()}")
    print(f"{'='*60}")
    
    if method == "gaussian":
        # Gaussian method (spatial-based, with spatial filtering)
        sigma = config['compound_graph']['gaussian']['sigma']
        threshold = config['compound_graph']['gaussian']['threshold']
        
        G_compound = build_compound_graph_gaussian(
            Graph_1,
            Graph_2,
            candidates_G1_to_G2,
            distances_G1_to_G2,
            sigma=sigma,
            threshold=threshold
        )
        
    elif method == "cena":
        # CENA method (structural-based, ALL cross-graph edges)
        threshold = config['compound_graph']['cena']['threshold']
        K = config['structural']['K']
        
        # Load structural similarities
        structural_filename = f"structural_similarities_to_{dataset2_name}_{radius}m_K{K}.pkl"
        structural_path = base_dir / dataset1_name / structural_filename
        
        if not structural_path.exists():
            print(f"  ❌ Structural similarities not found: {structural_path}")
            print(f"  Run script 03_structural_similarity.py first!")
            continue
        
        print(f"\nLoading structural similarities...")
        with open(structural_path, 'rb') as f:
            structural_data = pickle.load(f)
            structural_similarities = structural_data['similarities']
        
        print(f"  ✓ Loaded {len(structural_similarities):,} structural similarities")
        
        # Build CENA compound graph with ALL cross-edges
        G_compound = build_compound_graph_cena_full(
            Graph_1,
            Graph_2,
            structural_similarities,
            threshold=threshold,
            verbose=True
        )
        
    else:
        print(f"  ❌ Unknown method: {method}")
        continue
    
    # ============================================
    # Save Compound Graph
    # ============================================
    
    print(f"\nSaving compound graph...")
    
    output_dir = base_dir / dataset1_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on method
    if method == "gaussian":
        sigma = config['compound_graph']['gaussian']['sigma']
        compound_filename = f"compound_graph_gaussian_{dataset2_name}_{radius}m_sigma{sigma}.pkl"
    elif method == "cena":
        K = config['structural']['K']
        compound_filename = f"compound_graph_cena_{dataset2_name}_{radius}m_K{K}.pkl"
    
    compound_path = output_dir / compound_filename
    
    with open(compound_path, 'wb') as f:
        pickle.dump({
            'graph': G_compound,
            'method': method,
            'config': config,
            'dataset1': dataset1_name,
            'dataset2': dataset2_name
        }, f)
    
    print(f"  ✓ Saved: {compound_path}")



    # ============================================
    # Export Cross-Edges to GeoPackage for Visualization
    # ============================================

    print(f"\n{'='*60}")
    print(f"EXPORTING CROSS-EDGES FOR VISUALIZATION")
    print(f"{'='*60}")

    # Create visualization output directory
    viz_dir = Path("outputs/visualization")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    if method == "gaussian":
        sigma = config['compound_graph']['gaussian']['sigma']
        gpkg_filename = f"cross_edges_{dataset1_name}_to_{dataset2_name}_gaussian_{radius}m_sigma{sigma}.gpkg"
    elif method == "cena":
        K = config['structural']['K']
        gpkg_filename = f"cross_edges_{dataset1_name}_to_{dataset2_name}_cena_{radius}m_K{K}.gpkg"

    gpkg_path = viz_dir / gpkg_filename

    # Export cross-edges
    export_cross_edges_to_geopackage(
        G_compound,
        Graph_1,
        Graph_2,
        gpkg_path,
        dataset1_name,
        dataset2_name
    )

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nProcessed {len(dataset_pairs)} dataset pair(s) using {method} method:")
for dataset1, dataset2 in dataset_pairs:
    print(f"  ✓ {dataset1} ↔ {dataset2}")

print("\n" + "="*60)
print("✓ COMPOUND GRAPH CONSTRUCTION COMPLETE")
print("="*60)












