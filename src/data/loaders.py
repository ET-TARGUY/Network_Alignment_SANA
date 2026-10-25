"""Functions for loading and preprocessing network data from GeoJSON files."""
import geopandas as gpd
import networkx as nx


def load_graph_from_geojson(nodes_path, pipes_path, data_format='auto'):
    """
    Load graph from GeoJSON files with coordinates.
    
    Args:
        nodes_path: Path to nodes GeoJSON file
        pipes_path: Path to pipes/edges GeoJSON file
        data_format: 'wastewater', 'road', or 'auto' (default: auto-detect)
    
    Returns:
        NetworkX Graph with nodes containing x, y coordinates
    """
    # Load GeoJSON files
    gdf_nodes = gpd.read_file(nodes_path)
    gdf_edges = gpd.read_file(pipes_path)
    
    # Auto-detect format if needed
    if data_format == 'auto':
        if 'old_id' in gdf_nodes.columns and 'sourceNode' in gdf_edges.columns:
            data_format = 'wastewater'
        elif 'nodeID' in gdf_nodes.columns or 'u' in gdf_edges.columns:
            data_format = 'road'
        else:
            # Try to infer from available columns
            print(f"  Node columns: {list(gdf_nodes.columns)}")
            print(f"  Edge columns: {list(gdf_edges.columns)}")
            raise ValueError("Could not auto-detect data format. Please specify 'wastewater' or 'road'")
    
    # Create undirected graph
    G = nx.Graph()
    
    # Load based on format
    if data_format == 'wastewater':
        G = _load_wastewater_format(gdf_nodes, gdf_edges, G)
    elif data_format == 'road':
        G = _load_road_format(gdf_nodes, gdf_edges, G)
    else:
        raise ValueError(f"Unknown data format: {data_format}")
    
    return G


def _load_wastewater_format(gdf_nodes, gdf_edges, G):
    """Load wastewater network format (old_id, sourceNode, targetNode)."""
    # Add nodes with coordinates
    for idx, row in gdf_nodes.iterrows():
        node_id = row['old_id']
        geom = row['geometry']
        
        # Extract all attributes except geometry
        node_attrs = row.drop('geometry').to_dict()
        
        # Add x, y coordinates from geometry
        node_attrs['x'] = geom.x
        node_attrs['y'] = geom.y
        
        G.add_node(node_id, **node_attrs)
    
    # Add edges
    for idx, row in gdf_edges.iterrows():
        source = row['sourceNode']
        target = row['targetNode']
        edge_attrs = row.drop('geometry').to_dict()
        G.add_edge(source, target, **edge_attrs)
    
    return G


def _load_road_format(gdf_nodes, gdf_edges, G, target_crs='EPSG:2154'):
    """Load road network format (from momepy/OSMnx output)."""
    # Determine node ID column

    if gdf_nodes.crs != target_crs:
        print(f"  Reprojecting from {gdf_nodes.crs} to {target_crs}")
        gdf_nodes = gdf_nodes.to_crs(target_crs)
        gdf_edges = gdf_edges.to_crs(target_crs)


    if 'nodeID' in gdf_nodes.columns:
        node_id_col = 'nodeID'
    elif 'osmid' in gdf_nodes.columns:
        node_id_col = 'osmid'
    else:
        # Use index as node ID
        node_id_col = None
    
    # Add nodes with coordinates
    for idx, row in gdf_nodes.iterrows():
        # Get node ID
        if node_id_col:
            node_id = row[node_id_col]
        else:
            node_id = idx
        
        geom = row['geometry']
        
        # Extract all attributes except geometry
        node_attrs = row.drop('geometry').to_dict()
        
        # Add x, y coordinates from geometry
        if hasattr(geom, 'x') and hasattr(geom, 'y'):
            node_attrs['x'] = geom.x
            node_attrs['y'] = geom.y
        else:
            # Handle case where geometry might be different type
            coords = geom.coords[0] if hasattr(geom, 'coords') else (geom.centroid.x, geom.centroid.y)
            node_attrs['x'] = coords[0]
            node_attrs['y'] = coords[1]
        
        G.add_node(node_id, **node_attrs)
    
    # Determine edge source/target columns
    if 'u' in gdf_edges.columns and 'v' in gdf_edges.columns:
        source_col, target_col = 'u', 'v'
    elif 'node_start' in gdf_edges.columns and 'node_end' in gdf_edges.columns:
        source_col, target_col = 'node_start', 'node_end'
    else:
        # Try to extract from geometry endpoints
        print("  ⚠️ No explicit source/target columns found, using geometry endpoints")
        return _load_road_from_geometry(gdf_nodes, gdf_edges, G)
    
    # Add edges
    for idx, row in gdf_edges.iterrows():
        source = row[source_col]
        target = row[target_col]
        edge_attrs = row.drop('geometry').to_dict()
        
        # Only add edge if both nodes exist
        if source in G.nodes() and target in G.nodes():
            G.add_edge(source, target, **edge_attrs)
    
    return G


def _load_road_from_geometry(gdf_nodes, gdf_edges, G):
    """Load road network by matching geometry endpoints to nodes."""
    # Create a spatial index of nodes
    from shapely.geometry import Point
    
    # Build node lookup by coordinates (rounded to avoid floating point issues)
    coord_to_node = {}
    for node_id in G.nodes():
        x, y = G.nodes[node_id]['x'], G.nodes[node_id]['y']
        coord_key = (round(x, 6), round(y, 6))
        coord_to_node[coord_key] = node_id
    
    # Add edges by matching geometry endpoints
    for idx, row in gdf_edges.iterrows():
        geom = row['geometry']
        
        # Get start and end coordinates
        coords = list(geom.coords)
        start_coord = (round(coords[0][0], 6), round(coords[0][1], 6))
        end_coord = (round(coords[-1][0], 6), round(coords[-1][1], 6))
        
        # Find corresponding nodes
        if start_coord in coord_to_node and end_coord in coord_to_node:
            source = coord_to_node[start_coord]
            target = coord_to_node[end_coord]
            
            edge_attrs = row.drop('geometry').to_dict()
            G.add_edge(source, target, **edge_attrs)
    
    return G


def get_coordinates(G):
    """
    Extract node coordinates from graph.
    
    Args:
        G: NetworkX graph with x, y node attributes
    
    Returns:
        dict: {node_id: (x, y)}
    """
    coords = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            coords[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
    return coords


def print_graph_statistics(G, graph_name="Graph"):
    """
    Print basic statistics about a graph.
    
    Args:
        G: NetworkX graph
        graph_name: Name for display
    """
    print(f"\n{graph_name} Statistics:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    
    # Check connectivity
    if nx.is_connected(G):
        print(f"  Connectivity: Fully connected")
    else:
        components = list(nx.connected_components(G))
        print(f"  Connectivity: {len(components)} components")
        print(f"  Largest component: {len(max(components, key=len)):,} nodes")
    
    # Check coordinates
    nodes_with_coords = sum(1 for n in G.nodes() 
                           if 'x' in G.nodes[n] and 'y' in G.nodes[n])
    print(f"  Nodes with coordinates: {nodes_with_coords:,}/{G.number_of_nodes():,}")