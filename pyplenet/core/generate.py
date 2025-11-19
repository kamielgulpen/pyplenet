"""
Network generation module for PyPleNet.

This module provides functions to generate large-scale population networks
using file-based graph storage. It creates nodes from population data and
establishes edges based on interaction patterns, with support for scaling,
reciprocity, and preferential attachment.

The module is designed for generating networks that may be too large to fit
in memory, using the FileBasedGraph class for efficient disk-based storage.

Functions
---------
init_nodes : Initialize nodes in the graph from population data
init_links : Initialize edges in the graph from interaction data  
generate : Main function to generate a complete network

Examples
--------
>>> graph = generate('population.csv', 'interactions.xlsx', 
...                  fraction=0.4, scale=0.1, reciprocity_p=0.2)
>>> print(f"Generated network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
"""
import os
import math
import shutil

import numpy as np

from pyplenet.core.utils import (find_nodes, read_file, desc_groups)
from pyplenet.core.grn import establish_links
from pyplenet.core.graph import FileBasedGraph

def init_communities(G, df, num_of_communities):
    """
    Vectorized community assignment based on edge probabilities for FileBasedGraph.

    Parameters
    ----------
    G : FileBasedGraph
        The graph object with nodes already initialized.
    df : pandas.DataFrame
        DataFrame with columns 'group1', 'group2', and 'edge_count' describing
        edge probabilities between groups.
    num_of_communities : int
        Number of communities to create.

    Returns
    -------
    dict
        Mapping from node_id to community_id (int from 0 to num_of_communities-1).
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    # Create group mapping
    groups = sorted(set(df['group1'].unique()) | set(df['group2'].unique()))
    group_to_idx = {g: i for i, g in enumerate(groups)}
    n_groups = len(groups)

    # Build edge probability matrix
    edge_prob_matrix = np.zeros((n_groups, n_groups))
    total_edges = df['edge_count'].sum()
    for _, row in df.iterrows():
        i, j = group_to_idx[row['group1']], group_to_idx[row['group2']]
        prob = row['edge_count'] / total_edges
        edge_prob_matrix[i, j] = prob
        edge_prob_matrix[j, i] = prob

    # Map nodes to group indices using FileBasedGraph's node_attributes
    node_groups = np.array([
        group_to_idx.get(G.node_attributes[node].get('group'), -1)
        for node in nodes
    ])

    # Initialize community assignments randomly
    assignments = np.random.randint(0, num_of_communities, n_nodes)

    # Iterative refinement (single loop)
    for _ in range(10):  # Few iterations for convergence
        # Calculate community-group densities
        comm_group_counts = np.zeros((num_of_communities, n_groups))
        valid_mask = node_groups >= 0
        np.add.at(comm_group_counts,
                  (assignments[valid_mask], node_groups[valid_mask]), 1)

        # Calculate probabilities for all nodes at once
        probs = np.zeros((n_nodes, num_of_communities))
        valid_nodes = node_groups >= 0
        if valid_nodes.any():
            probs[valid_nodes] = edge_prob_matrix[node_groups[valid_nodes]] @ comm_group_counts.T

        # Reassign based on probabilities
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = np.where(probs_sum > 0, probs / probs_sum, 1/num_of_communities)

        # Probabilistic assignment
        assignments = np.array([
            np.random.choice(num_of_communities, p=probs[i])
            for i in range(n_nodes)
        ])

    return {nodes[i]: assignments[i] for i in range(n_nodes)}

def init_nodes(G, pops_path, scale = 1):
    """
    Initialize nodes in the graph from population data.
    
    Reads population group data from a file and creates nodes with attributes.
    Each group is assigned a number of nodes based on the population count
    scaled by the scale parameter. Node attributes are derived from the 
    characteristic columns in the population data.
    
    Parameters
    ----------
    G : FileBasedGraph
        The FileBasedGraph object to be populated with nodes.
    pops_path : str
        The filepath for the population group sizes file. Can be CSV or Excel format.
    scale : float, optional
        Scaling factor for the number of nodes created. Default is 1.
        Must be larger than 0. Final number of nodes = original_count * scale.
        
    Notes
    -----
    This function modifies the graph object in-place by:
    - Adding nodes with their attributes
    - Setting up group-to-attributes and attributes-to-group mappings
    - Creating group-to-nodes mappings for efficient lookup
    - Initializing link tracking between all group pairs
    
    The function reads the population file using desc_groups() which extracts
    both group descriptions and characteristic columns automatically.
    
    Examples
    --------
    >>> G = FileBasedGraph("my_graph")
    >>> init_nodes(G, "population.csv", scale=0.5)
    >>> print(f"Initialized {G.number_of_nodes()} nodes")
    """
    group_desc_dict, characteristic_cols = desc_groups(pops_path)
    
    group_to_attrs = {}
    group_to_nodes = {}

    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        group_to_attrs[group_id] = attrs
        n_nodes = int(np.ceil(scale * group_info['n']))
        group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))
        
        # Add nodes to graph
        for _ in range(n_nodes):
            G.add_node(node_id, **attrs)
            node_id += 1
    
    # Create a mapping from attrs (as a tuple of sorted items) to group_id
    attrs_to_group = {}
    for group_id, attrs in group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        attrs_to_group[attrs_key] = group_id
    
    # Store metadata in graph
    G.attrs_to_group = attrs_to_group
    G.group_to_attrs = group_to_attrs
    G.group_to_nodes = group_to_nodes
    
    # Dictionary to track links between groups
    group_ids = list(group_to_attrs.keys())
    G.existing_num_links = {(src, dst): 0 for src in group_ids for dst in group_ids}

def init_links(G, links_path, fraction, scale, reciprocity_p):
    """
    Initialize edges in the graph based on interaction data.
    
    Reads interaction/link data from a file and creates edges between nodes
    based on group attributes. Uses preferential attachment and supports
    reciprocal edge creation. The number of links is scaled by scale^2.
    
    Parameters
    ----------
    G : FileBasedGraph
        The graph object with nodes already initialized.
    links_path : str
        Path to the links/interactions data file. Can be CSV or Excel format.
    fraction : float
        Fraction parameter for preferential attachment in establish_links().
        Value between 0 and 1, controls the distribution of connections.
    scale : float
        Scaling factor applied to the population. Link scaling = scale^2.
    reciprocity_p : float
        Probability of creating reciprocal edges. Value between 0 and 1.
        
    Notes
    -----
    The function processes each row in the links file:
    - Extracts source and destination group attributes (columns ending with '_src' and '_dst')
    - Finds nodes matching these attributes using find_nodes()
    - Establishes the requested number of links using establish_links()
    - Tracks warnings for cases where existing links exceed requests
    
    Link scaling uses scale^2 because both source and destination populations
    are scaled by the same factor, so the interaction potential scales quadratically.
    
    Progress is displayed during processing, showing current row number.
    
    Examples
    --------
    >>> init_links(G, "interactions.xlsx", fraction=0.4, scale=0.1, reciprocity_p=0.2)
    Row 5 of 20
    Total requested links: 1250
    """
    
    check_bool = True
    warnings = []
    links_scale = scale**2
    
    df_n_group_links = read_file(links_path)
    print(f"Total requested links: {int(df_n_group_links['n'].sum() * links_scale)}")
    
    print("Preparing maximum number of linkes")
    for idx, row in df_n_group_links.iterrows():
        src_attrs = {k.replace('_src', ''): row[k] for k in row.index if k.endswith('_src')}
        dst_attrs = {k.replace('_dst', ''): row[k] for k in row.index if k.endswith('_dst')}

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        G.maximum_num_links = {}
        G.maximum_num_links[(src_id, dst_id)] = row['n']


    total_rows = len(df_n_group_links)
    for idx, row in df_n_group_links.iterrows():
        
        print(f"\rRow {idx + 1} of {total_rows}", end="")
        
        # Extract source and destination attributes
        src_attrs = {k.replace('_src', ''): row[k] for k in row.index if k.endswith('_src')}
        dst_attrs = {k.replace('_dst', ''): row[k] for k in row.index if k.endswith('_dst')}
        
        num_requested_links = int(math.ceil(row['n'] * links_scale))

        # Find nodes matching the attributes
        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        if not src_nodes or not dst_nodes:
            print("Group empty")
            continue 
        
        # Connect the nodes
        check_bool = establish_links(G, src_nodes, dst_nodes, src_id, dst_id,
                    num_requested_links, fraction, reciprocity_p)
        
        if not check_bool:
            existing_links = G.existing_num_links[(src_id, dst_id)]
            warnings.append(f"Row {idx} || Groups ({src_id})->({dst_id}) || {existing_links} >> {num_requested_links}")
    print()
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(warning)

def generate(pops_path, links_path, preferential_attachment, scale, reciprocity, base_path="graph_data"):
    """
    Generate a complete network using file-based adjacency list representation.
    
    This is the main function that orchestrates the entire network generation
    process. It creates a file-based graph, initializes nodes from population
    data, establishes links from interaction data, and finalizes the graph
    for persistence.
    
    Parameters
    ----------
    pops_path : str
        Path to population data file (CSV or Excel format).
    links_path : str
        Path to links/interactions data file (CSV or Excel format).
    preferetnial_attachment : float
        Parameter for strength of preferential attachment. Value between 0 and 1.
    scale : float
        Scaling factor for population and links. Must be > 0.
        Nodes are scaled by this factor, links by scale^2.
    reciprocity_p : float
        Probability of reciprocal edges. Value between 0 and 1.
    base_path : str, optional
        Base directory path for file-based graph storage. Default is "graph_data".
        Directory will be recreated if it already exists.
        
    Returns
    -------
    FileBasedGraph
        The generated and finalized graph object.
        
    Notes
    -----
    The function performs these steps:
    1. Clean and create the base directory for graph storage
    2. Initialize a FileBasedGraph instance
    3. Create nodes from population data (init_nodes)
    4. Create edges from interaction data (init_links)  
    5. Finalize the graph to ensure data persistence
    
    The base_path directory is completely recreated to ensure a clean state.
    All graph data including adjacency files and metadata are stored there.
    
    Examples
    --------
    >>> graph = generate('pop.csv', 'links.xlsx', 0.4, 0.1, 0.2)
    Generating Nodes
    1000 nodes initialized
    
    Generating Links
    -----------------
    Row 10 of 10
    Total requested links: 500
    -----------------
    Network Generated
    
    Graph finalized: 1000 nodes, 500 edges
    
    >>> graph = generate('pop.csv', 'links.xlsx', 0.4, 0.1, 0.2, 
    ...                  base_path="my_network")
    """
    
    print("Generating Nodes")
    
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)
    
    G = FileBasedGraph(base_path)
    
    init_nodes(G, pops_path, scale)
    print(f"{G.number_of_nodes()} nodes initialized")
    print()
    
    # invert the preferential attachment parameter
    preferential_attachment_fraction = 1 - preferential_attachment
    
    print("Generating Links")
    print("-----------------")
    init_links(G, links_path, preferential_attachment_fraction, scale, reciprocity)
    print("-----------------")
    print("Network Generated")
    print()
    
    # Finalize the graph
    G.finalize()
    
    return G