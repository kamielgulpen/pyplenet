"""
Network generation module for PyPleNet NetworkX.

This module provides functions to generate large-scale population networks
using NetworkX in-memory graph storage. It creates nodes from population data and
establishes edges based on interaction patterns, with support for scaling,
reciprocity, and preferential attachment.

This NetworkX-based implementation is significantly faster than file-based
approaches for graphs that fit in memory.

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

from pyplenet_nx.core.utils import (find_nodes, read_file, desc_groups)
from pyplenet_nx.core.grn import establish_links
from pyplenet_nx.core.graph import NetworkXGraph
import networkx as nx

import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

def build_group_pair_to_communities_lookup(G):
    """
    Build reverse lookup: (src_group, dst_group) -> [valid_community_ids]

    This is computed ONCE and eliminates the O(n_communities) check
    for every link attempt.

    Returns
    -------
    dict : (src_id, dst_id) -> list of community IDs
    """
    print("Building reverse lookup for community membership...")

    group_pair_to_communities = {}

    # For each community, get all group pairs within it
    for community_id in range(G.number_of_communities):
        groups_in_community = set(G.communities_to_groups.get(community_id, []))

        # Create pairs from all groups in this community
        for src_id in groups_in_community:
            for dst_id in groups_in_community:
                pair_key = (src_id, dst_id)

                if pair_key not in group_pair_to_communities:
                    group_pair_to_communities[pair_key] = []

                group_pair_to_communities[pair_key].append(community_id)

    print(f"  Cached {len(group_pair_to_communities)} group pair -> community mappings")
    print(f"  Average communities per pair: {np.mean([len(v) for v in group_pair_to_communities.values()]):.1f}")

    return group_pair_to_communities

def populate_communities(G, x, selected_indices):

    n_groups = int(np.sqrt(len(G.maximum_num_links)))

    # Build affinity matrix
    community_matrix = np.ones((x, n_groups))

    communities_to_nodes = {}
    nodes_to_communities = {}
    communities_to_groups = {}
    if x == 1 : selected_indices = [selected_indices[0]]
    for count, group in enumerate(selected_indices):
        # introduce bias based on selected indices
        community_matrix[count,group] += 1
        for i in range(n_groups):
            communities_to_nodes[(count,i)] = []

        communities_to_groups[count] = []

    for node in G.graph.nodes:
        group = G.nodes_to_group[node]
        probability_vector = G.probability_matrix[group]

        group_probabilities = np.dot(community_matrix, probability_vector) 
        group_probabilities = group_probabilities / (group_probabilities.sum())
        community = np.random.choice(x, p=group_probabilities)
        community_matrix[community,group] += 1
        communities_to_nodes[(community,group)].append(node)
        nodes_to_communities[node] = community
        communities_to_groups[community].append(group)

    G.communities_to_nodes = communities_to_nodes
    G.nodes_to_communities = nodes_to_communities
    G.communities_to_groups = communities_to_groups
 

def find_separated_groups(G, x):
    """
    Find x groups that should be in different communities (lowest inter-connections).
    
    Parameters:
    - df: DataFrame with edge counts between groups
    - x: Number of groups to return
    
    Returns:
    - List of x groups that have minimal connections
    """
    n_groups = int(np.sqrt(len(G.maximum_num_links)))

    # Build affinity matrix
    affinity = np.zeros((n_groups, n_groups))
    for edges in G.maximum_num_links:
        i,j = edges
        affinity[i, j] = G.maximum_num_links[edges]
    
    epsilon = 1e-5
    normalized = affinity / (affinity.sum(axis=1, keepdims=True) + epsilon)
    normalized[normalized == 0] = epsilon
    G.probability_matrix = normalized.copy()
    G.number_of_communities = x

    # Greedy selection: start with two least connected groups
    selected_indices = []
    remaining = set(range(n_groups))
    
    # Find pair with minimum connection
    np.fill_diagonal(normalized, np.inf)
    min_i, min_j = np.unravel_index(np.argmin(normalized), normalized.shape)
    selected_indices.extend([min_i, min_j])
    remaining.discard(min_i)
    remaining.discard(min_j)
    
    # Add groups with minimum total connection to selected groups
    while len(selected_indices) < min(x, n_groups):
        min_total = np.inf
        best_group = None
        
        for g in remaining:
            total_connection = normalized[g, selected_indices].sum()
            if total_connection < min_total:
                min_total = total_connection
                best_group = g
        
        selected_indices.append(best_group)
        remaining.discard(best_group)
    
    extend_selected_indices = selected_indices*x
    diff = x - len(selected_indices)
    if diff > 0 : selected_indices.extend(extend_selected_indices[:diff])
    print(diff, selected_indices[:diff], (len(selected_indices)), x)
    populate_communities(G, x, selected_indices)
    return selected_indices

def init_nodes(G, pops_path, scale = 1):
    """
    Initialize nodes from population data using NetworkX directly.

    Parameters
    ----------
    G : NetworkXGraph
        Wrapper with G.graph (nx.DiGraph) and metadata
    pops_path : str
        Path to population file (CSV or Excel)
    scale : float, optional
        Population scaling factor (default 1)
    group_to_community : dict, optional
        Mapping from group_id to community_id (from edge distribution analysis)
    """
    group_desc_dict, characteristic_cols = desc_groups(pops_path)

    G.nr_of_total_pop_nodes = read_file(pops_path).n.sum()
    group_to_attrs = {}
    group_to_nodes = {}
    nodes_to_group = {}

    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        group_to_attrs[group_id] = attrs
        n_nodes = int(np.ceil(scale * group_info['n']))
        group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))

        # Add nodes using NetworkX directly with community attribute
        for _ in range(n_nodes):
            G.graph.add_node(node_id, **attrs)
            nodes_to_group[node_id] = group_id
            node_id += 1


    # Create attribute to group mapping
    attrs_to_group = {}
    for group_id, attrs in group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        attrs_to_group[attrs_key] = group_id

    # Store metadata in wrapper
    G.attrs_to_group = attrs_to_group
    G.group_to_attrs = group_to_attrs
    G.group_to_nodes = group_to_nodes
    G.nodes_to_group = nodes_to_group

    # Initialize link tracking
    group_ids = list(group_to_attrs.keys())
    G.existing_num_links = {(src, dst): 0 for src in group_ids for dst in group_ids}

def init_links(G, links_path, fraction, scale, reciprocity_p, transitivity_p, number_of_communities):
    """
    Initialize edges in the graph based on interaction data.
    
    Reads interaction/link data from a file and creates edges between nodes
    based on group attributes. Uses preferential attachment and supports
    reciprocal edge creation. The number of links is scaled by scale^2.
    
    Parameters
    ----------
    G : NetworkXGraph
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

    df_n_group_links = read_file(links_path)
    links_scale = scale

    print("Preparing maximum number of linkes")
    G.maximum_num_links = {}

    requested_links = []
    group_ids = range(240)  # or whatever your group IDs are
    G.maximum_num_links = {(i, j): 0 for i in group_ids for j in group_ids}
    for idx, row in df_n_group_links.iterrows():
        src_attrs = {k.replace('_src', ''): row[k] for k in row.index if k.endswith('_src')}
        dst_attrs = {k.replace('_dst', ''): row[k] for k in row.index if k.endswith('_dst')}

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G,**dst_attrs)


        G.maximum_num_links[(src_id, dst_id)] = int(math.ceil(row['n'] * links_scale))
        requested_links.append(int(math.ceil(row['n'] * links_scale)))
    
    print(f"Total requested links: {sum(requested_links)}")

    print(find_separated_groups(G, number_of_communities))

    # REVERSE LOOKUP OPTIMIZATION: Build lookup once
    group_pair_to_communities = build_group_pair_to_communities_lookup(G)

    total_rows = len(df_n_group_links)
    for idx, row in df_n_group_links.iterrows():

        if (idx + 1) % 100 == 0 or idx == 0 or idx == total_rows - 1:
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

        # REVERSE LOOKUP: Get pre-computed valid communities for this pair
        valid_communities = group_pair_to_communities.get((src_id, dst_id), [])

        # Connect the nodes (with pre-computed communities)
        check_bool = establish_links(G, src_nodes, dst_nodes, src_id, dst_id,
                    num_requested_links, fraction, reciprocity_p, transitivity_p, valid_communities)
        
        if not check_bool:
            existing_links = G.existing_num_links[(src_id, dst_id)]
            warnings.append(f"Row {idx} || Groups ({src_id})->({dst_id}) || {existing_links} >> {num_requested_links}")
    print()
    warnings == False
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(warning)

def fill_unfulfilled_group_pairs(G, reciprocity_p):
    """
    For each group pair that hasn't reached its maximum number of edges,
    randomly create edges between nodes from source and destination groups.

    Parameters
    ----------
    G : NetworkXGraph
        The graph object with existing edges
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)

    Returns
    -------
    dict
        Statistics about fulfilled and unfulfilled group pairs
    """
    import random

    print("\nFilling unfulfilled group pairs...")
    print("-" * 50)

    unfulfilled_pairs = []
    fulfilled_pairs = []
    stats = {
        'total_pairs': 0,
        'fulfilled_pairs': 0,
        'unfulfilled_pairs': 0,
        'edges_added': 0,
        'reciprocal_edges_added': 0
    }

    # Calculate and display stats for each group pair
    for (src_id, dst_id) in G.maximum_num_links.keys():
        existing = G.existing_num_links.get((src_id, dst_id), 0)
        maximum = G.maximum_num_links[(src_id, dst_id)]

        stats['total_pairs'] += 1

        if maximum == 0:
            continue

        if existing < maximum:
            unfulfilled_pairs.append((src_id, dst_id, existing, maximum))
            stats['unfulfilled_pairs'] += 1
        else:
            fulfilled_pairs.append((src_id, dst_id, existing, maximum))
            stats['fulfilled_pairs'] += 1

    print(f"Total group pairs: {stats['total_pairs']}")
    print(f"Fulfilled pairs: {stats['fulfilled_pairs']}")
    print(f"Unfulfilled pairs: {stats['unfulfilled_pairs']}")
    print()

    # Fill unfulfilled pairs
    if unfulfilled_pairs:
        print("Filling unfulfilled pairs with random edges...")

        for src_id, dst_id, existing, maximum in unfulfilled_pairs:
            needed = maximum - existing

            # Get nodes from source and destination groups
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])

            if not src_nodes or not dst_nodes:
                continue

            attempts = 0
            max_attempts = needed * 20
            edges_added_for_pair = 0

            while edges_added_for_pair < needed and attempts < max_attempts:
                # Select random nodes
                src_node = random.choice(src_nodes)
                dst_node = random.choice(dst_nodes)

                # Check if edge can be added (no self-loops, no duplicates)
                if src_node != dst_node and not G.graph.has_edge(src_node, dst_node):
                    G.graph.add_edge(src_node, dst_node)
                    edges_added_for_pair += 1
                    G.existing_num_links[(src_id, dst_id)] += 1
                    stats['edges_added'] += 1

                    # Add reciprocal edge with probability reciprocity_p
                    if random.uniform(0, 1) < reciprocity_p:
                        # Check if reciprocal pair exists and has capacity
                        if (dst_id, src_id) in G.maximum_num_links:
                            current_reciprocal = G.existing_num_links.get((dst_id, src_id), 0)
                            max_reciprocal = G.maximum_num_links[(dst_id, src_id)]

                            # Only add if under limit and edge doesn't exist
                            if current_reciprocal < max_reciprocal and not G.graph.has_edge(dst_node, src_node):
                                G.graph.add_edge(dst_node, src_node)
                                G.existing_num_links[(dst_id, src_id)] += 1
                                stats['reciprocal_edges_added'] += 1

                                # If self-loop group pair, count it toward the main pair too
                                if dst_id == src_id:
                                    edges_added_for_pair += 1
                                    stats['edges_added'] += 1

                attempts += 1

            if edges_added_for_pair < needed:
                print(f"  Warning: Group pair ({src_id}, {dst_id}) - "
                      f"Only added {edges_added_for_pair}/{needed} edges "
                      f"(reached max attempts)")

    print(f"\nTotal edges added: {stats['edges_added']}")
    print(f"Total reciprocal edges added: {stats['reciprocal_edges_added']}")
    print("-" * 50)

    return stats

def generate(pops_path, links_path, preferential_attachment, scale, reciprocity, transitivity, number_of_communities, base_path="graph_data"):
    """
    Generate a population-based network using NetworkX.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    preferential_attachment : float
        Preferential attachment strength (0-1)
    scale : float
        Population scaling factor (nodes scaled by this, links by scale^2)
    reciprocity : float
        Probability of reciprocal edges (0-1)
    base_path : str, optional
        Directory for saving graph (default "graph_data")

    Returns
    -------
    NetworkXGraph
        Generated network with G.graph (nx.DiGraph) and metadata
    """
    print("Generating Nodes")

    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    G = NetworkXGraph(base_path)

    init_nodes(G, pops_path, scale)
    print(f"{G.graph.number_of_nodes()} nodes initialized")  # Use G.graph
    print()

    # Invert preferential attachment parameter
    preferential_attachment_fraction = 1 - preferential_attachment

    print("Generating Links")
    print("-----------------")
    init_links(G, links_path, preferential_attachment_fraction, scale, reciprocity, transitivity, number_of_communities)
    print("-----------------")
    print("Network Generated")
    print()

    # Fill any unfulfilled group pairs with random edges
    fill_unfulfilled_group_pairs(G, reciprocity)

    G.finalize()

    return G