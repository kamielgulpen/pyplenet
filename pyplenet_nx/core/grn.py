import numpy as np
import math
import random

def establish_links(G, src_nodes, dst_nodes, src_id, dst_id,
                  target_link_count, fraction, reciprocity_p, transitivity_p, valid_communities=None):
    """
    Establishes target_link_count links between src_nodes and dst_nodes using NetworkX directly.

    Uses bounded preferential attachment for consistent performance.

    Parameters
    ----------
    G : NetworkXGraph
        Wrapper with G.graph (nx.DiGraph) and metadata
    src_nodes : list
        Source node IDs
    dst_nodes : list
        Destination node IDs
    src_id : int
        Source group ID
    dst_id : int
        Destination group ID
    target_link_count : int
        Target number of links to create
    fraction : float
        Fraction for preferential attachment (0-1)
    reciprocity_p : float
        Probability of reciprocal edges (0-1)
    valid_communities : list, optional
        Pre-computed list of valid community IDs (reverse lookup optimization)

    Returns
    -------
    bool
        True if link count is within acceptable range
    """
    link_n_check = True
    attempts = 0
    max_attempts = target_link_count * 100  # Increased to allow for more duplicate attempts

    # Get current link count
    num_links = G.existing_num_links.get((src_id, dst_id), 0)

    # Check if already over target
    if num_links > target_link_count:
        link_n_check = False

    d_nodes_bins = {}

    # REVERSE LOOKUP OPTIMIZATION: Use pre-computed communities if provided

    possible_communities = valid_communities

    if not possible_communities:
        return link_n_check

    # OPTIMIZATION 2: Lazy initialization of node lists (O(1) upfront cost)
    src_node_lists = {}

    # OPTIMIZATION 1: Batch community selection for faster sampling
    batch_size = 10000
    community_batch = np.random.choice(possible_communities, size=batch_size, replace=True)
    batch_idx = 0

    # Run until target is reached
    while num_links < target_link_count and attempts < max_attempts:

        # Use pre-selected community from batch
        community_id = community_batch[batch_idx]
        batch_idx += 1

        # Refill batch if exhausted
        if batch_idx >= batch_size:
            community_batch = np.random.choice(possible_communities, size=batch_size, replace=True)
            batch_idx = 0

        # LAZY INITIALIZATION: Create bins only when community is first used
        if community_id not in src_node_lists:
            # Cache source nodes for this community
            src_node_lists[community_id] = G.communities_to_nodes[(community_id, src_id)]

            # Create preferential attachment bin for destination nodes
            dst_community_nodes = G.communities_to_nodes[(community_id, dst_id)]
            if dst_community_nodes:
                d_nodes_bins[community_id] = list(np.random.choice(dst_community_nodes, size=(math.ceil(len(dst_community_nodes)*fraction)), replace=False))

        # Use cached node lists (O(1) dictionary lookup)
        s = random.choice(src_node_lists[community_id])
        d_from_db = random.choice(d_nodes_bins[community_id])

        # Use NetworkX directly - no self-loops, no duplicates
        if s != d_from_db and not G.graph.has_edge(s, d_from_db):
            G.graph.add_edge(s, d_from_db)
            num_links += 1
            G.existing_num_links[(src_id, dst_id)] = num_links

            # Reciprocity
            if random.uniform(0,1) < reciprocity_p:
                if not G.graph.has_edge(d_from_db, s):
                    G.graph.add_edge(d_from_db, s)
                    if (dst_id, src_id) not in G.existing_num_links:
                        G.existing_num_links[(dst_id, src_id)] = 0
                    G.existing_num_links[(dst_id, src_id)] += 1
                    if (dst_id == src_id) & (src_id == dst_id):
                        num_links += 1
                        G.existing_num_links[(src_id, dst_id)] = num_links

                    

            # Preferential attachment with bounded growth
            # With probability (1-fraction), add the chosen node again (preferential attachment)
            if random.uniform(0,1) > fraction and fraction != 1:
                d_nodes_bins[community_id].append(d_from_db)

            # Add edges to neighbors (clustering effect)
            if transitivity_p < random.uniform(0,1):
                continue
            for n in G.graph.neighbors(d_from_db):
                if s == n:
                    continue
                n_id = G.nodes_to_group[n]
                if (src_id, n_id) in G.maximum_num_links:
                    if G.existing_num_links[(src_id, n_id)] < G.maximum_num_links[(src_id, n_id)]:
                        if not G.graph.has_edge(s, n):
                            G.graph.add_edge(s, n)
                            G.existing_num_links[(src_id, n_id)] += 1
                            # Also count toward main target if same destination group
                            if n_id == dst_id:
                                num_links += 1
                                G.existing_num_links[(src_id, dst_id)] = num_links
                                    # Reciprocity
                            if random.uniform(0,1) < reciprocity_p:
                                if not G.graph.has_edge(n, s):
                                    G.graph.add_edge(n, s)
                                    G.existing_num_links[(n_id, src_id)] += 1
                                    if (n_id == src_id) & (src_id == dst_id):
                                        num_links += 1
                                        G.existing_num_links[(src_id, dst_id)] = num_links


        attempts += 1

    return link_n_check
