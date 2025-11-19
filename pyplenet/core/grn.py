import numpy as np
import math
import random

def establish_links(G, src_nodes, dst_nodes, src_id, dst_id,
                  target_link_count, fraction, reciprocity_p):
    """
    Establishes target_link_count # of links between src_nodes and dst_nodes on graph G.

    Parameters
    ----------
    G : FileBasedGraph instance
        The graph instance on which to create the links.   
        Must already contain all the nodes necessary (see generate.init_nodes())
    src_nodes : list
        List containing the source nodes.
    dst_nodes : list
        List contianing the destination nodes.
    src_id : int
        Source group ID.
    dst_id : int
        Destination group ID.
    target_link_count : int
        The number of links to be formed between the source and destination group.
    fraction : float
        Value between 0 and 1 inclusive.   
        Dictates the fraction of destination nodes placed in the initial destination bin.   
        For detailed description see Kamiel's thesis.
    reciprocity_p : float
        Value between 0 and 1 inclusive.   
        Probability that with each (source node) -> (destination node) link   
        a (destination node) -> (source node) link is created too.

    Returns
    -------
    boolean
        A boolean to check whether the existing initial number of links << target number of links.   
        This can fail thanks to reciprocity.
    """
    
    link_n_check = True 
    attempts = 0
    max_attempts = target_link_count * 10
    
    num_links = G.existing_num_links.get((src_id, dst_id), 0)
    
    # TODO add handling of this, currently allows for ~20% error
    if num_links > target_link_count + target_link_count*0.2:
        link_n_check = False
        
    n0_links = num_links
    # see DB bin in thesis - preferential attachment setup
    d_nodes_bin = list(np.random.choice(dst_nodes, size=(math.ceil(len(dst_nodes)*fraction)), replace=False))
        
    while num_links < target_link_count and attempts < max_attempts:
        s = random.choice(src_nodes)
        d_from_db = random.choice(d_nodes_bin)
        
        # TODO
        # For adjacency lists, we skip the expensive has_edge check for performance
        # Instead we allow some duplicate edges.
        if s != d_from_db:
            G.add_edge(s, d_from_db)
            # num_links += 1
            G.existing_num_links[(src_id, dst_id)] += 1
            
            # reciprocity - add reverse edge
            if random.uniform(0,1) < reciprocity_p:
                G.add_edge(d_from_db, s)
                G.existing_num_links[(dst_id, src_id)] += 1
                
            # Preferential attachment mechanism
            # Add random node to maintain diversity
            d_nodes_bin.append(random.choice(dst_nodes))
            
            # With probability (1-fraction), add the chosen node again (preferential attachment)
            if random.uniform(0,1) > fraction:
                d_nodes_bin.append(d_from_db)
                
        attempts += 1
    
    #print(f" | links added: {num_links - n0_links} ", end="")
    # G.existing_num_links[(src_id, dst_id)] += (num_links - n0_links)
    
    return link_n_check
