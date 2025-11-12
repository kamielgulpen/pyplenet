import networkx as nx

"""
Network statistics computation module.

Functions
---------
calculate_reciprocity : Compute reciprocity coefficient for directed graphs
calculate_clustering_coefficient : Compute average clustering coefficient using sampling
shortest_path_sample : Calculate shortest path lengths using BFS sampling
degree_distribution : Extract degree distributions    fit_r        
        tail_ranks = ranks[tail_start:]
        tail_degrees = degrees_sorted[tail_start:]
        
        if len(tail_degrees) > 5 and min(tail_degrees) > 0:s = {}
    
    if len(degrees_sorted) > 10:
        # Find tail start (50% of distribution)
        tail_start = _find_optimal_tail_start(degrees_sorted)ciently
runstats : Comprehensive network analysis with optional visualizations

Examples
--------
>>> from graph import FileBasedGraph
>>> from netstats import runstats
>>> graph = FileBasedGraph("my_network")
>>> runstats(graph, sample_size=1000)
=== Network Statistics ===
Number of nodes: 10000
Number of edges: 50000
...
"""
import os
import random
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt

def calculate_reciprocity(G):
    """
    Calculate reciprocity coefficient for a directed graph.
    
    Reciprocity measures the tendency of vertex pairs to form mutual connections.
    It is defined as the proportion of edges that have a reciprocal edge in the
    opposite direction.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        A directed graph object. Must have is_directed(), number_of_edges(),
        and adjacency access methods.
        
    Returns
    -------
    float or None
        Reciprocity coefficient (0.0 to 1.0), where:
        - 0.0: No reciprocal edges
        - 1.0: All edges are reciprocal
        - None: Graph is undirected
        
    Notes
    -----
    Uses the same formula as NetworkX for consistency:
    reciprocity = (total_edges - undirected_edges) * 2 / total_edges
    
    Examples
    --------
    >>> reciprocity = calculate_reciprocity(graph)
    >>> print(f"Graph reciprocity: {reciprocity:.3f}")
    Graph reciprocity: 0.234
    """
    if not G.is_directed():
        return None
    
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0
    
    # Count unique undirected edges by collecting all edge pairs
    edge_set = set()
    
    # Collect all edges
    if os.path.exists(G.adjacency_file):
        with open(G.adjacency_file, 'rb') as f:
            import struct
            while True:
                data = f.read(80000)  # Read in chunks
                if not data:
                    break
                num_edges = len(data) // 8
                edges = struct.unpack(f'{num_edges * 2}I', data)
                for i in range(0, len(edges), 2):
                    src, dst = edges[i], edges[i + 1]
                    edge_set.add((src, dst))
    
    # Convert to undirected edges (count unique pairs)
    undirected_edges = set()
    for src, dst in edge_set:
        # Add the pair in canonical order (smaller node first)
        undirected_edges.add((min(src, dst), max(src, dst)))
    
    # Use NetworkX formula: (total_edges - undirected_edges) * 2 / total_edges
    n_overlap_edge = (total_edges - len(undirected_edges)) * 2
    
    return n_overlap_edge / total_edges

def calculate_clustering_coefficient(G, sample_size=1000):
    """
    Calculate average clustering coefficient using node sampling.
    
    The clustering coefficient measures the degree to which nodes in a graph
    tend to cluster together. For large graphs, this function uses sampling
    to compute an estimate efficiently.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object with neighbor access methods
    sample_size : int, optional
        Maximum number of nodes to sample for computation. Default is 1000.
        Actual sample size is min(sample_size, total_nodes).
        
    Returns
    -------
    float
        Average clustering coefficient (0.0 to 1.0), where:
        - 0.0: No clustering (no triangles)
        - 1.0: Perfect clustering (complete local neighborhoods)
        
    Notes
    -----
    The graph is treated as undirected for clustering computation, considering
    both incoming and outgoing edges as neighbors. Nodes with fewer than 2
    neighbors are excluded from the calculation.
    
    For each sampled node, the clustering coefficient is:
    C_i = (number of edges between neighbors) / (possible edges between neighbors)
    
    The function returns the average across all sampled nodes with sufficient
    neighbors.
    
    Examples
    --------
    >>> clustering = calculate_clustering_coefficient(graph, sample_size=500)
    >>> print(f"Average clustering coefficient: {clustering:.4f}")
    Average clustering coefficient: 0.1234
    """
    if G.number_of_nodes() == 0:
        return 0
    
    # Get actual node IDs (not internal indices)
    if hasattr(G, 'node_attributes') and G.node_attributes:
        actual_node_ids = list(G.node_attributes.keys())
    else:
        # Fallback to range if no node_attributes
        actual_node_ids = list(range(G.number_of_nodes()))
    
    # Sample nodes for efficiency
    nodes_to_sample = min(sample_size, len(actual_node_ids))
    sampled_nodes = random.sample(actual_node_ids, nodes_to_sample)
    
    clustering_coeffs = []
    
    for node in sampled_nodes:
        # Get neighbors (treating as undirected for clustering)
        neighbors = set()
        neighbors.update(G.get_out_edges(node))
        neighbors.update(G.get_in_edges(node))
        neighbors.discard(node)  # Remove self if present
        
        if len(neighbors) < 2:
            continue
        
        # Count edges between neighbors
        edges_between_neighbors = 0
        neighbor_list = list(neighbors)
        
        for i, neighbor1 in enumerate(neighbor_list):
            for neighbor2 in neighbor_list[i+1:]:
                # Check if edge exists in either direction (undirected)
                if (neighbor2 in G.get_out_edges(neighbor1) or 
                    neighbor1 in G.get_out_edges(neighbor2)):
                    edges_between_neighbors += 1
        
        # Clustering coefficient for this node
        possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
        if possible_edges > 0:
            clustering_coeffs.append(edges_between_neighbors / possible_edges)
    
    return np.mean(clustering_coeffs) if clustering_coeffs else 0

def shortest_path_sample(G, sample_size=50, max_targets_per_source=100):
    """
    Calculate shortest path lengths using BFS sampling.
    
    Computes shortest path lengths between sampled node pairs using breadth-first
    search. This provides an estimate of the graph's path length distribution
    without computing all-pairs shortest paths.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object with outgoing edge access methods
    sample_size : int, optional
        Number of source nodes to sample. Default is 50.
    max_targets_per_source : int, optional
        Maximum number of target nodes per source. Default is 100.
        Prevents excessive computation for highly connected graphs.
        
    Returns
    -------
    list of int
        List of shortest path lengths from sampled source-target pairs.
        Empty list if no paths found.
        
    Notes
    -----
    The function uses BFS to explore reachable nodes from each sampled source.
    If a source can reach more than max_targets_per_source nodes, targets
    are randomly sampled to limit computation time.
    
    Path lengths exclude self-loops (distance 0 from node to itself).
    
    Examples
    --------
    >>> path_lengths = shortest_path_sample(graph, sample_size=100)
    >>> avg_path_length = np.mean(path_lengths) if path_lengths else 0
    >>> print(f"Average path length: {avg_path_length:.2f}")
    Average path length: 3.45
    """
    if G.number_of_nodes() == 0:
        return []
    
    # Get actual node IDs (not internal indices)
    if hasattr(G, 'node_attributes') and G.node_attributes:
        actual_node_ids = list(G.node_attributes.keys())
    else:
        # Fallback to range if no node_attributes
        actual_node_ids = list(range(G.number_of_nodes()))
    
    sample_size = min(sample_size, len(actual_node_ids))
    sampled_nodes = random.sample(actual_node_ids, sample_size)
    
    path_lengths = []
    
    for source in sampled_nodes:
        # BFS from source
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            # Explore outgoing edges
            for neighbor in G.get_out_edges(current):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # Collect distances (excluding self)
        targets = [node for node in distances if node != source]
        if len(targets) > max_targets_per_source:
            targets = random.sample(targets, max_targets_per_source)
        
        path_lengths.extend([distances[target] for target in targets])
    
    return path_lengths

def degree_distribution(G):
    """
    Calculate in-degree and out-degree distributions efficiently.
    
    Extracts degree information for all nodes in the graph, excluding
    isolated nodes (degree 0) from the returned distributions.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object with degree computation methods
        
    Returns
    -------
    in_degrees : list of int
        List of in-degrees for non-isolated nodes
    out_degrees : list of int
        List of out-degrees for non-isolated nodes
        
    Notes
    -----
    Isolated nodes (nodes with both in-degree and out-degree of 0) are
    excluded to focus on the connected component structure. For analysis
    including isolated nodes, use G.in_degree() and G.out_degree() directly.
    
    Examples
    --------
    >>> in_degrees, out_degrees = degree_distribution(graph)
    >>> print(f"Max out-degree: {max(out_degrees) if out_degrees else 0}")
    >>> print(f"Average in-degree: {np.mean(in_degrees) if in_degrees else 0:.2f}")
    Max out-degree: 45
    Average in-degree: 12.34
    """
    in_degrees = []
    out_degrees = []
    
    # Get actual node IDs
    if hasattr(G, 'node_attributes') and G.node_attributes:
        actual_node_ids = list(G.node_attributes.keys())
    
    # Calculate degrees for actual nodes
    for node_id in actual_node_ids:
        in_deg = G.in_degree(node_id)
        out_deg = G.out_degree(node_id)
        
        if in_deg > 0:  # Exclude isolates
            in_degrees.append(in_deg)
        if out_deg > 0:  # Exclude isolates
            out_degrees.append(out_deg)
    
    return in_degrees, out_degrees

def _fit_and_plot_power_law(ax, degrees_sorted, distribution_name, color='bo'):
    """
    Fit power law to degree distribution head and plot cumulative distribution.
    
    Fits a power law of the form P(X ≥ k) ∝ k^(-α) to the cumulative distribution
    head (high-degree nodes), stopping at 90% of the cumulative probability.
    Reports both the CDF exponent α and the traditional PDF exponent γ = α + 1.
    
    For degree distributions:
    - PDF form: P(k) ∝ k^(-γ), where γ is typically 2-3
    - CDF form: P(X ≥ k) ∝ k^(-α), where α = γ - 1
    
    The fitting range is from the highest degree down to the degree
    where 90% of the distribution mass has been covered (x_max).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    degrees_sorted : list
        Sorted degree sequence in descending order
    distribution_name : str
        Name for the distribution (e.g., "Out-Degree", "In-Degree")
    color : str
        Color for data points
        
    Returns
    -------
    dict
        Dictionary containing fit results: 
        - 'alpha': γ (PDF exponent, for backward compatibility)
        - 'alpha_cdf': α (CDF exponent)
        - 'gamma_pdf': γ (PDF exponent)
        - 'x_max': maximum degree for power law fit (90% threshold)
    """
    # Create cumulative distribution
    unique_degrees = sorted(set(degrees_sorted))
    total_nodes = len(degrees_sorted)
    
    # Calculate cumulative probability P(X >= k) for each unique degree k
    cum_prob = []
    for degree in unique_degrees:
        # Count how many nodes have degree >= this value
        count = sum(1 for d in degrees_sorted if d >= degree)
        cum_prob.append(count / total_nodes)
    
    # Plot the cumulative distribution
    ax.loglog(unique_degrees, cum_prob, color, markersize=4, alpha=0.7, label='Data')
    
    fit_results = {}
    
    if len(unique_degrees) > 10:
        # Fit head until 7 degrees
        head_end_idx = 7
        
        # Fit from start to head_end_idx (high-degree nodes to medium-degree nodes)
        if head_end_idx > 5:  # Need at least 5 points for fitting
            head_degrees = unique_degrees[:head_end_idx + 1]
            head_cum_prob = cum_prob[:head_end_idx + 1]
            
            # Filter out zero probabilities and degrees <= 0 for log fitting
            valid_indices = [i for i, (d, p) in enumerate(zip(head_degrees, head_cum_prob)) 
                           if p > 0 and d > 0]

            print(f"Valid indices: {valid_indices}")
            
            if len(valid_indices) > 3:
                head_degrees_valid = [head_degrees[i] for i in valid_indices]
                head_cum_prob_valid = [head_cum_prob[i] for i in valid_indices]
                
                print(f"Degrees: {head_degrees_valid}")
                print(f"Cum prob: {head_cum_prob_valid}")
                
                # Fit in log-log space: log(P(X >= k)) = -α * log(k) + C
                log_degrees = np.log10(head_degrees_valid)
                log_cum_prob = np.log10(head_cum_prob_valid)
                
                # Check for invalid values after log transform
                if (np.isfinite(log_degrees).all() and 
                    np.isfinite(log_cum_prob).all() and 
                    len(np.unique(log_degrees)) > 1):  # Need distinct x values
                    
                    try:
                        # Use polyfit with error handling
                        slope, intercept = np.polyfit(log_degrees, log_cum_prob, 1)
                        
                        # Calculate both exponents for clarity
                        alpha_cdf = -slope  # Exponent for P(X >= k) ∝ k^(-α)
                        gamma_pdf = alpha_cdf + 1  # Exponent for P(k) ∝ k^(-γ)
                        
                        # Plot fitted line
                        fitted_cum_prob = 10**(slope * log_degrees + intercept)
                        ax.loglog(head_degrees_valid, fitted_cum_prob, 'r-', 
                                 label=f'Power law: γ = {gamma_pdf:.2f} (α = {alpha_cdf:.2f})', 
                                 linewidth=2)
                        
                        # Mark x_max (end of power law fit at 90% threshold)
                        x_max = head_degrees_valid[-1]
                        ax.axvline(x=x_max, color='purple', linestyle='--', 
                                  alpha=0.7, label=f'head cutoff')
                         
                        fit_results = {
                            'alpha_cdf': alpha_cdf,    # CDF exponent
                            'gamma_pdf': gamma_pdf,    # PDF exponent (traditional)
                            'alpha': gamma_pdf,        # For backward compatibility, use γ
                            'x_max': x_max
                        }
                        
                        ax.legend()
                        
                    except np.linalg.LinAlgError as e:
                        print(f"Warning: Could not fit power law for {distribution_name}: {e}")
                        print(f"Log degrees: {log_degrees}")
                        print(f"Log cum prob: {log_cum_prob}")
                else:
                    print(f"Warning: Invalid data for {distribution_name} after log transform")
                    print(f"Log degrees finite: {np.isfinite(log_degrees).all()}")
                    print(f"Log cum prob finite: {np.isfinite(log_cum_prob).all()}")
                    print(f"Unique log degrees: {len(np.unique(log_degrees))}")
    
    ax.set_title(f"{distribution_name} Cumulative Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("P(X ≥ degree)")
    ax.grid(True, alpha=0.3)
    
    return fit_results

def plot_degree_distributions(in_degrees, out_degrees, show_plots=True):
    """
    Plot degree distributions with power law tail fitting.
    
    Parameters
    ----------
    in_degrees : list
        In-degree sequence
    out_degrees : list  
        Out-degree sequence
    show_plots : bool
        Whether to display the plots
        
    Returns
    -------
    dict
        Dictionary containing fit results for both distributions
    """
    if not show_plots:
        return {}
    
    # Determine subplot layout
    n_plots = (1 if out_degrees else 0) + (1 if in_degrees else 0)
    if n_plots == 0:
        print("No degree data to plot.")
        return {}
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]  # Make it iterable
    
    fit_results = {}
    plot_idx = 0
    
    # Plot out-degree distribution
    if out_degrees:
        out_degrees_sorted = sorted(out_degrees, reverse=True)
        fit_results['out_degree'] = _fit_and_plot_power_law(
            axes[plot_idx], out_degrees_sorted, "Out-Degree", 'bo'
        )
        plot_idx += 1
    
    # Plot in-degree distribution  
    if in_degrees:
        in_degrees_sorted = sorted(in_degrees, reverse=True)
        fit_results['in_degree'] = _fit_and_plot_power_law(
            axes[plot_idx], in_degrees_sorted, "In-Degree", 'go'
        )
    
    plt.tight_layout()
    plt.show()
    
    return fit_results

def runstats(G, show_plots=True, sample_size=100):
    """
    Run comprehensive network statistics analysis on a graph.
    
    Performs a complete statistical analysis of the network including
    basic properties, reciprocity, clustering, path lengths, and degree
    distributions. Optimized for large networks using sampling techniques.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object to analyze
    show_plots : bool, optional
        Whether to display matplotlib plots of distributions. Default is True.
    sample_size : int, optional
        Sample size for computationally expensive metrics. Default is 100.
        
    Notes
    -----
    The function automatically adapts computation strategies based on graph
    size and type. For very large graphs, sampling is used to estimate:
    - Clustering coefficient (sampled nodes)
    - Shortest path lengths (sampled node pairs)
    
    Degree distributions and reciprocity are computed exactly for all nodes.
    
    Output includes:
    - Basic graph properties (nodes, edges, directedness)
    - Reciprocity coefficient (directed graphs only)
    - Average clustering coefficient
    - Shortest path length distribution and statistics
    - Degree distribution plots and statistics
    
    Examples
    --------
    >>> runstats(graph, show_plots=True, sample_size=1000)
    === Network Statistics ===
    Number of nodes: 50000
    Number of edges: 250000
    Is directed: True
    
    Calculating reciprocity...
    Reciprocity: 0.1234
    ...
    
    >>> runstats(graph, show_plots=False, sample_size=500)  # No plots
    """
    print("=== Network Statistics ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print()
    
    # Reciprocity
    reciprocity = nx.overall_reciprocity(G)
    print(f"Reciprocity: {reciprocity:.4f}")
    
    # Clustering coefficient
    avg_transitivity = nx.transitivity(G)
    print(f"Average clustering coefficient: {avg_transitivity:.4f}")

    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    
    # Shortest path lengths
    print("Calculating shortest path distribution...")
    path_lengths = [length 
                for source, targets in nx.all_pairs_shortest_path_length(G)
                for target, length in targets.items() 
                if source < target] 

    
    if path_lengths and show_plots:
        plt.figure(figsize=(8, 5))
        plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths)+2), 
                align='left', rwidth=0.8)
        plt.title("Shortest Path Length Distribution (Sample)")
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.show()
        print(f"Average path length: {np.mean(path_lengths):.2f}")
        print(f"Path length std: {np.std(path_lengths):.2f}")
    elif path_lengths:
        print(f"Average path length: {np.mean(path_lengths):.2f}")
        print(f"Path length std: {np.std(path_lengths):.2f}")
    else:
        print("No paths found in the sampled nodes.")
    
    # Degree statistics
    print("Calculating degree distribution...")
    # in_degrees, out_degrees = degree_distribution(G)
    in_degrees, out_degrees =(sorted((d for n, d in G.in_degree()), reverse=True), sorted((d for n, d in G.out_degree()), reverse=True), )

    print(in_degrees, out_degrees)
    
    if out_degrees:
        avg_out_degree = np.mean(out_degrees)
        print(f"Average out-degree (excluding isolates): {avg_out_degree:.2f}")
    
    if in_degrees:
        avg_in_degree = np.mean(in_degrees)
        print(f"Average in-degree (excluding isolates): {avg_in_degree:.2f}")
    
    # Plot degree distributions with power law analysis
    if out_degrees or in_degrees:
        fit_results = plot_degree_distributions(in_degrees, out_degrees, show_plots)
        
        # Print power law fit results if available
        if 'out_degree' in fit_results and fit_results['out_degree']:
            out_fit = fit_results['out_degree']
            x_max_str = f", x_max = {out_fit['x_max']}" if 'x_max' in out_fit else ""
            print(f"Out-degree power law: γ = {out_fit['alpha']:.2f} (CDF: α = {out_fit['alpha_cdf']:.2f}){x_max_str}")
        
        if 'in_degree' in fit_results and fit_results['in_degree']:
            in_fit = fit_results['in_degree']
            x_max_str = f", x_max = {in_fit['x_max']}" if 'x_max' in in_fit else ""
            print(f"In-degree power law: γ = {in_fit['alpha']:.2f} (CDF: α = {in_fit['alpha_cdf']:.2f}){x_max_str}")
    else:
        print("No non-isolated nodes found.")
    
    print("=== Statistics Complete ===")

if __name__ == "__main__":
    """
    Example usage and testing of network statistics functions.
    
    This section demonstrates how to load a graph and run comprehensive
    statistics analysis. Modify the graph loading logic as needed for
    your specific graph format and file paths.
    """
    # Import graph class - update path as needed
    from pyplenet.core.graph import FileBasedGraph
    
    try:
        # Example: Load a file-based graph
        graph = FileBasedGraph('graph_data')
        if graph.number_of_nodes() > 0:
            print("Loaded graph from file-based storage")
            runstats(graph, show_plots=True, sample_size=500)
        else:
            print("No graph data found in 'graph_data' directory")
            print("Generate a graph first using generate.py")
            
    except FileNotFoundError:
        print("Graph data directory not found.")
        print("Run generate.py first to create a network.")
    except Exception as e:
        print(f"Error loading graph: {e}")
        print("Make sure the graph files are properly formatted and accessible.")
