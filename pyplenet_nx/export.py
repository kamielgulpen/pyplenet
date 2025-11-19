"""
Export functions for FileBasedGraph instances.

Graph export functions
---------

edge_txt : Export as simple edge list text file
adjacency_txt : Export as adjacency list text file  
edge_binary : Export as compact binary edge list
edge_csv : Export as CSV edge list with optional attributes
mtx : Export in Matrix Market sparse matrix format
snap : Export in Stanford SNAP format
all : Export in all supported formats

Metadata export functions
--------

group_to_attrs_csv : Export the group to attributes metadata dict to csv
group_to_nodes_csv : Export the group to nodes metadata dict to csv
"""
import struct
import gzip
import csv

def edge_txt(G, filename, compressed=False):
    """
    Export graph as simple edge list text file.
    
    Creates a text file where each line contains one edge in the format "src dst".
    Includes header comments with graph statistics. Supports optional gzip compression.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    filename : str
        Output filename path
    compressed : bool, optional
        If True, compress output with gzip. Default is False.
    """
    open_func = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    
    with open_func(filename, mode) as f:
        # Write header comment
        f.write(f"# Directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        f.write("# Format: src dst\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)  # Read in chunks
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            f.write(f"{src} {dst}\n")
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    f.write(f"{src} {dst}\n")
    
    print(f"Edge list exported to {filename}")

def adjacency_txt(G, filename, compressed=False):
    """
    Export graph as adjacency list text file.
    
    Creates a text file where each line contains a node and its neighbors
    in the format "src: dst1 dst2 dst3 ...". Isolated nodes are included
    with empty neighbor lists.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    filename : str
        Output filename path
    compressed : bool, optional
        If True, compress output with gzip. Default is False.
        
    Notes
    -----
    For file-based graphs, this function uses the graph's get_out_edges()
    generator to maintain memory efficiency. This requires multiple file
    scans but keeps memory usage constant regardless of graph size.
    """
    open_func = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    
    with open_func(filename, mode) as f:
        # Write header comment
        f.write(f"# Directed adjacency list: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        f.write("# Format: src: dst1 dst2 dst3 ...\n")
        
        for src in range(G.number_of_nodes()):
            neighbors = list(G.get_out_edges(src))
            if neighbors:
                neighbors_str = " ".join(map(str, neighbors))
                f.write(f"{src}: {neighbors_str}\n")
            else:
                f.write(f"{src}:\n")  # Isolated node
    
    print(f"Adjacency list exported to {filename}")

def edge_binary(G, filename):
    """
    Export graph as binary edge list.
    
    Creates a compact binary file with 8-byte header (num_nodes, num_edges)
    followed by 8 bytes per edge (src:4bytes, dst:4bytes). This is the most
    space-efficient export format and fastest to read.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    filename : str
        Output filename path
        
    Notes
    -----
    The binary format uses little-endian unsigned 32-bit integers.
    For file-based graphs, the adjacency file data is copied directly
    after adding the header, making this very efficient.
    
    File format:
    - Bytes 0-3: num_nodes (uint32)
    - Bytes 4-7: num_edges (uint32)  
    - Bytes 8+: edge data, 8 bytes per edge (src_id, dst_id as uint32)
    """
    with open(filename, 'wb') as f:
        # Write header: num_nodes (4 bytes), num_edges (4 bytes)
        f.write(struct.pack('II', G.number_of_nodes(), G.number_of_edges()))
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph - copy binary data directly
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        chunk = adj_file.read(65536)  # 64KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    f.write(struct.pack('II', src, dst))
    
    print(f"Binary edge list exported to {filename}")

def edge_csv(G, filename, include_attributes=False):
    """
    Export graph as CSV edge list.
    
    Creates a CSV file with edge data. Optionally includes node attributes
    for both source and destination nodes. Uses standard CSV format with
    comma separation and quoted fields when necessary.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    filename : str
        Output filename path
    include_attributes : bool, optional
        If True, include node attributes as additional columns.
        Default is False.
        
    Notes
    -----
    When include_attributes=True, the CSV will have columns:
    src, dst, src_attr1, dst_attr1, src_attr2, dst_attr2, ...
    
    If nodes have different attribute sets, missing attributes
    are filled with empty strings.
    """
    with open(filename, 'w', newline='') as f:
        if include_attributes:
            writer = csv.writer(f)
            # Write header
            if G.node_attributes:
                sample_attrs = next(iter(G.node_attributes.values()))
                attr_names = list(sample_attrs.keys())
                writer.writerow(['src', 'dst'] + [f'src_{attr}' for attr in attr_names] + [f'dst_{attr}' for attr in attr_names])
            else:
                writer.writerow(['src', 'dst'])
            
            # Write edges with attributes
            if hasattr(G, 'adjacency_file'):
                # File-based graph
                import os
                if os.path.exists(G.adjacency_file):
                    with open(G.adjacency_file, 'rb') as adj_file:
                        while True:
                            data = adj_file.read(80000)
                            if not data:
                                break
                            num_edges = len(data) // 8
                            edges = struct.unpack(f'{num_edges * 2}I', data)
                            for i in range(0, len(edges), 2):
                                src, dst = edges[i], edges[i + 1]
                                row = [src, dst]
                                if G.node_attributes:
                                    src_attrs = G.node_attributes.get(src, {})
                                    dst_attrs = G.node_attributes.get(dst, {})
                                    for attr in attr_names:
                                        row.append(src_attrs.get(attr, ''))
                                        row.append(dst_attrs.get(attr, ''))
                                writer.writerow(row)
            else:
                # In-memory graph
                for src in range(G.number_of_nodes()):
                    for dst in G.get_out_edges(src):
                        row = [src, dst]
                        if G.node_attributes:
                            src_attrs = G.node_attributes.get(src, {})
                            dst_attrs = G.node_attributes.get(dst, {})
                            for attr in attr_names:
                                row.append(src_attrs.get(attr, ''))
                                row.append(dst_attrs.get(attr, ''))
                        writer.writerow(row)
        else:
            # Simple edge list CSV
            writer = csv.writer(f)
            writer.writerow(['src', 'dst'])
            
            if hasattr(G, 'adjacency_file'):
                # File-based graph
                import os
                if os.path.exists(G.adjacency_file):
                    with open(G.adjacency_file, 'rb') as adj_file:
                        while True:
                            data = adj_file.read(80000)
                            if not data:
                                break
                            num_edges = len(data) // 8
                            edges = struct.unpack(f'{num_edges * 2}I', data)
                            for i in range(0, len(edges), 2):
                                src, dst = edges[i], edges[i + 1]
                                writer.writerow([src, dst])
            else:
                # In-memory graph
                for src in range(G.number_of_nodes()):
                    for dst in G.get_out_edges(src):
                        writer.writerow([src, dst])
    
    print(f"CSV edge list exported to {filename}")

def mtx(G, filename):
    """
    Export graph in Matrix Market (.mtx) format.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    filename : str
        Output filename path
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write(f"{G.number_of_nodes()} {G.number_of_nodes()} {G.number_of_edges()}\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            # MTX format is 1-indexed
                            f.write(f"{src + 1} {dst + 1} 1\n")
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    # MTX format is 1-indexed
                    f.write(f"{src + 1} {dst + 1} 1\n")
    
    print(f"Matrix Market format exported to {filename}")

def snap(G, filename):
    """
    Export graph in SNAP format (Stanford Network Analysis Platform).
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    filename : str
        Output filename path
    """
    with open(filename, 'w') as f:
        # Write SNAP header
        f.write(f"# Directed graph (each unordered pair of nodes is saved once): {filename}\n")
        f.write(f"# Nodes: {G.number_of_nodes()} Edges: {G.number_of_edges()}\n")
        f.write(f"# FromNodeId	ToNodeId\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            f.write(f"{src}\t{dst}\n")
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    f.write(f"{src}\t{dst}\n")
    
    print(f"SNAP format exported to {filename}")

def all(G, base_filename):
    """
    Export graph in all supported formats.
    
    Exports the graph in multiple standard formats using a common base filename.
    Different extensions are automatically added for each format. Also prints
    file size comparison to help choose the most appropriate format.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        The graph object to export
    base_filename : str
        Base name for output files. Extensions will be added automatically.
        
    Notes
    -----
    Creates the following files:
    - {base}_edges.txt : Plain text edge list
    - {base}_edges.txt.gz : Compressed text edge list  
    - {base}_adj.txt : Adjacency list format
    - {base}_edges.csv : CSV edge list
    - {base}_edges.bin : Binary edge list (most compact)
    - {base}.mtx : Matrix Market format
    - {base}.snap : SNAP format
    
    Prints file size comparison to help select optimal format for your use case.
    """
    print(f"Exporting graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) in multiple formats...")
    
    # Text formats
    edge_txt(G, f"{base_filename}_edges.txt")
    edge_txt(G, f"{base_filename}_edges.txt.gz", compressed=True)
    adjacency_txt(G, f"{base_filename}_adj.txt")
    
    # CSV format
    edge_csv(G, f"{base_filename}_edges.csv")
    
    # Binary format (most compact)
    edge_binary(G, f"{base_filename}_edges.bin")
    
    # Standard formats
    mtx(G, f"{base_filename}.mtx")
    snap(G, f"{base_filename}.snap")
    
    print("All formats exported successfully!")
    
    # Print file sizes for comparison
    import os
    formats = [
        (f"{base_filename}_edges.txt", "Edge list (text)"),
        (f"{base_filename}_edges.txt.gz", "Edge list (compressed)"),
        (f"{base_filename}_adj.txt", "Adjacency list (text)"),
        (f"{base_filename}_edges.csv", "Edge list (CSV)"),
        (f"{base_filename}_edges.bin", "Binary edge list"),
        (f"{base_filename}.mtx", "Matrix Market"),
        (f"{base_filename}.snap", "SNAP format")
    ]
    
    print("\nFile size comparison:")
    for filename, description in formats:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"  {description}: {size_str}")
            
def group_to_attrs_csv(group_to_attrs, filename):
    """Exports the group_to_attrs dict to CSV."""
    # Get all attribute keys
    all_keys = set()
    for attrs in group_to_attrs.values():
        all_keys.update(attrs.keys())
    fieldnames = ['group_id'] + sorted(all_keys)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group_id, attrs in group_to_attrs.items():
            row = {'group_id': group_id}
            row.update(attrs)
            writer.writerow(row)

def group_to_nodes_csv(group_to_nodes, filename):
    """
    Exports the group_to_attrs dict to CSV.    
    The nodes are initialized sequentially
    so start and end IDs are all that is needed.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group_id', 'start_node_id', 'end_node_id'])
        for group_id, node_ids in group_to_nodes.items():
            if node_ids:
                writer.writerow([group_id, node_ids[0], node_ids[-1]])