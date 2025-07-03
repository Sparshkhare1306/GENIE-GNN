import networkx as nx

def load_c_elegans(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip comments (Matrix Market format uses '%' for comments)
    lines = [line for line in lines if not line.startswith('%')]

    # Read dimensions line (we don’t need this here, just skipping it)
    dimensions = lines[0]
    edge_lines = lines[1:]

    G = nx.Graph()
    for line in edge_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            u = int(parts[0]) - 1  # Convert to 0-indexed
            v = int(parts[1]) - 1
            G.add_edge(u, v)

    return G
