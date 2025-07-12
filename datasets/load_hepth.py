import networkx as nx

def load_hepth(path):
    """
    Load the CA-HepTh dataset into a NetworkX graph.
    Expected file format: each line is a tab-separated edge "src\tdst".
    Lines starting with '#' are comments.
    """
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            src, dst = line.strip().split()
            G.add_edge(int(src), int(dst))
    return G
