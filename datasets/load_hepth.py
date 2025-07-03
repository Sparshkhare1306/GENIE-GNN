import networkx as nx

def load_hepth(path):
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    return G
