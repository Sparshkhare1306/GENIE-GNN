import networkx as nx

def load_amazon(path):
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G
