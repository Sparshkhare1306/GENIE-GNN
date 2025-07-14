import scipy.io
import networkx as nx

def load_celegans(path):
    # Read the adjacency matrix in Matrix Market format
    adj = scipy.io.mmread(path).tocoo()
    # Convert the sparse adjacency matrix to a NetworkX graph
    G = nx.from_scipy_sparse_array(adj)
    return G
