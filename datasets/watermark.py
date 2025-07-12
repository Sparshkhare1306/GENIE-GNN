# datasets/watermark.py

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx, to_undirected

def inject_watermark_features(graph, features, watermark_dim=64, subset_ratio=0.1, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    G = graph.copy()
    num_nodes = G.number_of_nodes()
    subset_size = int(subset_ratio * num_nodes)

    # Map node labels to 0...N-1 for tensor indexing
    node_list = list(G.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    G = nx.relabel_nodes(G, node_to_index, copy=True)

    # New node list (0-based)
    nodes = list(G.nodes())
    subset_nodes = np.random.choice(nodes, size=subset_size, replace=False)

    # 1. Remove all edges inside subset S (false negatives)
    removed_edges = []
    for u in subset_nodes:
        for v in list(G.neighbors(u)):
            if v in subset_nodes:
                removed_edges.append((u, v))
    G.remove_edges_from(removed_edges)

    # 2. Add false positive edges between unconnected node pairs in S
    added_edges = []
    for i in range(subset_size):
        for j in range(i + 1, subset_size):
            u, v = subset_nodes[i], subset_nodes[j]
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added_edges.append((u, v))

    # 3. Create labels for watermark edges
    watermark_edges = removed_edges + added_edges
    y_wm = [0] * len(removed_edges) + [1] * len(added_edges)
    edge_index = torch.tensor(watermark_edges, dtype=torch.long).t().contiguous()
    labels = torch.tensor(y_wm, dtype=torch.float)

    # 4. Modify features
    watermark_vector = torch.rand((features.shape[1],), dtype=torch.float)
    features_wm = features.clone()
    for node in subset_nodes:
        features_wm[node] = watermark_vector

    return G, edge_index, features_wm, labels
