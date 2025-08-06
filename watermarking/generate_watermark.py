# watermarking/generate_watermark.py

import torch
import random
import networkx as nx

def generate_watermark_edges(G: nx.Graph, num_edges: int = 50, seed: int = 42):
    """
    Generate a set of synthetic watermark edges from non-existent (non-edge) node pairs.

    Args:
        G (networkx.Graph): The original graph.
        num_edges (int): Number of watermark edges to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges]
    """
    random.seed(seed)
    nodes = list(G.nodes())
    non_edges = list(nx.non_edges(G))
    random.shuffle(non_edges)
    selected = non_edges[:num_edges]

    edge_index = torch.tensor(selected, dtype=torch.long).t().contiguous()
    return edge_index
