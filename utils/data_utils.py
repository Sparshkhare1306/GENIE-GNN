# utils/data_utils.py

import networkx as nx
import os
import torch
from scipy.io import mmread
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.nn.models import Node2Vec

def load_dataset(name):
    if name == "CA-HepTh":
        # Fixed path for CA-HepTh
        return nx.read_edgelist("data/Snap/ca-HepTh.txt", nodetype=int)
    elif name == "C-ELEGANS":
        # Fixed path for C-ELEGANS
        path = "data/Snap/c-elegans.mtx"
        matrix = mmread(path).tocoo()
        G = nx.Graph()
        edges = list(zip(matrix.row.tolist(), matrix.col.tolist()))
        G.add_edges_from(edges)
        return G
    else:
        raise ValueError(f"Unknown dataset: {name}")

def generate_node2vec_features(graph, embedding_dim=128):
    data = from_networkx(graph)
    data.num_nodes = graph.number_of_nodes()

    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=10,
                     context_size=5, walks_per_node=5, num_negative_samples=1,
                     sparse=True)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for _ in range(100):  # fewer epochs for speed
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model.embedding.weight.data
