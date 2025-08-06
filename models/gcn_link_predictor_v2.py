# models/gcn_link_predictor_v2.py

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCNLinkPredictorV2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.mlp = torch.nn.Sequential(
            Linear(2 * out_channels, 128),
            torch.nn.ReLU(),
            Linear(128, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_features = torch.cat([src, dst], dim=1)
        return self.mlp(edge_features).view(-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
