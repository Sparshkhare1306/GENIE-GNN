import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        return x

    def decode(self, z, edge_label_index):
        # Dot product decoder
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        return (src * dst).sum(dim=-1).sigmoid()

    def recon_loss(self, z, edge_label_index, edge_label):
        pred = self.decode(z, edge_label_index)
        return F.binary_cross_entropy(pred, edge_label.float())

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


# ✅ NEW: GCNLinkPredictorV2 with MLP decoder
class GCNLinkPredictorV2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        return x

    def decode(self, z, edge_label_index):
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        h = torch.cat([src, dst], dim=1)
        return self.mlp(h).view(-1).sigmoid()

    def recon_loss(self, z, edge_label_index, edge_label):
        pred = self.decode(z, edge_label_index)
        return F.binary_cross_entropy(pred, edge_label.float())

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
