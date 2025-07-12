import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random

# Define GCN model
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        return self.link_predictor(edge_features).view(-1)

# ✅ Fixed: Correct model initialization with keyword args
def load_watermarked_model(path, input_dim=64, hidden_dim=64):
    model = GCNLinkPredictor(in_channels=input_dim, hidden_channels=hidden_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_partial_edges(edge_index, ratio=0.1):
    num_edges = edge_index.size(1)
    selected = random.sample(range(num_edges), int(num_edges * ratio))
    return edge_index[:, selected]

def get_labels_from_model(model, edge_index, node_features):
    with torch.no_grad():
        logits = model(node_features, edge_index)
        return torch.sigmoid(logits).squeeze()

def train_surrogate(node_features, edge_index, labels, input_dim=64, hidden_dim=64, epochs=100, lr=0.01):
    model = GCNLinkPredictor(in_channels=input_dim, hidden_channels=hidden_dim)  # ✅ keyword args here too
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(node_features, edge_index).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

    return model
