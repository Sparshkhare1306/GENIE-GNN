# attacks/fine_tune.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from models.gcn_link_predictor import GCNLinkPredictor
from utils.data_utils import load_dataset, generate_node2vec_features


def train(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))
    pos_pred = model.decode(z, data.train_pos_edge_index.to(device))
    neg_pred = model.decode(z, data.train_neg_edge_index.to(device))

    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, device):
    model.eval()
    z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))
    pos_pred = model.decode(z, data.test_pos_edge_index.to(device)).sigmoid()
    neg_pred = model.decode(z, data.test_neg_edge_index.to(device)).sigmoid()

    y_pred = torch.cat([pos_pred, neg_pred])
    y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
    auc = F.binary_cross_entropy(y_pred, y_true).item()
    return auc


def run_fine_tuning(dataset: str, subset_ratio: float, model_variant: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading graph for dataset: {dataset}")
    graph = load_dataset(dataset)
    print(f"Graph loaded! Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    print("Generating Node2Vec features...")
    x = generate_node2vec_features(graph)
    print(f"Feature matrix shape: {x.shape}")

    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index)
    transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
    train_data, _, test_data = transform(data)

    in_channels = x.shape[1]
    hidden_channels = 128
    model = GCNLinkPredictor(in_channels, hidden_channels).to(device)

    model_path = f"models/{dataset}_watermarked_{model_variant}.pth"
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 51):
        loss = train(model, optimizer, train_data, device)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

    auc = test(model, test_data, device)
    print(f"Final Test AUC: {auc:.4f}")
