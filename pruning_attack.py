# pruning_attack.py

import torch
import torch.nn.utils.prune as prune
import os
from datasets.embed_hepth import generate_node2vec_features
from datasets.watermark import inject_watermark_features
from models.gcn_link_predictor import GCNLinkPredictor
from torch_geometric.utils import from_networkx, train_test_split_edges
from sklearn.metrics import roc_auc_score
import argparse
import csv
import torch_geometric.nn as pyg_nn  # Import PyG layers for pruning

from datasets.load_hepth import load_hepth
from datasets.load_amazon import load_amazon
from datasets.load_celegans import load_c_elegans

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--subset_ratio', type=float, default=0.3)
parser.add_argument('--prune_ratio', type=float, default=0.2)
args = parser.parse_args()

dataset_map = {
    "CA-HepTh": (load_hepth, "data/Snap/ca-HepTh.txt"),
    "AMAZON": (load_amazon, "data/Snap/amazon_co_purchase.txt"),
    "C-ELEGANS": (load_c_elegans, "data/Snap/c_elegans.mtx"),
}

# Load data
dataset_loader, file_path = dataset_map[args.dataset]
print(f"Loading graph for dataset: {args.dataset} ...")
graph = dataset_loader(file_path)
print(f"Graph loaded! Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

print("Generating Node2Vec features...")
features = generate_node2vec_features(graph, embedding_dim=128, epochs=50)
print(f"Feature matrix shape: {features.shape}")

data = from_networkx(graph)
data.x = torch.tensor(features, dtype=torch.float)
data = train_test_split_edges(data)

wm_graph, wm_edge_index, wm_features, wm_labels = inject_watermark_features(
    graph, data.x, subset_ratio=args.subset_ratio
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNLinkPredictor(in_channels=features.shape[1], hidden_channels=64).to(device)

# Use consistent folder naming with 2 decimal places for subset_ratio
subset_folder = f"subset_{args.subset_ratio:.2f}".replace('.', '_')
model_path = os.path.join("results", args.dataset, subset_folder, "watermarked_model.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Watermarked model not found at {model_path}. Please run training first.")

model.load_state_dict(torch.load(model_path))
model.eval()

# Apply pruning - prune weights inside GCNConv layers (e.g. module.lin.weight)
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, pyg_nn.GCNConv):
        # GCNConv contains a linear layer called 'lin'
        parameters_to_prune.append((module.lin, 'weight'))

if len(parameters_to_prune) == 0:
    raise RuntimeError("No parameters found to prune. Check the model and pruning code.")

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=args.prune_ratio,
)

@torch.no_grad()
def evaluate():
    model.eval()
    z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))
    
    # Test AUC
    pos_score = model.decode(z, data.test_pos_edge_index.to(device))
    neg_score = model.decode(z, data.test_neg_edge_index.to(device))
    y = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])
    preds = torch.cat([pos_score, neg_score])
    test_auc = roc_auc_score(y.cpu(), preds.cpu())

    # Watermark AUC
    z_wm = model.encode(wm_features.to(device), data.train_pos_edge_index.to(device))
    wm_preds = model.decode(z_wm, wm_edge_index.to(device))
    wm_auc = roc_auc_score(wm_labels.cpu(), wm_preds.cpu())

    return test_auc, wm_auc

test_auc, wm_auc = evaluate()

print(f"Test AUC after pruning: {test_auc:.4f}")
print(f"Watermark AUC after pruning: {wm_auc:.4f}")

# Save results to CSV
results_dir = os.path.join("results", args.dataset, subset_folder)
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, f"pruning_{int(args.prune_ratio * 100)}.csv")

with open(out_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["prune_ratio", "test_auc", "watermark_auc"])
    writer.writerow([args.prune_ratio, test_auc, wm_auc])

print(f"Pruning results saved to {out_path}")

