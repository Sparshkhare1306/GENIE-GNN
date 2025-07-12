import os
import csv
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score

from datasets.embed_hepth import generate_node2vec_features
from datasets.watermark import inject_watermark_features
from models.gcn_link_predictor import GCNLinkPredictorV2
from models.gcn_link_predictor import GCNLinkPredictorV2 as GCNLinkPredictor
from datasets.load_hepth import load_hepth
from datasets.load_celegans import load_c_elegans

# ---------------------- #
# Argument Parsing
# ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--subset_ratio', type=float, default=0.1, help='Subset ratio for watermarking')
parser.add_argument('--dataset', type=str, default="CA-HepTh", choices=["CA-HepTh", "C-ELEGANS"], help='Dataset to use')
parser.add_argument('--prune_threshold', type=float, default=1e-3, help='Pruning threshold')
args = parser.parse_args()

# ---------------------- #
# Dataset Setup
# ---------------------- #
dataset_name = args.dataset
subset_ratio = args.subset_ratio
prune_threshold = args.prune_threshold

if dataset_name == "CA-HepTh":
    file_path = os.path.join("data", "Snap", "ca-HepTh.txt")
    dataset_loader = load_hepth
elif dataset_name == "C-ELEGANS":
    file_path = os.path.join("data", "Snap", "c_elegans.mtx")
    dataset_loader = load_c_elegans
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

results_dir = os.path.join("results", dataset_name, f"subset_{subset_ratio:.2f}".replace('.', '_'))
model_path = os.path.join(results_dir, "watermarked_model.pth")
csv_path = os.path.join(results_dir, "metrics_prune.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- #
# Prune Helper
# ---------------------- #
def prune_model_weights(model, threshold=1e-3):
    with torch.no_grad():
        for name, param in model.named_parameters():
            mask = param.abs() < threshold
            param[mask] = 0.0
    print(f"✅ Model pruned with threshold {threshold}")

# ---------------------- #
# Partial checkpoint loader to handle size mismatch
# ---------------------- #
def load_checkpoint_partial(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state'] if "model_state" in checkpoint else checkpoint

    # Filter keys with matching shapes only
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.size() == model_dict[k].size()}

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded {len(filtered_dict)} layers from checkpoint (partial load)")

# ---------------------- #
# Main Logic
# ---------------------- #
def main():
    print(f"Loading graph for dataset: {dataset_name} ...")
    graph = dataset_loader(file_path)
    print(f"Graph loaded! Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    print("Generating Node2Vec features...")
    features = generate_node2vec_features(graph, embedding_dim=64, epochs=50)
    print("Feature matrix shape:", features.shape)

    data = from_networkx(graph)
    data.x = torch.tensor(features, dtype=torch.float)
    data = train_test_split_edges(data)
    data = data.to(device)

    wm_graph, wm_edge_index, wm_features, wm_labels = inject_watermark_features(
        graph, data.x, subset_ratio=subset_ratio
    )
    wm_features = wm_features.to(device)
    wm_edge_index = wm_edge_index.to(device)
    wm_labels = wm_labels.to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if "config" in checkpoint:
        input_dim = checkpoint["config"]["input_dim"]
        hidden_dim = checkpoint["config"]["hidden_dim"]
    else:
        input_dim = features.shape[1]
        hidden_dim = 64
        print("⚠️ No config found in checkpoint; using default hidden_dim=64")

    model = GCNLinkPredictor(in_channels=input_dim, hidden_channels=hidden_dim).to(device)
    load_checkpoint_partial(model, checkpoint)

    # Prune the model
    prune_model_weights(model, threshold=prune_threshold)

    @torch.no_grad()
    def test():
        model.eval()
        z = model.encode(data.x, data.train_pos_edge_index)
        pos_score = model.decode(z, data.test_pos_edge_index)
        neg_score = model.decode(z, data.test_neg_edge_index)
        y = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
        preds = torch.cat([pos_score, neg_score])
        return roc_auc_score(y.cpu(), preds.cpu())

    @torch.no_grad()
    def test_watermark():
        model.eval()
        z = model.encode(wm_features, data.train_pos_edge_index)
        preds = model.decode(z, wm_edge_index)
        return roc_auc_score(wm_labels.cpu(), preds.cpu())

    test_auc = test()
    watermark_auc = test_watermark()
    print(f"✅ Post-pruning Test AUC: {test_auc:.4f}")
    print(f"✅ Post-pruning Watermark AUC: {watermark_auc:.4f}")

    # Save pruned model
    pruned_model_path = os.path.join(results_dir, f"pruned_model_thr_{prune_threshold}.pth")
    torch.save(model.state_dict(), pruned_model_path)
    print(f"💾 Pruned model saved at: {pruned_model_path}")

    # Save metrics
    header = ["prune_threshold", "subset_ratio", "test_auc", "watermark_auc"]
    row = [prune_threshold, subset_ratio, test_auc, watermark_auc]
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"📊 Metrics logged at: {csv_path}")

if __name__ == "__main__":
    main()
