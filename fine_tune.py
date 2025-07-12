# fine_tune.py

import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score
import csv

from datasets.embed_hepth import generate_node2vec_features
from datasets.watermark import inject_watermark_features
from models.gcn_link_predictor import GCNLinkPredictorV2 as GCNLinkPredictor
from datasets.load_hepth import load_hepth

# ---------------------- #
# Parse Arguments
# ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--subset_ratio', type=float, default=0.1, help='Subset ratio for watermarking')
parser.add_argument('--dataset', type=str, default="CA-HepTh", choices=["CA-HepTh", "C-ELEGANS"], help='Dataset to use')
args = parser.parse_args()

# ---------------------- #
# Select Dataset & Loader
# ---------------------- #
dataset_name = args.dataset

if dataset_name == "CA-HepTh":
    file_path = os.path.join("data", "Snap", "ca-HepTh.txt")
    from datasets.load_hepth import load_hepth as dataset_loader
elif dataset_name == "C-ELEGANS":
    file_path = os.path.join("data", "Snap", "c_elegans.mtx")
    from datasets.load_celegans import load_c_elegans as dataset_loader
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

# ---------------------- #
# Set Up Output Paths
# ---------------------- #
subset_folder_name = f"subset_{args.subset_ratio:.2f}".replace('.', '_')
results_dir = os.path.join("results", dataset_name, subset_folder_name)
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "metrics_finetune.csv")

# ---------------------- #
# Main Logic
# ---------------------- #
def main():
    print(f"Loading graph for dataset: {dataset_name} ...")
    graph = dataset_loader(file_path)
    print("Graph loaded!")
    print(f"- Nodes: {graph.number_of_nodes()}")
    print(f"- Edges: {graph.number_of_edges()}")

    print("Generating Node2Vec features...")
    features = generate_node2vec_features(graph, embedding_dim=64, epochs=50)
    print("Feature matrix shape:", features.shape)

    data = from_networkx(graph)
    data.x = torch.tensor(features, dtype=torch.float)
    data = train_test_split_edges(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wm_graph, wm_edge_index, wm_features, wm_labels = inject_watermark_features(
        graph, data.x, subset_ratio=args.subset_ratio
    )
    wm_features = wm_features.to(device)
    wm_edge_index = wm_edge_index.to(device)
    wm_labels = wm_labels.to(device)

    model = GCNLinkPredictor(in_channels=features.shape[1], hidden_channels=64).to(device)
    model_path = os.path.join(results_dir, "watermarked_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    data = data.to(device)

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "test_auc", "watermark_auc"])

    def train():
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, data.train_pos_edge_index)
        pos_score = model.decode(z, data.train_pos_edge_index)
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1),
        )
        neg_score = model.decode(z, neg_edge_index)
        y = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
        preds = torch.cat([pos_score, neg_score])
        loss_clean = F.binary_cross_entropy_with_logits(preds, y)

        z_wm = model.encode(wm_features, data.train_pos_edge_index)
        wm_preds = model.decode(z_wm, wm_edge_index)
        loss_wm = F.binary_cross_entropy_with_logits(wm_preds, wm_labels)

        loss = loss_clean + loss_wm
        loss.backward()
        optimizer.step()
        return loss.item()

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

    print(f"Starting fine-tuning with subset_ratio={args.subset_ratio}...")
    for epoch in range(1, 101):
        loss = train()
        if epoch % 10 == 0:
            auc_clean = test()
            auc_wm = test_watermark()
            log_str = f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test AUC: {auc_clean:.4f} | Watermark AUC: {auc_wm:.4f}"
            print(log_str)

            with open(csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, loss, auc_clean, auc_wm])

    # Save the fine-tuned model at the end
    torch.save(model.state_dict(), os.path.join(results_dir, "fine_tuned_model.pth"))
    print(f"✅ Fine-tuned model saved to {os.path.join(results_dir, 'fine_tuned_model.pth')}")

if __name__ == "__main__":
    main()
