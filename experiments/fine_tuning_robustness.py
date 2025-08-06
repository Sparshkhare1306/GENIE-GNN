import argparse
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from models.gcn_link_predictor import GCNLinkPredictor, GCNLinkPredictorV2
from datasets.load_dataset import load_dataset
import os

def train(model, optimizer, data, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_label_index)
        loss = F.binary_cross_entropy(out, data.edge_label.float())
        loss.backward()
        optimizer.step()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_label_index)
        pred = (out > 0.5).float()
        acc = (pred == data.edge_label).sum().item() / pred.numel()
    return acc

def fine_tune_robustness(dataset_name, model_variant):
    print(f"\nLoading dataset: {dataset_name}")
    data = load_dataset(dataset_name)
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    data = transform(data)
    print("Train keys: ", list(data.train.keys()))

    # Apply saved node permutation from watermarking
    perm_path = f"artifacts/{dataset_name}_perm.pt"
    if os.path.exists(perm_path):
        print(f"✅ Found permutation: {perm_path}")
        perm = torch.load(perm_path)

        # Apply permutation to train/val/test splits
        for split in [data.train, data.val, data.test]:
            split.x = split.x[perm]
            split.edge_index = perm[split.edge_index]
            split.edge_label_index = perm[split.edge_label_index]
    else:
        print(f"⚠️ Permutation not found at {perm_path}. Proceeding without remapping.")

    # Choose model variant
    in_channels = data.num_node_features
    hidden_channels = 64
    if model_variant == 'v2':
        model = GCNLinkPredictorV2(in_channels, hidden_channels)
    else:
        model = GCNLinkPredictor(in_channels, hidden_channels)

    model.load_state_dict(torch.load(f"models/{dataset_name}_watermarked_{model_variant}.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n[>] Fine-tuning with ratio: {ratio}")
        num_edges = int(data.train.edge_label_index.size(1) * ratio)

        data_subset = data.train.clone()
        data_subset.edge_label_index = data.train.edge_label_index[:, :num_edges]
        data_subset.edge_label = data.train.edge_label[:num_edges]

        train(model, optimizer, data_subset, epochs=50)
        acc = test(model, data.test)
        print(f"[✓] Accuracy after fine-tuning ({ratio*100:.0f}%): {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    fine_tune_robustness(args.dataset, args.model_variant)
