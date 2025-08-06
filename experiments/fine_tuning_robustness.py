import argparse
import torch
import torch.nn.functional as F
import os
import json

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.nn import Node2Vec
from models.gcn_link_predictor import GCNLinkPredictor, GCNLinkPredictorV2
from utils.data_utils import load_dataset

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

def generate_node2vec_features(data, dataset_name, embedding_dim=128):
    embed_path = f"artifacts/{dataset_name}_node2vec.pt"
    
    if os.path.exists(embed_path):
        print(f"✅ Loading cached Node2Vec embeddings from {embed_path}")
        return torch.load(embed_path)

    print("⚙️ Generating Node2Vec features...")
    edge_index = to_undirected(data.edge_index)
    node2vec = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True
    )

    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    node2vec.train()
    for epoch in range(1, 101):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] Loss: {total_loss:.4f}")

    embeddings = node2vec.embedding.weight.data.clone()

    os.makedirs("artifacts", exist_ok=True)
    torch.save(embeddings, embed_path)
    print(f"💾 Saved Node2Vec embeddings to {embed_path}")
    return embeddings


def fine_tune_robustness(dataset_name, model_variant):
    print(f"\nLoading dataset: {dataset_name}")
    G = load_dataset(dataset_name)
    data = from_networkx(G)

    data.x = generate_node2vec_features(data, dataset_name)

    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    print("Train keys: ", list(train_data.keys()))

    perm_path = f"artifacts/{dataset_name}_perm.pt"
    if os.path.exists(perm_path):
        print(f"✅ Found permutation: {perm_path}")
        perm = torch.load(perm_path)
        for split in [train_data, val_data, test_data]:
            split.x = split.x[perm]
            split.edge_index = perm[split.edge_index]
            split.edge_label_index = perm[split.edge_label_index]
    else:
        print(f"⚠️ Permutation not found at {perm_path}. Proceeding without remapping.")

    in_channels = data.x.size(1)
    hidden_channels = 64
    if model_variant == 'v2':
        model = GCNLinkPredictorV2(in_channels, hidden_channels)
    else:
        model = GCNLinkPredictor(in_channels, hidden_channels)

    model.load_state_dict(torch.load(f"models/{dataset_name}_watermarked_{model_variant}.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    results = {}

    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n[>] Fine-tuning with ratio: {ratio}")
        num_edges = int(train_data.edge_label_index.size(1) * ratio)

        data_subset = train_data.clone()
        data_subset.edge_label_index = train_data.edge_label_index[:, :num_edges]
        data_subset.edge_label = train_data.edge_label[:num_edges]

        train(model, optimizer, data_subset, epochs=50)
        acc = test(model, test_data)
        print(f"[✓] Accuracy after fine-tuning ({ratio*100:.0f}%): {acc:.4f}")
        results[str(ratio)] = acc

    # ✅ Save results for plotting
    os.makedirs("results", exist_ok=True)
    save_path = f"results/fine_tuning_robustness_{dataset_name}_{model_variant}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    fine_tune_robustness(args.dataset, args.model_variant)
