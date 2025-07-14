import argparse
import os
import random
import csv
import torch
from torch_geometric.utils import from_networkx, negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from models.gcn_link_predictor import GCNLinkPredictor

def load_dataset(dataset_name, embedding_dim=64):
    if dataset_name == "C-ELEGANS":
        import datasets.load_celegans as celegans
        graph_nx = celegans.load_celegans("data/C-elegans/celegansneural.mtx")
        from datasets.embed_celegans import generate_node2vec_features
        features = generate_node2vec_features(graph_nx, embedding_dim=embedding_dim)
    elif dataset_name == "CA-HepTh":
        import datasets.load_hepth as hepth
        graph_nx = hepth.load_hepth("data/Snap/ca-HepTh.txt")
        from datasets.embed_hepth import generate_node2vec_features
        features = generate_node2vec_features(graph_nx, embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return graph_nx, features

def sample_edges(edge_index, ratio=0.2):
    num_edges = edge_index.size(1)
    num_samples = max(1, int(num_edges * ratio))
    selected = random.sample(range(num_edges), num_samples)
    return edge_index[:, selected]

@torch.no_grad()
def get_labels(model, edge_index, features):
    logits = model(features, edge_index)
    probs = torch.sigmoid(logits)
    return (probs > 0.5).float()

def train_surrogate(x, train_edge_index, train_labels, val_edge_index, val_labels,
                    input_dim=64, hidden_dim=64, epochs=50, lr=0.01, device="cpu"):
    model = GCNLinkPredictor(in_channels=input_dim, hidden_channels=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, train_edge_index).squeeze()
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x, val_edge_index).squeeze()
            val_probs = torch.sigmoid(val_out).detach().cpu().numpy()
            y_true = val_labels.cpu().numpy()
            if len(set(y_true)) > 1:
                val_auc = roc_auc_score(y_true, val_probs)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = model.state_dict()
            else:
                val_auc = float('nan')

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    else:
        print("⚠️ Warning: No valid model selected based on Val AUC. Using last epoch model.")

    return model, best_auc

@torch.no_grad()
def evaluate(model, edge_index_pos, edge_index_neg, features):
    model.eval()
    z = model.encode(features, edge_index_pos)
    pos_score = model.decode(z, edge_index_pos)
    neg_score = model.decode(z, edge_index_neg)
    y_true = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])
    y_pred = torch.cat([pos_score, neg_score])
    return roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 64
    hidden_dim = 64

    graph_nx, features = load_dataset(args.dataset, embedding_dim)
    data = from_networkx(graph_nx)
    data.x = torch.tensor(features, dtype=torch.float).to(device)
    data.edge_index = data.edge_index.to(device)

    print("[DEBUG] edge_index shape:", data.edge_index.shape)

    # Load watermarked model
    wm_model_path = f"results/{args.dataset}/subset_0_20/watermarked_model.pth"
    print(f"[DEBUG] Loading watermarked model from: {wm_model_path}")
    wm = GCNLinkPredictor(in_channels=embedding_dim, hidden_channels=hidden_dim).to(device)
    state = torch.load(wm_model_path, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    print(wm)
    print("[DEBUG] State dict keys:", list(state.keys()))
    wm.load_state_dict(state, strict=False)
    wm.eval()

    # Sample and split positive edges
    pos_edges = sample_edges(data.edge_index, ratio=args.query_ratio)
    pos_list = [(u.item(), v.item()) for u, v in zip(pos_edges[0], pos_edges[1])]
    train_pos, val_pos = train_test_split(pos_list, test_size=0.2, random_state=42)
    train_pos_index = torch.tensor(train_pos, dtype=torch.long).t().contiguous().to(device)
    val_pos_index = torch.tensor(val_pos, dtype=torch.long).t().contiguous().to(device)

    # Labels from original model
    train_labels = get_labels(wm, train_pos_index, data.x)
    val_labels = get_labels(wm, val_pos_index, data.x)

    # Train surrogate model
    surrogate, best_val_auc = train_surrogate(
        data.x, train_pos_index, train_labels,
        val_pos_index, val_labels,
        input_dim=embedding_dim, hidden_dim=hidden_dim, epochs=50, lr=0.01, device=device
    )

    # Generate test edges
    edge_index_neg = negative_sampling(edge_index=data.edge_index, num_nodes=data.x.size(0), num_neg_samples=pos_edges.size(1)).to(device)
    test_auc = evaluate(surrogate, pos_edges, edge_index_neg, data.x)
    print(f"🧪 Surrogate model Test AUC: {test_auc:.4f}")

    # Watermark evaluation
    wm_auc = None
    wm_edge_path = f"results/{args.dataset}/subset_0_20/wm_edges.pt"
    wm_label_path = f"results/{args.dataset}/subset_0_20/wm_labels.pt"
    if os.path.exists(wm_edge_path) and os.path.exists(wm_label_path):
        wm_edge_index = torch.load(wm_edge_path).to(device)
        wm_labels = torch.load(wm_label_path).float().to(device)
        with torch.no_grad():
            wm_logits = surrogate(data.x, wm_edge_index)
            wm_probs = torch.sigmoid(wm_logits)
            wm_auc = roc_auc_score(wm_labels.cpu().numpy(), wm_probs.cpu().numpy())
        print(f"🔏 Surrogate model Watermark AUC: {wm_auc:.4f}")
    else:
        print("⚠️ No watermark edges found for evaluation.")

    # Save results
    out_dir = f"results/{args.dataset}/model_extraction"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(surrogate.state_dict(), os.path.join(out_dir, "surrogate_model.pth"))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "query_ratio", "val_auc", "test_auc", "watermark_auc"])
        writer.writerow([
            args.dataset,
            args.query_ratio,
            round(best_val_auc, 4),
            round(test_auc, 4),
            round(wm_auc, 4) if wm_auc else "N/A"
        ])
    print(f"✅ Saved surrogate model and metrics to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["C-ELEGANS", "CA-HepTh"], required=True)
    parser.add_argument("--query_ratio", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
