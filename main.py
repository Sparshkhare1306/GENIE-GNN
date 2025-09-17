# main.py
import os, argparse, json, random
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit

from datasets.datasets import CAHepTh, CElegans
from models.gcn_link_predictor import GCNLinkPredictorV2
from genie import generate_watermark_edges, add_watermark_to_split, verify_watermark

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds.cpu() == labels.float().cpu()).float().mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["CA-HepTh", "C-ELEGANS"], required=True)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--subset_ratio", type=float, default=0.3)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load dataset
    if args.dataset == "CA-HepTh":
        dataset = CAHepTh(root="./data/Snap")
    elif args.dataset == "C-ELEGANS":
        dataset = CElegans(root="./data/NetworkRepository")
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    data = dataset[0]

    # Dummy features if none exist
    if not hasattr(data, "x") or data.x is None:
        data.x = torch.eye(data.num_nodes)

    # ✅ Train/val/test split
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    # ✅ Inject watermark into training split
    wm_edge_index, wm_edge_label = generate_watermark_edges(data, num_edges=50, seed=42)
    train_data = add_watermark_to_split(train_data, wm_edge_index, wm_edge_label)

    # ✅ Model
    model = GCNLinkPredictorV2(in_channels=data.x.size(-1), hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ✅ Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        logits = model.decode(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label.float())
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(test_data.x, test_data.edge_index)
                logits = model.decode(z, test_data.edge_label_index)
                acc = accuracy_from_logits(logits, test_data.edge_label)
                wm_acc = verify_watermark(model, data, wm_edge_index, wm_edge_label)
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}, WM Acc: {wm_acc:.4f}")

    # ✅ Save
    results_dir = os.path.join("results", args.dataset, f"subset_{args.subset_ratio:.2f}".replace(".", "_"))
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "watermarked_model.pth")
    torch.save({
        "model_state": model.state_dict(),
        "config": {"input_dim": data.x.size(-1), "hidden_dim": args.hidden_channels}
    }, ckpt_path)

    with open(os.path.join(results_dir, "train_summary.json"), "w") as f:
        json.dump({"final_test_acc": float(acc), "final_wm_acc": float(wm_acc)}, f, indent=2)

    print(f"Saved model -> {ckpt_path}")

if __name__ == "__main__":
    main()
