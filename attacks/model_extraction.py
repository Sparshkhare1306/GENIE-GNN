#!/usr/bin/env python3
# attacks/model_extraction.py
import argparse
import os
import random
import csv
import sys
import torch
from torch.nn import Linear
from torch_geometric.utils import from_networkx, negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ensure repo root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_link_predictor import GCNLinkPredictor

# Helper: try to import old variant if present (used to load checkpoints with older key naming)
def try_import_old_v2():
    try:
        from models.old_gcn_link_predictor_v2 import OldGCNLinkPredictorV2
        return OldGCNLinkPredictorV2
    except Exception:
        try:
            from models.gcn_link_predictor_v2 import GCNLinkPredictorV2 as OldGCNLinkPredictorV2
            return OldGCNLinkPredictorV2
        except Exception:
            return None

# -----------------------
# Utilities
# -----------------------
def get_in_channels_from_state_dict(state_dict):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and ("conv" in k or "convs" in k) and "weight" in k:
            if v.ndim >= 2:
                return v.shape[1]
    return None

def sample_edges(edge_index, ratio=0.2):
    """
    Randomly sample a subset of edges from edge_index according to ratio.
    Returns an edge_index (2, E_sampled) with selected columns.
    """
    num_edges = int(edge_index.size(1))
    num_samples = max(1, int(num_edges * ratio))
    selected = random.sample(range(num_edges), num_samples)
    return edge_index[:, selected]

@torch.no_grad()
def get_teacher_logits(teacher_model, edge_label_index, features, full_edge_index, device="cpu"):
    """
    Query teacher to get logits for given edges.
    """
    teacher_model.eval()
    # teacher expected to provide encode/decode API
    z = teacher_model.encode(features, full_edge_index)
    logits = teacher_model.decode(z, edge_label_index)
    return logits.detach()

# -----------------------
# Safe ROC AUC wrapper
# -----------------------
def safe_roc_auc_score(y_true, y_score):
    """
    Compute ROC AUC, returning NaN if y_true has only one class or if computation fails.
    """
    try:
        import numpy as _np
        y_true_arr = _np.asarray(y_true)
        if len(_np.unique(y_true_arr)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true_arr, _np.asarray(y_score)))
    except Exception:
        return float("nan")

# -----------------------
# Surrogate training (fixed soft-label handling, safe AUC)
# -----------------------
def train_surrogate(x, full_edge_index, train_edge_index, train_targets, val_edge_index, val_targets,
                    input_dim=64, hidden_dim=64, epochs=50, lr=0.01, device="cpu", soft_labels=False):
    """
    Train a surrogate link predictor. Supports either hard labels (0/1) or soft teacher logits.
    Returns (trained_model, best_val_auc).
    """
    model = GCNLinkPredictor(in_channels=input_dim, hidden_channels=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # encode with full adjacency (match teacher)
        z = model.encode(x, full_edge_index)
        logits = model.decode(z, train_edge_index).view(-1)

        if soft_labels:
            # train_targets are teacher logits -> convert to probabilities for BCEWithLogitsLoss
            train_probs = torch.sigmoid(train_targets).to(device)
            loss = criterion(logits, train_probs)
        else:
            loss = criterion(logits, train_targets.float().to(device))

        loss.backward()
        optimizer.step()

        # validation AUC
        model.eval()
        with torch.no_grad():
            z_val = model.encode(x, full_edge_index)
            val_logits = model.decode(z_val, val_edge_index).view(-1)
            if soft_labels:
                val_probs_target = torch.sigmoid(val_targets).to(device)
                y_true_bin = (val_probs_target > 0.5).cpu().numpy()
            else:
                y_true_bin = val_targets.cpu().numpy()

            y_pred_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = safe_roc_auc_score(y_true_bin, y_pred_prob)

        if val_auc == val_auc and val_auc > best_auc:  # val_auc == val_auc guards NaN
            best_auc = val_auc
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == epochs:
            if val_auc != val_auc:
                print(f"[SURR] Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val AUC: NaN (single-class val set)")
            else:
                print(f"[SURR] Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[WARN] No best state selected (validation AUC never improved or invalid). Using last epoch weights.")

    return model, (best_auc if best_auc >= 0 else float("nan"))

@torch.no_grad()
def evaluate_surrogate(model, full_edge_index, pos_edge_index, neg_edge_index, features, device="cpu"):
    """
    Evaluate surrogate on provided positive and negative edges, returning ROC AUC.
    Uses safe_roc_auc_score to avoid exceptions when labels are degenerate.
    """
    model.eval()
    z = model.encode(features, full_edge_index)
    pos_score = torch.sigmoid(model.decode(z, pos_edge_index)).view(-1)
    neg_score = torch.sigmoid(model.decode(z, neg_edge_index)).view(-1)
    y_true = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).cpu().numpy()
    y_pred = torch.cat([pos_score, neg_score]).cpu().numpy()
    return safe_roc_auc_score(y_true, y_pred)

# -----------------------
# Dataset + node2vec embedding loader
# -----------------------
def load_dataset(dataset_name, embedding_dim=64):
    """
    Loads dataset graph (NetworkX) and node2vec features using provided dataset-specific loaders.
    Returns (graph_nx, feats_tensor).
    """
    if dataset_name == "C-ELEGANS":
        import datasets.load_celegans as celegans
        graph_nx = celegans.load_celegans("data/C-elegans/celegansneural.mtx")
        from datasets.embed_celegans import generate_node2vec_features
        feats = generate_node2vec_features(graph_nx, embedding_dim=embedding_dim)
    elif dataset_name == "CA-HepTh":
        import datasets.load_hepth as hepth
        graph_nx = hepth.load_hepth("data/Snap/ca-HepTh.txt")
        from datasets.embed_hepth import generate_node2vec_features
        feats = generate_node2vec_features(graph_nx, embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats, dtype=torch.float)
    return graph_nx, feats

# -----------------------
# Main
# -----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    graph_nx, features = load_dataset(args.dataset, embedding_dim=args.embedding_dim)
    data = from_networkx(graph_nx)
    full_edge_index = data.edge_index.to(device)
    x = features.to(device)

    print(f"[INFO] loaded graph: nodes={data.num_nodes}, edges={full_edge_index.size(1)}")
    print(f"[INFO] initial feature dim = {x.size(1)}")

    # subset dir layout (copied from your original script)
    subset_str = str(args.subset_ratio).replace(".", "_")
    if subset_str.endswith("_3"):
        subset_str = subset_str + "0"
    wm_dir = os.path.join("results", args.dataset, f"subset_{subset_str}")
    wm_model_path = os.path.join(wm_dir, "watermarked_model.pth")
    wm_edge_path = os.path.join(wm_dir, "wm_edges.pt")
    wm_label_path = os.path.join(wm_dir, "wm_labels.pt")

    if not os.path.exists(wm_model_path):
        raise FileNotFoundError(f"Watermarked model not found at {wm_model_path}")

    print(f"[INFO] Loading watermarked model from: {wm_model_path}")
    state = torch.load(wm_model_path, map_location=device)

    # support two checkpoint formats: full dict with model_state/model_args, or raw state_dict
    state_dict = state.get("model_state", state) if isinstance(state, dict) else state
    model_args = state.get("model_args", None) if isinstance(state, dict) else None

    teacher_in = None
    if model_args and "in_channels" in model_args:
        teacher_in = model_args["in_channels"]
    else:
        teacher_in = get_in_channels_from_state_dict(state_dict)
    if teacher_in is None:
        teacher_in = args.embedding_dim
        print(f"[WARN] Could not infer teacher in_channels; falling back to embedding_dim={teacher_in}")

    print(f"[INFO] Teacher expected in_channels = {teacher_in}")

    # auto pad/trim if requested
    if x.size(1) != teacher_in:
        if args.auto_pad_features:
            if x.size(1) < teacher_in:
                pad_size = teacher_in - x.size(1)
                padded = torch.zeros((x.size(0), teacher_in), dtype=x.dtype, device=x.device)
                padded[:, : x.size(1)] = x
                x = padded
                print(f"[WARN] Padded features from {features.size(1)} -> {teacher_in}")
            else:
                x = x[:, :teacher_in]
                print(f"[WARN] Trimmed features from {features.size(1)} -> {teacher_in}")
        else:
            raise RuntimeError(
                f"[ERROR] Feature dim mismatch: features={x.size(1)} teacher_in={teacher_in}. "
                "Re-run with --auto_pad_features to allow automatic padding/trimming."
            )

    # choose teacher architecture
    variant = args.model_variant
    if model_args and "variant" in model_args:
        variant = model_args["variant"]

    print(f"[INFO] Teacher variant={variant}, inferred in_channels={teacher_in}")

    teacher_model = None
    if variant == "v2":
        Old = try_import_old_v2()
        if Old is None:
            raise RuntimeError("Old teacher variant v2 cannot be imported. Place compatible model class in models/")
        teacher_model = Old(in_channels=teacher_in).to(device)
    else:
        teacher_model = GCNLinkPredictor(in_channels=teacher_in, hidden_channels=args.hidden_dim).to(device)

    # load weights (try relaxed)
    try:
        teacher_model.load_state_dict(state_dict, strict=False)
        print("[INFO] Loaded teacher checkpoint (strict=False).")
    except Exception:
        teacher_model.load_state_dict(state_dict, strict=False)
        print("[WARN] Loaded teacher checkpoint with relaxed loading.")

    # -------------------------
    # Sample positive and negative edges, build train/val splits (balanced)
    # -------------------------
    pos_edges = sample_edges(full_edge_index, ratio=args.query_ratio)  # positive pool
    num_pos = pos_edges.size(1)
    # sample matching number of negative edges
    neg_edges_sampled = negative_sampling(edge_index=full_edge_index, num_nodes=x.size(0), num_neg_samples=num_pos).to(device)

    # convert to python tuples for splitting convenience
    pos_list = [(int(u), int(v)) for u, v in zip(pos_edges[0].cpu(), pos_edges[1].cpu())]
    neg_list = [(int(u), int(v)) for u, v in zip(neg_edges_sampled[0].cpu(), neg_edges_sampled[1].cpu())]

    # combined edges and labels (balanced)
    combined_edges = pos_list + neg_list
    combined_labels = [1] * len(pos_list) + [0] * len(neg_list)

    # If combined is too small or degenerate, handle fallback
    if len(combined_edges) < 2 or len(set(combined_labels)) < 2:
        # degenerate fallback: put everything in both train and val (not ideal, but avoids crash)
        train_edges_list = combined_edges
        val_edges_list = combined_edges
        train_labels_list = combined_labels
        val_labels_list = combined_labels
    else:
        train_edges_list, val_edges_list, train_labels_list, val_labels_list = train_test_split(
            combined_edges, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
        )
        # ensure non-empty splits
        if len(train_edges_list) == 0:
            train_edges_list = val_edges_list
            train_labels_list = val_labels_list
        if len(val_edges_list) == 0:
            val_edges_list = train_edges_list
            val_labels_list = train_labels_list

    # helper: convert list of (u,v) to edge_index tensor
    def list_to_edge_index(edge_list):
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
        return torch.stack([src, dst], dim=0).contiguous()

    train_edge_index = list_to_edge_index(train_edges_list).to(device)
    val_edge_index = list_to_edge_index(val_edges_list).to(device)

    print(f"[INFO] Querying teacher for labels on sampled edges (train/val sizes = {train_edge_index.size(1)}/{val_edge_index.size(1)})...")
    teacher_train_logits = get_teacher_logits(teacher_model, train_edge_index, x, full_edge_index, device=device)
    teacher_val_logits = get_teacher_logits(teacher_model, val_edge_index, x, full_edge_index, device=device)

    if args.soft_labels:
        train_targets = teacher_train_logits.detach().to(device)
        val_targets = teacher_val_logits.detach().to(device)
    else:
        train_targets = (torch.sigmoid(teacher_train_logits) > 0.5).float().to(device)
        val_targets = (torch.sigmoid(teacher_val_logits) > 0.5).float().to(device)

    # train surrogate
    print("[INFO] Training surrogate model...")
    surrogate, best_val_auc = train_surrogate(
        x, full_edge_index,
        train_edge_index, train_targets,
        val_edge_index, val_targets,
        input_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        epochs=args.surrogate_epochs,
        lr=args.surrogate_lr,
        device=device,
        soft_labels=args.soft_labels
    )

    # evaluate surrogate (use original positive samples pos_edges and new neg sampling)
    neg_edges_for_test = negative_sampling(edge_index=full_edge_index, num_nodes=x.size(0), num_neg_samples=pos_edges.size(1)).to(device)
    test_auc = evaluate_surrogate(surrogate, full_edge_index, pos_edges.to(device), neg_edges_for_test, x, device=device)
    if test_auc != test_auc:
        print(f"[RESULT] Surrogate Test AUC: NaN (degenerate labels or computation issue), best_val_auc: {best_val_auc}")
    else:
        if isinstance(best_val_auc, float) and best_val_auc == best_val_auc:
            print(f"[RESULT] Surrogate Test AUC: {test_auc:.4f}, best_val_auc: {best_val_auc:.4f}")
        else:
            print(f"[RESULT] Surrogate Test AUC: {test_auc:.4f}, best_val_auc: NaN")

    # watermark evaluation if exists
    wm_auc = None
    if os.path.exists(wm_edge_path) and os.path.exists(wm_label_path):
        wm_edge_index = torch.load(wm_edge_path).to(device)
        wm_labels = torch.load(wm_label_path).float().to(device)
        with torch.no_grad():
            surrogate.eval()
            z = surrogate.encode(x, full_edge_index)
            wm_logits = surrogate.decode(z, wm_edge_index)
            wm_probs = torch.sigmoid(wm_logits).cpu().numpy().ravel()
            wm_auc_val = safe_roc_auc_score(wm_labels.cpu().numpy(), wm_probs)
            if wm_auc_val != wm_auc_val:
                print(f"[RESULT] Surrogate Watermark AUC: NaN (degenerate watermark labels)")
                wm_auc = float("nan")
            else:
                print(f"[RESULT] Surrogate Watermark AUC: {wm_auc_val:.4f}")
                wm_auc = wm_auc_val
    else:
        print("[WARN] No watermark edges/labels found at expected paths. Skipping watermark eval.")

    # save outputs
    out_dir = os.path.join("results", args.dataset, "model_extraction")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(surrogate.state_dict(), os.path.join(out_dir, "surrogate_model.pth"))
    metrics_path = os.path.join(out_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "subset_ratio", "query_ratio", "soft_labels", "val_auc", "test_auc", "watermark_auc"])
        wm_val_to_write = ("N/A" if wm_auc is None or (isinstance(wm_auc, float) and wm_auc != wm_auc) else round(wm_auc, 4))
        val_auc_out = ("N/A" if best_val_auc is None or (isinstance(best_val_auc, float) and best_val_auc != best_val_auc) else round(best_val_auc, 4))
        test_auc_out = ("N/A" if test_auc is None or (isinstance(test_auc, float) and test_auc != test_auc) else round(test_auc, 4))
        writer.writerow([args.dataset, args.subset_ratio, args.query_ratio, args.soft_labels, val_auc_out, test_auc_out, wm_val_to_write])
    print(f"[DONE] Saved surrogate and metrics to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["CA-HepTh", "C-ELEGANS"], required=True)
    parser.add_argument("--subset_ratio", type=float, required=True)
    parser.add_argument("--query_ratio", type=float, default=1.0)
    parser.add_argument("--model_variant", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--surrogate_epochs", type=int, default=50)
    parser.add_argument("--surrogate_lr", type=float, default=0.01)
    parser.add_argument("--auto_pad_features", action="store_true")
    parser.add_argument("--soft_labels", action="store_true")
    args = parser.parse_args()
    main(args)
