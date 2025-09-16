#!/usr/bin/env python3
# attacks/pruning_attack.py
"""
Pruning attack script.

Usage:
  python -m attacks.pruning_attack --dataset CA-HepTh --subset_ratio 0.3 --prune_ratio 0.2 --save_pruned_model
"""

import os
import sys
import argparse
import csv
import torch
import torch.nn.utils.prune as prune
import torch_geometric.nn as pyg_nn
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx, train_test_split_edges

# Ensure local repo packages are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local dataset/model utilities (expected in your repo)
from datasets.embed_hepth import generate_node2vec_features
from datasets.watermark import inject_watermark_features
from datasets.load_hepth import load_hepth
from datasets.load_amazon import load_amazon
from datasets.load_celegans import load_celegans

from models.gcn_link_predictor import GCNLinkPredictor

# -----------------------
# Helpers
# -----------------------
def get_in_channels_from_state_dict(state_dict):
    """
    Try to infer the model's input feature dimension (in_channels)
    by inspecting convolution weight tensors in a saved state_dict.
    """
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and ("conv" in k or "convs" in k) and "weight" in k:
            if v.ndim >= 2:
                return int(v.shape[1])
    return None

def get_hidden_from_state_dict(state_dict):
    """
    Try to infer hidden/out dimension from conv weight shapes (v.shape[0]).
    """
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and ("conv" in k or "convs" in k) and "weight" in k:
            if v.ndim >= 2:
                return int(v.shape[0])
    return None

def unwrap_state_dict(checkpoint):
    """
    Given checkpoint loaded by torch.load, return the inner state_dict and any model_args dict (or None).
    Handles common checkpoint formats: {model_state:..., state_dict:..., or raw state_dict}
    """
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
            return checkpoint["model_state"], checkpoint.get("model_args", None)
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"], checkpoint.get("model_args", None)
        # fallback: search nested dict looking like a state_dict
        for k, v in checkpoint.items():
            if isinstance(v, dict) and any(("convs.0.lin.weight" in kk or "convs.0.weight" in kk) for kk in v.keys()):
                return v, checkpoint.get("model_args", None)
        # treat checkpoint as state_dict itself
        return checkpoint, checkpoint.get("model_args", None)
    else:
        return checkpoint, None

def pad_or_trim_features(x: torch.Tensor, target_dim: int, device: torch.device, desc: str = "features"):
    """
    If x has fewer columns than target_dim, pad with zeros to the right.
    If x has more columns, trim to target_dim.
    Returns new tensor on device.
    """
    orig_dim = x.size(1)
    if orig_dim == target_dim:
        return x.to(device)
    if orig_dim < target_dim:
        pad_size = target_dim - orig_dim
        padded = torch.zeros((x.size(0), target_dim), dtype=x.dtype, device=device)
        padded[:, :orig_dim] = x.to(device)
        print(f"[WARN] Padded {desc} from {orig_dim} -> {target_dim}")
        return padded
    else:
        trimmed = x[:, :target_dim].to(device)
        print(f"[WARN] Trimmed {desc} from {orig_dim} -> {target_dim}")
        return trimmed

# -----------------------
# Main
# -----------------------
def run_pruning(dataset_name, subset_ratio, prune_ratio, save_pruned_model=False, prune_method="global", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset_map = {
        "CA-HepTh": (load_hepth, "data/Snap/ca-HepTh.txt"),
        "AMAZON": (load_amazon, "data/Snap/amazon_co_purchase.txt"),
        "C-ELEGANS": (load_celegans, "data/Snap/c_elegans.mtx"),
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader, path = dataset_map[dataset_name]
    print(f"[INFO] Loading graph for dataset: {dataset_name} from {path} ...")
    graph = loader(path)
    print(f"[INFO] Graph loaded! Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Node2Vec features (may take a bit)
    print("[INFO] Generating Node2Vec features...")
    features = generate_node2vec_features(graph, embedding_dim=64, epochs=50)
    print(f"[INFO] Feature matrix shape: {features.shape}")

    data = from_networkx(graph)
    data.x = torch.tensor(features, dtype=torch.float)
    data = train_test_split_edges(data)

    # Inject watermark modifications (returns wm_graph, wm_edge_index, wm_features, wm_labels)
    wm_graph, wm_edge_index, wm_features, wm_labels = inject_watermark_features(graph, data.x, subset_ratio=subset_ratio)

    # Locate watermarked model produced earlier during training
    subset_folder = f"subset_{subset_ratio:.2f}".replace(".", "_")
    model_path = os.path.join("results", dataset_name, subset_folder, "watermarked_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Watermarked model not found at {model_path}. Please run training first.")

    print(f"[INFO] Loading model checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict, model_args = unwrap_state_dict(checkpoint)

    # Infer in_channels/hidden if possible
    inferred_in = None
    inferred_hidden = None
    if isinstance(model_args, dict):
        inferred_in = model_args.get("in_channels", None)
        inferred_hidden = model_args.get("hidden_channels", None)

    if inferred_in is None:
        inferred_in = get_in_channels_from_state_dict(state_dict)
    if inferred_hidden is None:
        inferred_hidden = get_hidden_from_state_dict(state_dict)

    if inferred_in is None:
        inferred_in = int(data.x.shape[1])
        print(f"[WARN] Could not infer in_channels from checkpoint; falling back to data.x dim = {inferred_in}")
    else:
        print(f"[INFO] Inferred in_channels = {inferred_in}")

    if inferred_hidden is None:
        inferred_hidden = 64
        print(f"[WARN] Could not infer hidden_channels from checkpoint; defaulting to {inferred_hidden}")
    else:
        print(f"[INFO] Inferred hidden_channels = {inferred_hidden}")

    # Build model that matches checkpoint in_channels/hidden
    model = GCNLinkPredictor(in_channels=inferred_in, hidden_channels=inferred_hidden).to(device)

    # Try loading the checkpoint into the model
    try:
        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Loaded checkpoint into model (strict=False).")
    except Exception as exc:
        print(f"[WARN] load_state_dict raised an exception, trying fallback. Exception: {exc}")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("[INFO] Loaded checkpoint into model on second attempt (strict=False).")
        except Exception as exc2:
            raise RuntimeError(f"[ERROR] Failed to load model state_dict. Exception: {exc2}")

    # Move model and features to device
    model.to(device)
    # Pad/trim data.x and wm_features to match inferred_in
    data.x = pad_or_trim_features(data.x, inferred_in, device, desc="data.x")
    wm_features = pad_or_trim_features(wm_features, inferred_in, device, desc="wm_features")
    wm_edge_index = wm_edge_index.to(device)
    wm_labels = wm_labels.to(device)

    # Identify parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, pyg_nn.GCNConv):
            if hasattr(module, "lin") and hasattr(module.lin, "weight"):
                parameters_to_prune.append((module.lin, "weight"))
                print(f"[INFO] Will prune: {name}.lin.weight")
            elif hasattr(module, "weight"):
                parameters_to_prune.append((module, "weight"))
                print(f"[INFO] Will prune: {name}.weight (fallback)")

    if len(parameters_to_prune) == 0:
        raise RuntimeError("[ERROR] No parameters found to prune. Check the model structure.")

    # Apply pruning
    print(f"[INFO] Applying pruning: method={prune_method}, amount={prune_ratio}")
    if prune_method == "global":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_ratio
        )
    else:
        for (m, pname) in parameters_to_prune:
            prune.l1_unstructured(m, name=pname, amount=prune_ratio)

    # Optionally save pruned model (as checkpoint dict)
    results_dir = os.path.join("results", dataset_name, subset_folder)
    os.makedirs(results_dir, exist_ok=True)
    pruned_model_path = os.path.join(results_dir, f"watermarked_model_pruned_{int(prune_ratio*100)}.pth")
    if save_pruned_model:
        save_ckpt = {"model_state": model.state_dict(), "model_args": {"in_channels": inferred_in, "hidden_channels": inferred_hidden}}
        torch.save(save_ckpt, pruned_model_path)
        print(f"[INFO] Saved pruned model to {pruned_model_path}")

    # Evaluation
    @torch.no_grad()
    def evaluate():
        model.eval()
        train_adj = data.train_pos_edge_index.to(device)
        z = model.encode(data.x.to(device), train_adj)

        pos_logits = model.decode(z, data.test_pos_edge_index.to(device)).view(-1)
        neg_logits = model.decode(z, data.test_neg_edge_index.to(device)).view(-1)
        pos_probs = torch.sigmoid(pos_logits).cpu().numpy()
        neg_probs = torch.sigmoid(neg_logits).cpu().numpy()
        y_true = [1] * len(pos_probs) + [0] * len(neg_probs)
        y_scores = list(pos_probs) + list(neg_probs)
        test_auc = float(roc_auc_score(y_true, y_scores))

        # Watermark eval: use wm_features and wm_edge_index/labels
        z_wm = model.encode(wm_features.to(device), train_adj)
        wm_logits = model.decode(z_wm, wm_edge_index).view(-1)
        wm_probs = torch.sigmoid(wm_logits).cpu().numpy()
        wm_auc = float(roc_auc_score(wm_labels.cpu().numpy(), wm_probs))

        return test_auc, wm_auc

    try:
        test_auc, wm_auc = evaluate()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise

    print(f"[RESULT] Test AUC after pruning: {test_auc:.4f}")
    print(f"[RESULT] Watermark AUC after pruning: {wm_auc:.4f}")

    # Save CSV results
    out_csv = os.path.join(results_dir, f"pruning_{int(prune_ratio*100)}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prune_ratio", "test_auc", "watermark_auc", "pruned_model_path"])
        writer.writerow([prune_ratio, test_auc, wm_auc, pruned_model_path if save_pruned_model else "N/A"])
    print(f"[INFO] Pruning results saved to {out_csv}")

    return {
        "dataset": dataset_name,
        "subset": subset_folder,
        "prune_ratio": prune_ratio,
        "test_auc": test_auc,
        "wm_auc": wm_auc,
        "pruned_model_path": pruned_model_path if save_pruned_model else None,
        "csv": out_csv,
    }

# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pruning attack on watermarked GNN models")
    parser.add_argument("--dataset", type=str, required=True, choices=["CA-HepTh", "AMAZON", "C-ELEGANS"])
    parser.add_argument("--subset_ratio", type=float, default=0.3)
    parser.add_argument("--prune_ratio", type=float, default=0.2)
    parser.add_argument("--save_pruned_model", action="store_true", help="Save pruned model to results folder")
    parser.add_argument("--prune_method", type=str, default="global", choices=["global", "per_module"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pruning(
        dataset_name=args.dataset,
        subset_ratio=args.subset_ratio,
        prune_ratio=args.prune_ratio,
        save_pruned_model=args.save_pruned_model,
        prune_method=args.prune_method,
    )
