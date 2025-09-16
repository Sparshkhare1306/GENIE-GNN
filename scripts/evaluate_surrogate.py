#!/usr/bin/env python
"""
Robust surrogate evaluation.

- Tries to load the surrogate model and compute logits for edge sets.
- If the checkpoint is a state_dict with convs.* weights but cannot be reconstructed reliably,
  falls back to evaluating watermark detectability using Node2Vec embeddings (logistic regression).
- Saves a JSON results file (surrogate_eval_results.json) to wm_dir if provided, otherwise
  next to the surrogate model.
"""
import argparse
import os
import inspect
import re
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import torch.nn as nn


def try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return None
    except Exception:
        raise


def dot_product_logits(emb, edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    logits = (emb[src] * emb[dst]).sum(dim=1)
    return logits


def compute_logits_safe(model, x, edges, device="cpu"):
    """
    Try several calling conventions to obtain logits for `edges`.
    If model won't produce logits but produces embeddings, compute dot-product logits.
    """
    model.to(device)
    model.eval()

    # prefer explicit decode-like methods
    for name in ("decode", "link_predict", "score_edges", "predict_edges"):
        if hasattr(model, name):
            out = try_call(getattr(model, name), x, edges)
            if out is not None:
                return out

    # inspect signature for edge_label_index / edge_index
    try:
        sig = inspect.signature(model.forward)
        params = list(sig.parameters.keys())[1:]
    except Exception:
        params = []

    if "edge_label_index" in params:
        out = try_call(model.forward, x, edge_label_index=edges)
        if out is not None:
            return out

    if "edge_index" in params:
        out = try_call(model.forward, x, edge_index=edges)
        if out is not None:
            return out

    # positional
    out = try_call(model.forward, x, edges)
    if out is not None:
        return out

    out = try_call(model, x, edges)
    if out is not None:
        return out

    # try to get embeddings then dot-product
    emb = try_call(model.forward, x)
    if emb is None:
        emb = try_call(model, x)
    if emb is None:
        raise TypeError("Unable to obtain logits or embeddings from surrogate model with tried call patterns.")

    logits = dot_product_logits(emb, edges.to(device))
    return logits


def evaluate(model, data, pos_edges, device="cpu"):
    """Compute AUC over a sampled positive set vs negatives using model (which produces logits)."""
    model.eval()
    pos = pos_edges
    neg = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos.size(1),
    )
    edges = torch.cat([pos, neg], dim=1)
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).cpu().numpy()

    with torch.no_grad():
        logits = compute_logits_safe(model, data.x.to(device), edges.to(device), device=device)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    try:
        return roc_auc_score(labels, probs)
    except Exception:
        return float("nan")


def load_dataset(name: str):
    import networkx as nx
    from torch_geometric.utils import from_networkx
    from torch_geometric.data import Data
    from scipy.io import mmread
    import numpy as np
    import torch
    import os

    ddir = f"datasets/{name}"
    txt_path = os.path.join(ddir, f"{name}.txt")
    mtx_path = os.path.join(ddir, f"{name}.mtx")

    if os.path.exists(txt_path):
        print(f"[INFO] Loading edgelist graph from {txt_path}")
        G = nx.read_edgelist(txt_path, nodetype=int)
        G = nx.convert_node_labels_to_integers(G)
        data = from_networkx(G)
        data.num_nodes = G.number_of_nodes()

    elif os.path.exists(mtx_path):
        print(f"[INFO] Loading MatrixMarket graph from {mtx_path}")
        adj = mmread(mtx_path).tocsr()
        edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=adj.shape[0])

    else:
        raise FileNotFoundError(f"No dataset file found for {name} in {ddir}")

    # Assign features if missing
    if not hasattr(data, "x") or data.x is None:
        print(f"[INFO] Assigning random features (num_nodes={data.num_nodes}, dim=64)")
        data.x = torch.randn(data.num_nodes, 64)

    return data


def build_linear_encoder_from_state_dict(state_dict, expected_input_dim=64, device="cpu"):
    """
    Build a linear-stack proxy encoder from convs.*.lin.weight keys in state_dict.
    This is a best-effort heuristic (transpose / reshape) but may not always match real model.
    Returns nn.Module (encoder) or None.
    """
    pattern = re.compile(r"convs\.(\d+)\.lin\.weight")
    conv_keys = {}
    for k, v in state_dict.items():
        m = pattern.match(k)
        if m:
            idx = int(m.group(1))
            conv_keys.setdefault(idx, {})["weight"] = v.clone().cpu()
    if not conv_keys:
        return None

    max_idx = max(conv_keys.keys())
    dims = []
    prev_out = None
    for i in range(max_idx + 1):
        ent = conv_keys.get(i)
        if ent is None or "weight" not in ent:
            raise RuntimeError(f"Missing conv weight info for convs.{i}")
        w = ent["weight"]
        s0, s1 = tuple(w.shape)
        # guess orientation: try to satisfy expected_input_dim for first layer
        if i == 0:
            if s1 == expected_input_dim:
                in_dim, out_dim = s1, s0
                transpose = False
            elif s0 == expected_input_dim:
                in_dim, out_dim = s0, s1
                transpose = True
            else:
                # fallback: prefer in_dim = expected_input_dim
                in_dim = expected_input_dim
                out_dim = max(s0, s1)
                transpose = False
        else:
            # try match prev_out
            if s1 == prev_out:
                in_dim, out_dim = s1, s0
                transpose = False
            elif s0 == prev_out:
                in_dim, out_dim = s0, s1
                transpose = True
            else:
                in_dim = prev_out or min(s0, s1)
                out_dim = max(s0, s1)
                transpose = False
        dims.append((in_dim, out_dim, transpose))
        prev_out = out_dim

    # build linear proxy
    class Encoder(nn.Module):
        def __init__(self, spec):
            super().__init__()
            layers = []
            for (in_d, out_d, _) in spec:
                layers.append(nn.Linear(in_d, out_d))
                layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)

        def forward(self, x, edge_index=None):
            return self.net(x)

    enc = Encoder(dims).to(device)
    enc_state = enc.state_dict()
    mapped = {}
    # map saved weights into enc
    for i in range(len(dims)):
        linear_w_key = f"net.{2*i}.weight"
        linear_b_key = f"net.{2*i}.bias"
        saved_w_keys = [f"convs.{i}.lin.weight", f"module.convs.{i}.lin.weight"]
        saved_b_keys = [f"convs.{i}.bias", f"module.convs.{i}.bias"]
        sv = None
        for k in saved_w_keys:
            if k in state_dict:
                sv = state_dict[k].clone().cpu()
                break
        if sv is None:
            continue
        target_shape = enc_state[linear_w_key].shape  # (out, in)
        if tuple(sv.shape) == target_shape:
            mapped[linear_w_key] = sv
        else:
            # try transpose
            try:
                if tuple(sv.t().shape) == target_shape:
                    mapped[linear_w_key] = sv.t().contiguous()
                elif sv.numel() == enc_state[linear_w_key].numel():
                    mapped[linear_w_key] = sv.reshape(target_shape)
                else:
                    print(f"[WARN] Shape mismatch for saved convs.{i}.lin.weight {tuple(sv.shape)} -> expected {target_shape}; skipping")
            except Exception:
                print(f"[WARN] Failed adapting convs.{i}.lin.weight; skipping")
        # bias
        sb = None
        for k in saved_b_keys:
            if k in state_dict:
                sb = state_dict[k].clone().cpu()
                break
        if sb is not None:
            if tuple(sb.shape) == tuple(enc_state[linear_b_key].shape):
                mapped[linear_b_key] = sb
            else:
                if sb.numel() == enc_state[linear_b_key].numel():
                    mapped[linear_b_key] = sb.reshape(enc_state[linear_b_key].shape)
                else:
                    print(f"[WARN] Shape mismatch for saved convs.{i}.bias {tuple(sb.shape)} -> expected {tuple(enc_state[linear_b_key].shape)}; skipping")

    enc.load_state_dict(mapped, strict=False)
    return enc


def fallback_eval_with_node_embeddings(wm_dir, data, wm_edges, wm_labels):
    """
    Loads node2vec embeddings from wm_dir (node2vec_embeddings.pt) and trains
    a logistic regression on (src_emb || dst_emb) to predict edge labels for watermark edges.
    Splits wm edges into train/test for a holdout AUC.
    """
    cand = os.path.join(wm_dir, "node2vec_embeddings.pt")
    if not os.path.exists(cand):
        raise FileNotFoundError(f"Node2Vec embeddings not found at {cand}. Please run scripts/save_embeddings.py first.")
    emb = torch.load(cand)  # shape: [num_nodes, dim]
    if isinstance(emb, torch.Tensor):
        emb = emb.numpy()
    # prepare X,y from wm_edges (2,E) and wm_labels (E,)
    E = wm_edges.shape[1]
    src = wm_edges[0].numpy().astype(int)
    dst = wm_edges[1].numpy().astype(int)
    X = np.concatenate([emb[src], emb[dst]], axis=1)
    y = wm_labels.astype(int)
    # train/test split
    if len(y) < 10:
        test_frac = 0.5
    else:
        test_frac = 0.3
    n_test = max(1, int(len(y) * test_frac))
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    clf = LogisticRegression(max_iter=2000)
    if len(train_idx) == 0:
        # train on all, evaluate on same (degenerate)
        clf.fit(X, y)
        probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        return auc
    clf.fit(X[train_idx], y[train_idx])
    probs = clf.predict_proba(X[test_idx])[:, 1]
    auc = roc_auc_score(y[test_idx], probs)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["CA-HepTh", "C-ELEGANS"])
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--surrogate_model", required=True, help="Path to surrogate model (.pth)")
    parser.add_argument("--wm_dir", default=None, help="Directory containing wm_edges.pt and wm_labels.pt and node2vec_embeddings.pt")
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(args.dataset)

    # pad baseline to requested embedding_dim
    pad = 0
    if data.x.size(1) < args.embedding_dim:
        pad = args.embedding_dim - data.x.size(1)
        data.x = torch.cat([data.x, torch.zeros(data.num_nodes, pad)], dim=1)
    print(f"[INFO] Padded features from {data.x.size(1) - pad} -> {data.x.size(1)}")

    sur_path = args.surrogate_model
    print(f"[INFO] Loading surrogate model from: {sur_path}")
    loaded = torch.load(sur_path, map_location=device)
    model = None
    used_fallback = False

    # initialize result variables
    test_auc = None
    wm_auc = None

    try:
        if isinstance(loaded, torch.nn.Module):
            model = loaded.to(device)
        elif isinstance(loaded, dict):
            possible_sd = None
            if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                possible_sd = loaded["state_dict"]
            else:
                if all(isinstance(v, torch.Tensor) for v in loaded.values()):
                    possible_sd = loaded
            if possible_sd is not None:
                # attempt to create an encoder
                try:
                    enc = build_linear_encoder_from_state_dict(possible_sd, expected_input_dim=args.embedding_dim, device=device)
                    if enc is None:
                        raise RuntimeError("no convs.* keys found")
                    class Wrapper(nn.Module):
                        def __init__(self, encoder):
                            super().__init__()
                            self.encoder = encoder
                        def forward(self, x, edge_label_index=None, edge_index=None):
                            emb = self.encoder(x)
                            if edge_label_index is None and edge_index is None:
                                return emb
                            eidx = edge_label_index if edge_label_index is not None else edge_index
                            logits = dot_product_logits(emb, eidx.to(emb.device))
                            return logits
                    model = Wrapper(enc).to(device)
                except Exception as e:
                    print(f"[WARN] Rebuilding encoder from state_dict failed: {e}")
                    model = None
            else:
                # try to find an embedded module
                found_module = None
                for k in ("model", "surrogate", "net"):
                    if k in loaded and isinstance(loaded[k], torch.nn.Module):
                        found_module = loaded[k]
                        break
                if found_module is not None:
                    model = found_module.to(device)
                else:
                    raise RuntimeError("Unrecognized surrogate checkpoint format.")
        else:
            raise RuntimeError("Unexpected surrogate file content.")
    except Exception as e:
        print(f"[WARN] Could not build model from checkpoint: {e}")
        model = None

    # If we successfully have a model, try a quick test AUC
    if model is not None:
        model.to(device)
        model.eval()
        try:
            pos_edges = data.edge_index[:, :min(1000, max(1, data.edge_index.size(1) // 10))]
            test_auc = evaluate(model, data, pos_edges, device=device)
            print(f"[RESULT] Test AUC (surrogate) on random sample: {test_auc:.4f}")
        except Exception as e:
            print(f"[WARN] Surrogate model evaluation failed: {e}")
            model = None

    # If watermark evaluation requested
    if args.wm_dir:
        wm_edges_path = os.path.join(args.wm_dir, "wm_edges.pt")
        wm_labels_path = os.path.join(args.wm_dir, "wm_labels.pt")
        if os.path.exists(wm_edges_path) and os.path.exists(wm_labels_path):
            wm_edges = torch.load(wm_edges_path)
            wm_labels = torch.load(wm_labels_path).float().cpu().numpy()
            if model is not None:
                try:
                    with torch.no_grad():
                        wm_logits = compute_logits_safe(model, data.x.to(device), wm_edges.to(device), device=device)
                        wm_probs = torch.sigmoid(wm_logits).cpu().numpy().ravel()
                    wm_auc = roc_auc_score(wm_labels, wm_probs)
                    print(f"[RESULT] Watermark AUC (via surrogate model): {wm_auc:.4f}")
                except Exception as e:
                    print(f"[WARN] Model-based watermark evaluation failed: {e}")
                    used_fallback = True
            else:
                used_fallback = True

            if used_fallback:
                try:
                    wm_auc = fallback_eval_with_node_embeddings(args.wm_dir, data, wm_edges, wm_labels)
                    print(f"[RESULT] Watermark AUC (via Node2Vec embeddings + logistic regression fallback): {wm_auc:.4f}")
                except Exception as e:
                    print(f"[ERROR] Fallback watermark evaluation failed: {e}")
        else:
            print("[WARN] No watermark files found in", args.wm_dir)

    # Save results to JSON
    results = {
        "dataset": args.dataset,
        "surrogate_model": args.surrogate_model,
        "wm_dir": args.wm_dir,
        "test_auc": float(test_auc) if test_auc is not None else None,
        "wm_auc": float(wm_auc) if wm_auc is not None else None,
        "used_fallback": bool(used_fallback),
    }

    if args.wm_dir:
        out_path = os.path.join(args.wm_dir, "surrogate_eval_results.json")
    else:
        out_path = os.path.join(os.path.dirname(args.surrogate_model), "surrogate_eval_results.json")

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Saved results to {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save results JSON: {e}")


if __name__ == "__main__":
    main()
