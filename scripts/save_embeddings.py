#!/usr/bin/env python
import argparse
import os
import sys
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data

# Try importing mmread only when needed; give helpful error if missing.
def try_import_mmread():
    try:
        from scipy.io import mmread
        return mmread
    except Exception as e:
        raise ImportError(
            "scipy is required to read Matrix Market (.mtx) files. "
            "Install it with `pip install scipy` or `pip install -r requirements.txt`."
        ) from e

def load_dataset(name: str) -> Data:
    """Load raw graph dataset into PyG Data. Handles plain edgelist or Matrix Market (.mtx)."""
    if name == "CA-HepTh":
        path = "datasets/CA-HepTh/CA-HepTh.txt"
    elif name == "C-ELEGANS":
        path = "datasets/C-ELEGANS/C-ELEGANS.txt"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Detect MatrixMarket file by header or extension
    # Some files may be named .txt but contain MatrixMarket content (%%MatrixMarket).
    first_line = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # read first non-empty non-comment line to detect header
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped != "":
                first_line = stripped
                break

    is_mtx = False
    if first_line.startswith("%%MatrixMarket"):
        is_mtx = True
    elif path.lower().endswith(".mtx"):
        is_mtx = True

    if is_mtx:
        mmread = try_import_mmread()
        # mmread will return sparse matrix (COO or CSR). We convert to coo and extract edges.
        M = mmread(path)
        # convert to COOrdinate format for row/col
        try:
            M = M.tocoo()
        except Exception:
            # mmread might already be coo, but ensure attributes exist
            pass
        rows = M.row.astype(int)
        cols = M.col.astype(int)
        # nodes count
        num_nodes = max(rows.max() if rows.size else 0, cols.max() if cols.size else 0) + 1
        # create edge_index with both directions (undirected)
        if rows.size == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            # add reverse edges to make undirected if not already present
            edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        data = Data(edge_index=edge_index, num_nodes=int(num_nodes))
        return data
    else:
        # plain edge list
        G = nx.read_edgelist(path, nodetype=int)
        # convert node labels to contiguous integers in case they are not 0..N-1
        G = nx.convert_node_labels_to_integers(G)
        data = from_networkx(G)
        data.num_nodes = G.number_of_nodes()
        return data

def generate_node2vec_embeddings(data: Data, embedding_dim=64, walk_length=20,
                                 context_size=10, walks_per_node=10, epochs=5, device="cpu"):
    """Run Node2Vec to get node embeddings."""
    model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        sparse=True
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Node2Vec] Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model().detach().cpu()

def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading dataset {args.dataset}...")
    data = load_dataset(args.dataset)

    print("[INFO] Generating Node2Vec embeddings...")
    embeddings = generate_node2vec_embeddings(
        data,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.save(embeddings, os.path.join(outdir, "node2vec_embeddings.pt"))
    print(f"[SAVED] Node2Vec embeddings -> {outdir}/node2vec_embeddings.pt")

    # pick watermark edges (random positives + negatives)
    num_pos = min(data.edge_index.size(1) // 20 if data.edge_index.size(1) > 0 else 0, 1000)  # sample up to 5% or 1000 edges
    if num_pos <= 0:
        print(f"[WARN] No edges found in dataset {args.dataset}; skipping watermark creation.")
        return

    perm = torch.randperm(data.edge_index.size(1))[:num_pos]
    wm_pos_edges = data.edge_index[:, perm]

    wm_neg_edges = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_pos
    )

    wm_edges = torch.cat([wm_pos_edges, wm_neg_edges], dim=1)
    wm_labels = torch.cat([
        torch.ones(wm_pos_edges.size(1)),
        torch.zeros(wm_neg_edges.size(1))
    ])

    torch.save(wm_edges, os.path.join(outdir, "wm_edges.pt"))
    torch.save(wm_labels, os.path.join(outdir, "wm_labels.pt"))
    print(f"[SAVED] wm_edges.pt and wm_labels.pt -> {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["CA-HepTh", "C-ELEGANS"])
    parser.add_argument("--subset_ratio", type=float, default=0.3,
                        help="Subset ratio (used for directory consistency only)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
