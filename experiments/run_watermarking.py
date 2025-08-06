import os
import argparse
import torch

from attacks.watermark import apply_watermark
from utils.data_utils import load_dataset, generate_node2vec_features
from models.gcn_link_predictor import GCNLinkPredictor, GCNLinkPredictorV2

# experiments/run_watermarking.py

from models.gcn_link_predictor import GCNLinkPredictor, GCNLinkPredictorV2
from utils.data_utils import load_dataset
from watermarking.generate_watermark import generate_watermark_edges

def run_watermarking(dataset_name, model_variant):
    print(f"Loading graph for dataset: {dataset_name}")
    G = load_dataset(dataset_name)
    print(f"Graph loaded! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("Generating Node2Vec features...")
    watermark_edges = generate_watermark_edges(G, num_edges=50, seed=42)

    # Save watermark edges
    os.makedirs("data/watermark_edges", exist_ok=True)
    watermark_path = f"data/watermark_edges/{dataset_name}_watermark.pt"
    torch.save(watermark_edges, watermark_path)
    print(f"✅ Saved watermark edges to: {watermark_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["CA-HepTh", "C-ELEGANS"])
    parser.add_argument("--model_variant", type=str, default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    run_watermarking(args.dataset, args.model_variant)
