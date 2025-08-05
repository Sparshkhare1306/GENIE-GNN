import os
import argparse
import torch

from attacks.watermark import apply_watermark
from utils.data_utils import load_dataset, generate_node2vec_features
from models.gcn_link_predictor import GCNLinkPredictor, GCNLinkPredictorV2


def main(dataset: str, model_variant: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading graph for dataset: {dataset}")
    graph = load_dataset(dataset)
    print(f"Graph loaded! Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    print("Generating Node2Vec features...")
    x = generate_node2vec_features(graph)
    print("Feature matrix shape:", x.shape)

    # ✅ Confirm model variant selection
    print(f"Using model variant: {model_variant}")
    if model_variant == 'v1':
        model = GCNLinkPredictor(in_channels=x.shape[1], hidden_channels=64).to(device)
    elif model_variant == 'v2':
        model = GCNLinkPredictorV2(in_channels=x.shape[1], hidden_channels=64).to(device)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    print("Applying watermarking...")
    model = apply_watermark(graph, x, model, device)

    save_path = f"models/{dataset}_watermarked_{model_variant}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Watermarked model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["CA-HepTh", "C-ELEGANS"], required=True)
    parser.add_argument("--model_variant", type=str, choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    main(args.dataset, args.model_variant)
