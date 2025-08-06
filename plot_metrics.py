# plot_metrics.py

import json
import argparse
import matplotlib.pyplot as plt
import os

def plot_fine_tuning_robustness(dataset: str, model_variant: str):
    results_path = f"results/fine_tuning_robustness_{dataset}_{model_variant}.json"
    if not os.path.exists(results_path):
        print(f"❌ Result file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    ratios = sorted([float(k) for k in results.keys()])
    accuracies = [results[str(r)] for r in ratios]

    plt.figure(figsize=(8, 5))
    plt.plot(ratios, accuracies, marker='o', linestyle='-', color='mediumblue')
    plt.xlabel("Fine-Tuning Ratio")
    plt.ylabel("Accuracy")
    plt.title(f"Fine-Tuning Robustness: {dataset} (Model Variant {model_variant})")
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"results/fine_tuning_plot_{dataset}_{model_variant}.png"
    plt.savefig(plot_path)
    print(f"✅ Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_variant", type=str, required=True)
    args = parser.parse_args()

    plot_fine_tuning_robustness(args.dataset, args.model_variant)
