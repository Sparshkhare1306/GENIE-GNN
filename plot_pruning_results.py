import os
import csv
import matplotlib.pyplot as plt

# Configuration
datasets = ["CA-HepTh", "C-ELEGANS"]
subset_ratios = [0.1, 0.2]
prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Data container
results = {}

# Load results
for dataset in datasets:
    results[dataset] = {}
    for subset in subset_ratios:
        subset_str = f"subset_{subset:.2f}".replace(".", "_")
        data = []
        for prune in prune_ratios:
            path = f"results/{dataset}/{subset_str}/pruning_{int(prune * 100)}.csv"
            if os.path.exists(path):
                with open(path, "r") as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    data.append((
                        float(row["prune_ratio"]),
                        float(row["test_auc"]),
                        float(row["watermark_auc"])
                    ))
        results[dataset][subset] = sorted(data)

# Plotting function
def plot_results(title, data_by_subset, save_path):
    plt.figure(figsize=(10, 6))
    for subset_key, values in data_by_subset.items():
        x = [v[0] for v in values]
        test_auc = [v[1] for v in values]
        wm_auc = [v[2] for v in values]
        plt.plot(x, test_auc, label=f"{subset_key} - Test AUC", marker='o')
        plt.plot(x, wm_auc, label=f"{subset_key} - WM AUC", marker='x')
    plt.xlabel("Prune Ratio")
    plt.ylabel("AUC Score")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Create output directory
os.makedirs("plots", exist_ok=True)

# Individual plots
for dataset in datasets:
    plot_results(
        f"Pruning Effects on {dataset}",
        {f"{dataset} {subset:.1f}": results[dataset][subset] for subset in subset_ratios},
        f"plots/pruning_plot_{dataset}.png"
    )

# Combined plot
combined_data = {}
for dataset in datasets:
    for subset in subset_ratios:
        combined_data[f"{dataset} {subset:.1f}"] = results[dataset][subset]

plot_results(
    "Combined Pruning Effects Across Datasets",
    combined_data,
    "plots/pruning_plot_combined.png"
)
