import os
import csv
import matplotlib.pyplot as plt

def load_metrics(file_path):
    ratios, test_aucs, wm_aucs = [], [], []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratios.append(float(row["query_ratio"]))
            test_aucs.append(float(row["test_auc"]))
            wm_auc = row["watermark_auc"]
            wm_aucs.append(float(wm_auc) if wm_auc != "N/A" else None)
    return ratios, test_aucs, wm_aucs

def plot_dataset(dataset):
    csv_path = f"results/{dataset}/model_extraction/metrics.csv"
    if not os.path.exists(csv_path):
        print(f"[!] No results found for {dataset}")
        return
    
    ratios, test_aucs, wm_aucs = load_metrics(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(ratios, test_aucs, marker='o', label="Test AUC")
    if any(wm_aucs):
        plt.plot(ratios, wm_aucs, marker='x', label="Watermark AUC")
    plt.title(f"Surrogate Model Performance on {dataset}")
    plt.xlabel("Query Ratio")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{dataset}/model_extraction/auc_vs_query_ratio.png")
    print(f"[✓] Saved plot for {dataset}")

if __name__ == "__main__":
    for dataset in ["C-ELEGANS", "CA-HepTh"]:
        plot_dataset(dataset)
