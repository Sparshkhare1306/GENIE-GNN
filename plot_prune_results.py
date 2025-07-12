import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

DATASETS = ["CA-HepTh", "C-ELEGANS"]

for dataset in DATASETS:
    dataset_dir = os.path.join("results", dataset)
    csv_path = os.path.join(dataset_dir, "metrics_prune.csv")
    plots_dir = os.path.join(dataset_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"❌ No CSV found for {dataset} at {csv_path}")
        continue

    # Load and sort data
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["subset_ratio", "prune_threshold"])
    
    # Save sorted version
    summary_csv = os.path.join(dataset_dir, "summary_metrics.csv")
    df.to_csv(summary_csv, index=False)
    print(f"📄 Sorted metrics saved at: {summary_csv}")

    # Plot Test AUC
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df, x="prune_threshold", y="test_auc", hue="subset_ratio", marker="o", palette="viridis"
    )
    plt.title(f"{dataset} - Test AUC vs. Prune Threshold")
    plt.xlabel("Prune Threshold")
    plt.ylabel("Test AUC")
    plt.legend(title="Subset Ratio", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    test_plot_path = os.path.join(plots_dir, "test_auc_plot.png")
    plt.savefig(test_plot_path)
    plt.close()
    print(f"✅ Test AUC plot saved to: {test_plot_path}")

    # Plot Watermark AUC
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df, x="prune_threshold", y="watermark_auc", hue="subset_ratio", marker="o", palette="rocket"
    )
    plt.title(f"{dataset} - Watermark AUC vs. Prune Threshold")
    plt.xlabel("Prune Threshold")
    plt.ylabel("Watermark AUC")
    plt.legend(title="Subset Ratio", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    watermark_plot_path = os.path.join(plots_dir, "watermark_auc_plot.png")
    plt.savefig(watermark_plot_path)
    plt.close()
    print(f"✅ Watermark AUC plot saved to: {watermark_plot_path}")
