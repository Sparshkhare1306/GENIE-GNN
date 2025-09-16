#!/usr/bin/env python3
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to the summary CSV
csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "surrogate_eval_summary.csv")
csv_path = os.path.abspath(csv_path)

# Load CSV
df = pd.read_csv(csv_path)

# Extract subset ratio from the path string
def parse_subset_ratio(path: str):
    m = re.search(r"subset_0_(\d+)", path)
    return int(m.group(1)) / 100 if m else None

df["subset_ratio"] = df["path"].apply(parse_subset_ratio)

# Group by dataset
datasets = df["dataset"].unique()

outdir = "results/plots"
os.makedirs(outdir, exist_ok=True)

for ds in datasets:
    sub = df[df["dataset"] == ds].sort_values("subset_ratio")

    # --- Line plot: subset ratio vs AUC ---
    plt.figure(figsize=(6,4))
    plt.plot(sub["subset_ratio"], sub["test_auc"], marker="o", label="Test AUC")
    plt.plot(sub["subset_ratio"], sub["wm_auc"], marker="s", label="WM AUC")
    plt.xlabel("Subset ratio used for watermark")
    plt.ylabel("AUC")
    plt.title(f"Surrogate Evaluation on {ds}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{ds}_lineplot.png", dpi=300)
    plt.close()

    # --- Scatter plot: Test AUC vs WM AUC ---
    plt.figure(figsize=(5,5))
    plt.scatter(sub["test_auc"], sub["wm_auc"], c=sub["subset_ratio"], cmap="viridis", s=80)
    for _, row in sub.iterrows():
        plt.text(row["test_auc"]+0.002, row["wm_auc"]+0.002, f"{row['subset_ratio']:.2f}", fontsize=8)
    plt.xlabel("Test AUC")
    plt.ylabel("Watermark AUC")
    plt.title(f"Tradeoff: {ds}")
    plt.colorbar(label="Subset ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{ds}_scatter.png", dpi=300)
    plt.close()

print(f"[DONE] Plots saved in {outdir}")
