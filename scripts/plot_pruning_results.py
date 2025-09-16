#!/usr/bin/env python3
# scripts/plot_pruning_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "results"
summary = os.path.join(ROOT, "pruning_summary.csv")
out_dir = os.path.join(ROOT, "plots")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(summary)
for ds in df['dataset'].unique():
    subdf = df[df['dataset'] == ds]
    plt.figure(figsize=(6,4))
    for subset in sorted(subdf['subset'].unique()):
        ss = subdf[subdf['subset'] == subset]
        ss_sorted = ss.sort_values('prune_ratio')
        plt.plot(ss_sorted['prune_ratio'], ss_sorted['test_auc'], marker='o', label=f"{subset} test")
        plt.plot(ss_sorted['prune_ratio'], ss_sorted['watermark_auc'], marker='x', linestyle='--', label=f"{subset} wm")
    plt.xlabel("Prune ratio")
    plt.ylabel("AUC")
    plt.title(f"Pruning results: {ds}")
    plt.legend()
    plt.grid(True)
    outp = os.path.join(out_dir, f"{ds}_pruning_auc.png")
    plt.savefig(outp, dpi=200, bbox_inches='tight')
    print("Saved", outp)
    plt.close()
print("[DONE] Plots saved to", out_dir)
