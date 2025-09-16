#!/usr/bin/env python3
# scripts/save_best_surrogate.py
# Finds the best val_auc entry in each dataset's results/.../model_extraction/metrics.csv
# and copies the current surrogate_model.pth to a timestamped file for safekeeping.

import csv
import os
import shutil
from datetime import datetime

# Datasets to check — edit if you used different dataset names
DATASETS = ["CA-HepTh", "C-ELEGANS"]

def find_best(metrics_path):
    best = None
    best_row = None
    if not os.path.exists(metrics_path):
        return None, None
    with open(metrics_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # try keys 'val_auc' or 'best_val_auc' or fallback
            val_str = row.get("val_auc", row.get("best_val_auc", "nan"))
            try:
                val = float(val_str)
            except:
                val = float("nan")
            # treat NaN as not comparable
            if best is None:
                if val == val:  # not NaN
                    best = val
                    best_row = row
            else:
                if val == val and val > best:
                    best = val
                    best_row = row
    return best, best_row

def main():
    for ds in DATASETS:
        metrics_path = os.path.join("results", ds, "model_extraction", "metrics.csv")
        best, row = find_best(metrics_path)
        if row is None:
            print(f"[SKIP] No valid metrics for {ds} ({metrics_path} missing or no numeric val_auc).")
            continue
        print(f"[INFO] {ds}: best val_auc = {best}, row = {row}")
        out_dir = os.path.join("results", ds, "model_extraction")
        src = os.path.join(out_dir, "surrogate_model.pth")
        if not os.path.exists(src):
            print(f"[WARN] No surrogate model found at {src} — nothing to copy.")
            continue
        qr = row.get("query_ratio", "qr")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_name = f"surrogate_model_best_qr{qr}_{timestamp}.pth"
        dst = os.path.join(out_dir, dst_name)
        try:
            shutil.copy(src, dst)
            print(f"[SAVED] Copied {src} -> {dst}")
        except Exception as e:
            print(f"[ERROR] Failed to copy: {e}")

if __name__ == "__main__":
    main()
