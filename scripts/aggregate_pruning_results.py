#!/usr/bin/env python3
# scripts/aggregate_pruning_results.py
import os
import csv
import glob
import json

ROOT = "results"

def find_pruning_csvs(root=ROOT):
    paths = glob.glob(os.path.join(root, "*", "subset_*", "pruning_*.csv"))
    return sorted(paths)

def parse_csv(path):
    # returns dict with dataset, subset, prune_ratio, test_auc, watermark_auc
    parts = path.split(os.sep)
    # expected pattern: results/<dataset>/subset_<x>/pruning_<n>.csv
    try:
        dataset = parts[1]
        subset = parts[2]
    except Exception:
        dataset = "unknown"
        subset = "unknown"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        # header, then a single row
        if len(rows) >= 2:
            header = rows[0]
            row = rows[1]
            # canonicalize
            d = dict(zip(header, row))
            return {
                "path": path,
                "dataset": dataset,
                "subset": subset,
                "prune_ratio": float(d.get("prune_ratio", "nan")),
                "test_auc": float(d.get("test_auc", "nan")),
                "watermark_auc": float(d.get("watermark_auc", "nan"))
            }
    return {"path": path, "dataset": dataset, "subset": subset}

def main():
    csvs = find_pruning_csvs()
    out = []
    for p in csvs:
        try:
            out.append(parse_csv(p))
        except Exception as e:
            print("Failed to parse", p, e)
    out_path = os.path.join(ROOT, "pruning_summary.csv")
    # write summary
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset","subset","prune_ratio","test_auc","watermark_auc","path"])
        for r in out:
            writer.writerow([r["dataset"], r["subset"], r["prune_ratio"], r["test_auc"], r["watermark_auc"], r["path"]])
    print("[DONE] Wrote", out_path)

if __name__ == "__main__":
    main()
