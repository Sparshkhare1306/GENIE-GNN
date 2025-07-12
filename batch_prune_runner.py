import os
import subprocess
import csv
from datetime import datetime

# -------------------- CONFIG --------------------
DATASET = "C-ELEGANS"  # Change this to your dataset name
SUBSET_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
PRUNE_THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2]
RESULTS_DIR = os.path.join("results", DATASET)
CSV_PATH = os.path.join(RESULTS_DIR, "metrics_prune.csv")
PRUNE_SCRIPT = "prune.py"
# ------------------------------------------------

# Create results directory if not exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Write CSV header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "dataset", "subset_ratio", "prune_threshold",
            "test_auc", "watermark_auc"
        ])

# Run pruning for each combination
for subset_ratio in SUBSET_RATIOS:
    for threshold in PRUNE_THRESHOLDS:
        print(f"\n🚀 Running prune.py | Subset: {subset_ratio} | Threshold: {threshold}")
        
        cmd = [
            "python", PRUNE_SCRIPT,
            "--dataset", DATASET,
            "--subset_ratio", str(subset_ratio),
            "--prune_threshold", str(threshold)
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            print(output)

            # Extract AUCs from printed output
            lines = output.strip().split("\n")
            test_auc = None
            wm_auc = None
            for line in lines:
                if "Post-pruning Test AUC" in line:
                    test_auc = float(line.split(":")[-1].strip())
                if "Post-pruning Watermark AUC" in line:
                    wm_auc = float(line.split(":")[-1].strip())

            # Save to CSV
            if test_auc is not None and wm_auc is not None:
                with open(CSV_PATH, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(), DATASET,
                        subset_ratio, threshold, test_auc, wm_auc
                    ])
                print(f"✅ Logged results: Subset={subset_ratio} | Threshold={threshold}")
            else:
                print("⚠️ AUCs not found in output. Skipped logging.")

        except subprocess.CalledProcessError as e:
            print("❌ Error during execution:")
            print(e.output)
