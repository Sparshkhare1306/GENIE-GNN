import os
import csv
import matplotlib.pyplot as plt

datasets = ["CA-HepTh", "C-ELEGANS"]
subset_ratios = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

for dataset_name in datasets:
    valid_ratios = []
    test_aucs = []
    wm_aucs = []

    base_dir = os.path.join("results", dataset_name)

    for ratio in subset_ratios:
        folder_name = f"subset_{ratio:.2f}".replace('.', '_')
        csv_path = os.path.join(base_dir, folder_name, "metrics.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping {dataset_name} {ratio}, file not found: {csv_path}")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                print(f"Skipping {dataset_name} {ratio}, no data in: {csv_path}")
                continue
            last_row = rows[-1]  # last epoch metrics

        valid_ratios.append(ratio)
        test_aucs.append(float(last_row["test_auc"]))
        wm_aucs.append(float(last_row["watermark_auc"]))

    if valid_ratios:
        plt.plot(valid_ratios, test_aucs, marker='o', label=f'{dataset_name} Test AUC')
        plt.plot(valid_ratios, wm_aucs, marker='s', label=f'{dataset_name} Watermark AUC')

plt.xlabel('Watermark Subset Ratio')
plt.ylabel('AUC Score')
plt.title('AUC vs Subset Ratio for Multiple Datasets')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/combined_auc_vs_ratio.png")
plt.show()
