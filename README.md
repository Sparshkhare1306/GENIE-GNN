# GENIE-GNN: Watermarking Graph Neural Networks for Link Prediction


## 📘 GENIE Reproduction – Watermark Robustness Experiments
## 🔎 Overview of This Reproduction

This repository reproduces watermark robustness experiments from the GENIE paper, focusing on two attacks:

Model Extraction – training surrogate GCN models by querying a watermarked model.

Pruning – removing a fraction of weights from the GCN and evaluating watermark survivability.

## ✅ What has been done:

# Implemented a watermarked GCN link predictor with Node2Vec features.

# Built the model extraction pipeline (CA-HepTh, C-ELEGANS) with variable subset ratios.

# Implemented the pruning attack pipeline with configurable pruning ratios.

# Automated experiment orchestration in test.py.

# Added result saving (CSV + logs) and plotting scripts for easy visualization.

# Packaged everything in a reproducible format for sharing.

## 🎯 What you can do:

# Reproduce all experiments with a single command:

`python test.py --step all`


Inspect intermediate results (results/) and generated plots (results/plots/).

Extend to other robustness experiments (e.g., fine-tuning, watermark overwriting) using the same pipeline.

**📂 Project Structure**

genie_gnn/
│
├── data/                       # Datasets (SNAP CA-HepTh, C-ELEGANS, etc.)
│   └── Snap/
│       ├── ca-HepTh.txt
│       └── C-elegans.txt
│
├── models/                     # Model definitions
│   └── gcn_link_predictor.py
│
├── attacks/                    # Attack scripts
│   ├── model_extraction.py     # Model extraction pipeline
│   └── pruning_attack.py       # Pruning attack pipeline
│
├── utils/                      # Helper functions
│
├── results/                    # Logs, checkpoints, CSVs, and plots
│   ├── CA-HepTh/
│   │   ├── subset_0_30/
│   │   │   ├── watermarked_model.pth
│   │   │   ├── pruning_20.csv
│   │   │   └── plots/
│   └── C-ELEGANS/
│
├── test.py                     # Master script to reproduce experiments
├── plot_metrics.py              # Visualization of metrics
├── requirements.txt             # Dependencies
└── README.md                    # This file

## ⚙️ Setup
** 1. Create environment **
` conda create -n genie_gnn python=3.10 -y `
`conda activate genie_gnn`

** 2. Install dependencies **
` pip install -r requirements.txt `

** 3. Install PyTorch Geometric (if not already installed) **

Follow the instructions at: PyTorch Geometric Installation

## Example (for CPU):

` pip install torch-scatter torch-sparse torch-geometric `

🚀 Usage
1. Train a watermarked model
` python main.py --dataset CA-HepTh --subset_ratio 0.3 --save_model`

2. Run pruning attack
` python -m attacks.pruning_attack --dataset CA-HepTh --subset_ratio 0.3 --prune_ratio 0.2 --save_pruned_model` 

3. Run model extraction attack
` python -m attacks.model_extraction --dataset CA-HepTh --subset_ratio 0.3 --query_ratio 0.5`

4. Run everything end-to-end
` python test.py --step all `

## 📊 Results

Results are saved under results/{DATASET}/subset_{RATIO}/.

For each run you will find:

watermarked_model.pth → Original watermarked model checkpoint

watermarked_model_pruned_XX.pth → Pruned model checkpoint

pruning_XX.csv → Pruning results (Test AUC, WM AUC)

extraction_results.csv → Model extraction results

plots/ → Visualizations of accuracy, watermark robustness, etc.

Example output:

results/CA-HepTh/subset_0_30/
│
├── watermarked_model.pth
├── watermarked_model_pruned_20.pth
├── pruning_20.csv
├── extraction_results.csv
└── plots/
    ├── pruning_curve.png
    └── extraction_performance.png

## 🧪 Experiments Included

CA-HepTh dataset with subset ratio 0.3

Model extraction at different query ratios

Pruning at different prune ratios (0.2, 0.4, 0.6, …)

C-ELEGANS dataset with subset ratio 0.3

Same robustness evaluation (model extraction + pruning).

## 📝 Notes

Code is based on PyTorch Geometric (PyG).

Currently implemented attacks: pruning and model extraction.

Future extensions: fine-tuning robustness, watermark overwriting, etc.

Default device is CPU (can switch to GPU by setting --device cuda).