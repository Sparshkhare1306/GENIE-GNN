# GENIE-GNN: Watermarking Graph Neural Networks for Link Prediction

This repository contains a reproduction and extension of the paper:

> **GENIE: Watermarking Graph Neural Networks for Link Prediction**  
> [Paper Link](https://arxiv.org/abs/2406.04805)

The goal is to embed robust and verifiable watermarks in GNN models for **link prediction** tasks. The watermark should survive **model extraction**, **pruning**, and **fine-tuning** attacks, while preserving the model’s performance on the original task.

---

## 📌 Summary

This project:

- Reproduces GENIE baseline using Node2Vec embeddings + GCN on standard graph datasets.
- Implements watermark embedding and detection as described in the paper.
- Tests watermark robustness under:
  - Model extraction attacks (partial data)
  - Weight pruning
  - Finetuning
- Includes experimental results and visualizations.

---

## 🗂️ Project Structure

```bash
GENIE-GNN/
├── data/Snap/                # Graph datasets (Amazon, ca-HepTh, C-ELEGANS)
├── datasets/                 # Preprocessing and watermarking logic
├── models/                   # GCN model for link prediction
├── results/                  # Outputs, logs, visualizations, metrics
├── token.txt                 # Removed/ignored (do not track)
├── cached_model.pth          # Trained model weights
├── watermarked_model.pth     # Model after watermark embedding
├── .gitignore
├── README.md
```
## 📥 Installation

## Clone the repository
```bash
git clone git@github.com:Sparshkhare1306/GENIE-GNN.git
cd GENIE-GNN
```
## Set up Python environment
```bash
conda create -n genie_venv python=3.10
conda activate genie_venv
```

## Install dependencies
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install matplotlib numpy scikit-learn tqdm
pip install scipy networkx
```
## 📊 Datasets Used

1. ca-HepTh.txt — Collaboration network from arXiv

2. c_elegans.mtx — C. Elegans neural network

3. amazon_co_purchase.txt — Amazon co-purchase network


All datasets are stored under data/Snap/.


## 🧪 Running Experiments

You can run experiments for watermarking and robustness testing by executing scripts like:
``` bash
python run_experiment.py --dataset hepth --attack pruning
```
(Currently scripts are modularized — refer to each script in datasets/ and models/ for functionality.)

You may implement or invoke:

- Watermark embedding (datasets/watermark.py)

- Link prediction using GCN (models/gcn_link_predictor.py)

- AUC and performance metrics

- Pruning attacks and subset extraction

## 📈 Results Overview
Watermark robustness visualizations are available:

- results/CA-HepTh/auc_vs_ratio.png

- results/C-ELEGANS/auc_vs_ratio.png

- results/combined_auc_vs_ratio.png

## Each folder under results/ also contains:

- config.txt – experimental setup

- metrics.csv, metrics_finetune.csv – performance logs

- log.txt – runtime logs

- watermarked_model.pth – saved model with embedded watermark

## 📁 Example Metrics Output
| Dataset   | Subset Ratio | Test AUC | Watermark AUC | Survived Attack |
| --------- | ------------ | -------- | ------------- | --------------- |
| CA-HepTh  | 0.1          | 0.925    | 0.89          | ✅ Yes           |
| C-ELEGANS | 0.3          | 0.891    | 0.87          | ✅ Yes           |


🔐 Security and .gitignore
- This repository uses .gitignore to:

- Exclude sensitive files (e.g., token.txt)

- Ignore cache files and system artifacts

- Avoid committing large datasets or model binaries unintentionally

🧠 Key Concepts
- Link Prediction: Predict if an edge exists between two nodes in a graph.

- Graph Watermarking: Embed a small pattern or substructure into a GNN’s training process to prove ownership.

- Model Extraction: Attack where adversaries replicate a model using limited access to it.

- Pruning: Removing weights or neurons to reduce model size.

- AUC: Area under ROC curve, measures classification performance.





