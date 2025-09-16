#!/bin/bash

DATASETS=("CA-HepTh" "C-ELEGANS")  # extend to AMAZON later if needed
SUBSET_RATIO=0.3                   # keep consistent with training
PRUNE_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for DATASET in "${DATASETS[@]}"; do
  for PRUNE in "${PRUNE_RATIOS[@]}"; do
    echo "Running pruning on $DATASET with prune_ratio=$PRUNE ..."
    python attacks/pruning_attack.py \
      --dataset $DATASET \
      --subset_ratio $SUBSET_RATIO \
      --prune_ratio $PRUNE
  done
done
