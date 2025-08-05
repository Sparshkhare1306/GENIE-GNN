# experiments/run_finetune.py
import sys
import os

import argparse
from attacks.fine_tune import run_fine_tuning
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CA-HepTh", "C-ELEGANS"],
        default="CA-HepTh",
        help="Dataset to use"
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.1,
        help="Subset ratio used during watermarking"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="Which model variant to use: 'v1' = dot-product, 'v2' = MLP"
    )

    args = parser.parse_args()
    run_fine_tuning(dataset=args.dataset, subset_ratio=args.subset_ratio, model_variant=args.model_variant)
