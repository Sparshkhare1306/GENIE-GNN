#!/usr/bin/env python3
"""
test.py -- orchestrate the experiments so Bolin can run a single script to reproduce results.

It runs (in order):
  1) model extraction(s)
  2) pruning grid(s)
  3) evaluate surrogates (if you have evaluation scripts)
  4) aggregate pruning CSVs and plotting

It calls existing scripts by subprocess, so it preserves the implementation you already have.

Usage examples:
  python test.py --run_all
  python test.py --step extraction --datasets CA-HepTh C-ELEGANS --subsets 0.05 0.30 --query_ratios 0.05 0.1
  python test.py --step pruning --prune_rats 0.0 0.05 0.1 0.2 0.3
"""
import argparse
import subprocess
import os
import datetime
import shlex
import sys

# Default experiment settings (edit as needed)
DEFAULT_DATASETS = ["CA-HepTh", "C-ELEGANS"]
DEFAULT_SUBSETS = ["0.05", "0.30"]   # use strings the same way you name subset folders (0.05 -> subset_0_05)
DEFAULT_QUERY_RATIOS = ["0.05", "0.1", "0.2", "0.5"]
DEFAULT_PRUNE_RATIOS = ["0.0", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def run_cmd(cmd, logfile_path, env=None):
    """Run a command (list or string) and append output to logfile (text). Raises on error."""
    if isinstance(cmd, list):
        cmd_list = cmd
    else:
        # If passed a string, use shell-like splitting for readability
        cmd_list = shlex.split(cmd)
    timestamp = datetime.datetime.now().isoformat()
    with open(logfile_path, "a") as lf:
        lf.write(f"\n\n[{timestamp}] RUN: {' '.join(cmd_list)}\n")
        lf.flush()
        print(f"[RUNNING] {' '.join(cmd_list)}  -> logging to {logfile_path}")
        proc = subprocess.Popen(cmd_list, stdout=lf, stderr=subprocess.STDOUT, env=env)
        ret = proc.wait()
        lf.write(f"[{datetime.datetime.now().isoformat()}] RETCODE: {ret}\n")
    if ret != 0:
        raise RuntimeError(f"Command failed (rc={ret}): {' '.join(cmd_list)}. See {logfile_path}")

def make_env_with_root():
    e = os.environ.copy()
    # Ensure python can import your local packages
    e["PYTHONPATH"] = PROJECT_ROOT + (":" + e.get("PYTHONPATH", "" ) if e.get("PYTHONPATH") else "")
    # Make behavior reproducible where possible
    e["PYTHONHASHSEED"] = "0"
    return e

def cmd_model_extraction(dataset, subset_ratio, query_ratio, extra_args="", logfile=None):
    # call attacks/model_extraction.py with chosen args
    return [
        sys.executable, "-u", os.path.join(PROJECT_ROOT, "attacks", "model_extraction.py"),
        "--dataset", dataset,
        "--subset_ratio", str(subset_ratio),
        "--query_ratio", str(query_ratio),
        "--auto_pad_features",
        "--surrogate_epochs", "50",
        "--hidden_dim", "64"
    ] + (shlex.split(extra_args) if extra_args else [])

def cmd_pruning(dataset, subset_ratio, prune_ratio, extra_args="", logfile=None):
    return [
        sys.executable, "-u", "-m", "attacks.pruning_attack",
        "--dataset", dataset,
        "--subset_ratio", str(subset_ratio),
        "--prune_ratio", str(prune_ratio),
        "--save_pruned_model"
    ] + (shlex.split(extra_args) if extra_args else [])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=str, choices=["extraction","pruning","eval","aggregate","plot","all"], default="all",
                   help="Which step to run")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS,
                   help="Subset ratios (strings like 0.05 or 0.30)")
    p.add_argument("--query_ratios", nargs="+", default=["0.05"])
    p.add_argument("--prune_rats", nargs="+", default=DEFAULT_PRUNE_RATIOS)
    p.add_argument("--logdir", type=str, default=os.path.join("results", "logs"))
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = os.path.join(args.logdir, f"test_run_{ts}.log")

    env = make_env_with_root()

    # Step: extraction
    if args.step in ("extraction","all"):
        for ds in args.datasets:
            for ss in args.subsets:
                for qr in args.query_ratios:
                    cmd = cmd_model_extraction(ds, ss, qr, extra_args="--auto_pad_features")
                    print("PLAN:", " ".join(cmd))
                    if args.dry_run:
                        continue
                    run_cmd(cmd, run_log, env=env)

    # Step: pruning sweep
    if args.step in ("pruning","all"):
        for ds in args.datasets:
            for ss in args.subsets:
                for pr in args.prune_rats:
                    cmd = cmd_pruning(ds, ss, pr)
                    print("PLAN:", " ".join(cmd))
                    if args.dry_run:
                        continue
                    run_cmd(cmd, run_log, env=env)

    # Step: evaluate surrogates (if you have a script; optional)
    if args.step in ("eval","all"):
        # If you have shell scripts run_evaluate_surrogates_* in scripts/, call them here.
        eval_script = os.path.join(PROJECT_ROOT, "scripts", "run_evaluate_surrogates_CA-HepTh.sh")
        if os.path.exists(eval_script):
            if not args.dry_run:
                run_cmd(["bash", eval_script], run_log, env=env)
        else:
            print("[WARN] No evaluate_surrogates script found for CA-HepTh; skipping eval step")

    # Step: aggregate and plot
    if args.step in ("aggregate","all"):
        agg = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "aggregate_pruning_results.py")]
        plot = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "plot_pruning_results.py")]
        surrogate_plot = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "plot_surrogate_eval.py")]
        if not args.dry_run:
            run_cmd(agg, run_log, env=env)
            run_cmd(plot, run_log, env=env)
            # optional surrogate plot (if file exists)
            if os.path.exists(os.path.join(PROJECT_ROOT, "scripts", "plot_surrogate_eval.py")):
                run_cmd(surrogate_plot, run_log, env=env)

    print(f"[DONE] Orchestration finished. Log: {run_log}")

if __name__ == "__main__":
    main()
