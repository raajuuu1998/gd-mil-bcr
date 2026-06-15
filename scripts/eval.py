#!/usr/bin/env python
"""
Aggregate OOF result files into the benchmark table (Table 1) and the paired
significance tests (Table 2).

Reads every results/*.json (each a dict with key "oof_risk": {pid: risk}) and
the cohort CSV, then prints C-index with 95% bootstrap CI per method and the
paired bootstrap comparisons reported in the paper.

Example
-------
python scripts/eval.py --results_dir results --cohort data/TCGA_PRAD/cohort.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from gdmil import bootstrap_ci, paired_bootstrap_test

# display name -> results filename (without extension)
METHODS = [
    ("Clinical Cox",     "clinical"),
    ("ResNet50 + ABMIL", "abmil_resnet50"),
    ("UNI2-h + CLAM",    "clam_uni2h"),
    ("UNI2-h + ABMIL",   "abmil_uni2h"),
    ("UNI2-h + TransMIL", "transmil_uni2h"),
    ("UNI2-h + PatchGCN", "patchgcn_uni2h"),
    ("Virchow2 + ABMIL", "abmil_virchow2"),
    ("GD-MIL (ours)",    "fusion_gdmil"),
]

# paired comparisons reported in Table 2: (method, comparator)
COMPARISONS = [
    ("GD-MIL (ours)", "Clinical Cox"),
    ("GD-MIL (ours)", "UNI2-h + ABMIL"),
    ("GD-MIL (ours)", "Virchow2 + ABMIL"),
    ("Virchow2 + ABMIL", "Clinical Cox"),
    ("UNI2-h + ABMIL", "Clinical Cox"),
]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--cohort", default="data/TCGA_PRAD/cohort.csv")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    t = dict(zip(cohort.patient_id, cohort.survival_time.astype(float)))
    e = dict(zip(cohort.patient_id, cohort.bcr_event.astype(float)))

    risks = {}
    for name, fname in METHODS:
        path = os.path.join(args.results_dir, f"{fname}.json")
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        risks[name] = json.load(open(path))["oof_risk"]

    # ---- Table 1: C-index + bootstrap CI -------------------------------
    print("\n=== Table 1: C-index (95% bootstrap CI) ===")
    print(f"{'Method':22s}  {'C-index':>8s}  {'95% CI':>20s}")
    for name, _ in METHODS:
        if name not in risks:
            continue
        r = risks[name]
        pids = [p for p in cohort.patient_id if p in r]
        times = np.array([t[p] for p in pids])
        evs = np.array([e[p] for p in pids])
        rk = np.array([r[p] for p in pids])
        c, lo, hi = bootstrap_ci(times, rk, evs, n_boot=args.n_boot)
        print(f"{name:22s}  {c:8.3f}  [{lo:6.3f}, {hi:6.3f}]")

    # ---- Table 2: paired bootstrap significance ------------------------
    print("\n=== Table 2: paired bootstrap tests ===")
    print(f"{'Method':18s}  {'Comparator':18s}  {'dc':>7s}  {'95% CI':>20s}  {'p':>9s}")
    for a, b in COMPARISONS:
        if a not in risks or b not in risks:
            continue
        pids = [p for p in cohort.patient_id if p in risks[a] and p in risks[b]]
        times = np.array([t[p] for p in pids])
        evs = np.array([e[p] for p in pids])
        ra = np.array([risks[b][p] for p in pids])   # comparator
        rb = np.array([risks[a][p] for p in pids])   # method of interest
        res = paired_bootstrap_test(times, ra, rb, evs, n_boot=args.n_boot)
        print(f"{a:18s}  {b:18s}  {res['delta']:+7.3f}  "
              f"[{res['ci_lo']:+6.3f}, {res['ci_hi']:+6.3f}]  "
              f"{res['p_value']:8.4f}{stars(res['p_value'])}")


if __name__ == "__main__":
    main()
