#!/usr/bin/env python
"""
Regenerate the C-index comparison bar chart (Figure 2) and the Kaplan-Meier
risk-stratification curve (Figure 3) from saved OOF result files.

Attention maps (Figures 4-5) require the raw WSI and tile coordinates and are
produced separately; they are not regenerated here.

Example
-------
python scripts/make_figures.py --results_dir results \
    --cohort data/TCGA_PRAD/cohort.csv --assets_dir assets
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from gdmil import bootstrap_ci

NAVY = "#1f3a5f"
ORANGE = "#c0392b"
GRAY = "#4b5563"

METHODS = [
    ("Clinical Cox",     "clinical",        GRAY),
    ("ResNet50 + ABMIL", "abmil_resnet50",  GRAY),
    ("UNI2-h + CLAM",    "clam_uni2h",      GRAY),
    ("UNI2-h + ABMIL",   "abmil_uni2h",     GRAY),
    ("UNI2-h + TransMIL", "transmil_uni2h", GRAY),
    ("UNI2-h + PatchGCN", "patchgcn_uni2h", GRAY),
    ("Virchow2 + ABMIL", "abmil_virchow2",  GRAY),
    ("GD-MIL (ours)",    "fusion_gdmil",    ORANGE),
]


def load(results_dir, fname):
    path = os.path.join(results_dir, f"{fname}.json")
    return json.load(open(path))["oof_risk"] if os.path.exists(path) else None


def fig_cindex(results_dir, cohort, assets_dir, n_boot):
    t = dict(zip(cohort.patient_id, cohort.survival_time.astype(float)))
    e = dict(zip(cohort.patient_id, cohort.bcr_event.astype(float)))

    names, cs, los, his, colors = [], [], [], [], []
    clin_c = None
    for name, fname, color in METHODS:
        r = load(results_dir, fname)
        if r is None:
            continue
        pids = [p for p in cohort.patient_id if p in r]
        times = np.array([t[p] for p in pids])
        evs = np.array([e[p] for p in pids])
        rk = np.array([r[p] for p in pids])
        c, lo, hi = bootstrap_ci(times, rk, evs, n_boot=n_boot)
        names.append(name); cs.append(c); los.append(lo); his.append(hi)
        colors.append(color)
        if name == "Clinical Cox":
            clin_c = c

    order = np.argsort(cs)
    names = [names[i] for i in order]
    cs = [cs[i] for i in order]
    los = [los[i] for i in order]
    his = [his[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    yy = np.arange(len(names))
    ax.barh(yy, cs, color=colors, height=0.62, zorder=3)
    err = [np.array(cs) - np.array(los), np.array(his) - np.array(cs)]
    ax.errorbar(cs, yy, xerr=err, fmt="none", ecolor=NAVY, capsize=3, lw=1.2, zorder=4)
    for y, c in zip(yy, cs):
        ax.text(c + 0.004, y, f"{c:.3f}", va="center", fontsize=9, color="#222")
    if clin_c is not None:
        ax.axvline(clin_c, ls="--", color=NAVY, lw=1, zorder=2,
                   label=f"Clinical Cox baseline")
        ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Concordance Index (C-index)", fontsize=11)
    ax.set_xlim(0.47, 0.80)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(assets_dir, "plot1_cindex.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig_km(results_dir, cohort, assets_dir):
    r = load(results_dir, "fusion_gdmil")
    if r is None:
        print("  [skip] fusion_gdmil.json not found, cannot draw KM")
        return
    df = cohort.copy()
    df = df[df.patient_id.isin(r)].reset_index(drop=True)
    df["risk"] = df.patient_id.map(r)
    med = df["risk"].median()
    hi = df[df.risk > med]
    lo = df[df.risk <= med]

    lr = logrank_test(hi.survival_time, lo.survival_time,
                      hi.bcr_event, lo.bcr_event)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    km = KaplanMeierFitter()
    km.fit(hi.survival_time, hi.bcr_event,
           label=f"High Risk (n={len(hi)}, events={int(hi.bcr_event.sum())})")
    km.plot_survival_function(ax=ax, color=ORANGE, ci_alpha=0.15)
    km.fit(lo.survival_time, lo.bcr_event,
           label=f"Low Risk  (n={len(lo)}, events={int(lo.bcr_event.sum())})")
    km.plot_survival_function(ax=ax, color=NAVY, ci_alpha=0.15)

    ax.text(0.97, 0.97, f"p < 0.0001" if lr.p_value < 1e-4 else f"p = {lr.p_value:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#999"))
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("BCR-Free Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(assets_dir, "plot2_km.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}  (log-rank p={lr.p_value:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--cohort", default="data/TCGA_PRAD/cohort.csv")
    ap.add_argument("--assets_dir", default="assets")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.assets_dir, exist_ok=True)
    cohort = pd.read_csv(args.cohort)
    cohort["patient_id"] = cohort["patient_id"].astype(str)

    print("Regenerating figures...")
    fig_cindex(args.results_dir, cohort, args.assets_dir, args.n_boot)
    fig_km(args.results_dir, cohort, args.assets_dir)
    print("Done.")


if __name__ == "__main__":
    main()
