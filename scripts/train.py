#!/usr/bin/env python
"""
Train a single model under five-fold CV and save out-of-fold risks.

Examples
--------
# Imaging baseline
python scripts/train.py --model abmil --fm uni2h --out results/abmil_uni2h.json

# GD-MIL backbone (grade-adversarial), then fuse with clinical variables
python scripts/train.py --model gadvmil --fm uni2h --grade-adv \
    --fuse --out results/fusion_gdmil.json

# Clinical-only Cox baseline
python scripts/train.py --clinical-only --out results/clinical.json
"""

import argparse
import glob
import json
import os

import pandas as pd
import torch

from gdmil import get_model, run_cv_oof, fit_fusion_oof, fit_clinical_oof

FM_DIM = {"uni2h": 1536, "virchow2": 1280, "resnet50": 512}


def build_bag_loader(emb_dir):
    cache = {}
    for bf in glob.glob(os.path.join(emb_dir, "*.pt")):
        pid = "-".join(os.path.basename(bf).replace(".pt", "").split("-")[:3])
        if pid not in cache:
            d = torch.load(bf, map_location="cpu", weights_only=False)
            cache[pid] = d["features"].float()
    return lambda pid: cache[pid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="abmil",
                    choices=["abmil", "clam", "transmil", "patchgcn", "gadvmil"])
    ap.add_argument("--fm", default="uni2h", choices=list(FM_DIM))
    ap.add_argument("--data-root", default="data/TCGA_PRAD")
    ap.add_argument("--cohort", default="data/TCGA_PRAD/cohort.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grade-adv", action="store_true",
                    help="train with the GD-MIL grade adversary")
    ap.add_argument("--fuse", action="store_true",
                    help="late-fuse OOF imaging risk with clinical variables")
    ap.add_argument("--clinical-only", action="store_true",
                    help="fit the clinical Cox baseline and exit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cohort = pd.read_csv(args.cohort)
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- clinical-only baseline ----------------------------------------
    if args.clinical_only:
        oof = fit_clinical_oof(cohort, seed=args.seed)
        json.dump({"oof_risk": oof}, open(args.out, "w"), indent=2)
        print(f"Saved clinical OOF risks -> {args.out}")
        return

    # ---- imaging / GD-MIL backbone -------------------------------------
    in_dim = FM_DIM[args.fm]
    emb_dir = os.path.join(args.data_root, f"embeddings_{args.fm}")
    load_bag = build_bag_loader(emb_dir)

    make_model = lambda: get_model(args.model, in_dim,
                                   **({"lam": args.lam} if args.model == "gadvmil" else {}))

    img_oof = run_cv_oof(
        make_model, load_bag, cohort, in_dim, device,
        epochs=args.epochs, lr=args.lr, lam=args.lam, seed=args.seed,
        use_grade_adv=args.grade_adv,
    )

    if args.fuse:
        fused = fit_fusion_oof(cohort, img_oof, seed=args.seed)
        json.dump({"oof_risk": fused, "oof_img_risk": img_oof},
                  open(args.out, "w"), indent=2)
        print(f"Saved fused GD-MIL OOF risks -> {args.out}")
    else:
        json.dump({"oof_risk": img_oof}, open(args.out, "w"), indent=2)
        print(f"Saved imaging OOF risks -> {args.out}")


if __name__ == "__main__":
    main()
