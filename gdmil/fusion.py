"""
Late multimodal fusion.

Fits an L2-penalized Cox model on the out-of-fold imaging risk produced by
the grade-adversarial backbone, concatenated with the clinical variables
(grade group, T-stage, age). Fusion is always fitted on OOF imaging scores so
the leakage-free guarantee extends through the full pipeline.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

CLIN_COLS = ["grade_group", "t_stage", "age"]


def fit_fusion_oof(cohort, oof_img_risk, penalizer=0.1, n_folds=5, seed=42):
    """Produce final fused OOF risk scores.

    Parameters
    ----------
    cohort       : DataFrame with patient_id, bcr_event, survival_time,
                   grade_group, t_stage, age
    oof_img_risk : dict {patient_id: imaging_risk} from the GD-MIL backbone

    Returns
    -------
    dict {patient_id: fused_risk}
    """
    from sklearn.model_selection import StratifiedKFold

    df = cohort.copy()
    df = df[df["patient_id"].isin(oof_img_risk)].reset_index(drop=True)
    df["img_risk"] = df["patient_id"].map(oof_img_risk)

    feat_cols = CLIN_COLS + ["img_risk"]
    y = df["bcr_event"].astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fused = {}
    for tr_idx, te_idx in skf.split(df, y):
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

        mu = tr[feat_cols].mean()
        sd = tr[feat_cols].std().replace(0, 1.0)
        tr[feat_cols] = (tr[feat_cols] - mu) / sd
        te[feat_cols] = (te[feat_cols] - mu) / sd

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(tr[feat_cols + ["survival_time", "bcr_event"]],
                duration_col="survival_time", event_col="bcr_event")

        hz = cph.predict_partial_hazard(te[feat_cols])
        for p, r in zip(te["patient_id"].values, np.asarray(hz).ravel()):
            fused[p] = float(r)

    return fused


def fit_clinical_oof(cohort, penalizer=0.1, n_folds=5, seed=42):
    """Clinical-only Cox baseline producing OOF risks (grade, T, age)."""
    from sklearn.model_selection import StratifiedKFold

    df = cohort.copy()
    y = df["bcr_event"].astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = {}
    for tr_idx, te_idx in skf.split(df, y):
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        mu = tr[CLIN_COLS].mean()
        sd = tr[CLIN_COLS].std().replace(0, 1.0)
        tr[CLIN_COLS] = (tr[CLIN_COLS] - mu) / sd
        te[CLIN_COLS] = (te[CLIN_COLS] - mu) / sd

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(tr[CLIN_COLS + ["survival_time", "bcr_event"]],
                duration_col="survival_time", event_col="bcr_event")
        hz = cph.predict_partial_hazard(te[CLIN_COLS])
        for p, r in zip(te["patient_id"].values, np.asarray(hz).ravel()):
            oof[p] = float(r)
    return oof
