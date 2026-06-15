"""
GD-MIL: Grade-Disentangled Multiple Instance Learning for multimodal
biochemical recurrence prediction in prostate cancer.
"""

from .models import (
    ABMIL,
    CLAM,
    TransMIL,
    PatchGCN,
    GAdvMIL,
    GradReverse,
    grad_rev,
    get_model,
)
from .data import BCRDataset, collate_fn, get_emb_path
from .losses import cox_loss
from .engine import (
    train_one_epoch,
    evaluate,
    predict_risks,
    run_cv_oof,
)
from .fusion import fit_fusion_oof, fit_clinical_oof
from .stats import cindex, bootstrap_ci, paired_bootstrap_test

__version__ = "1.0.0"

__all__ = [
    "ABMIL", "CLAM", "TransMIL", "PatchGCN", "GAdvMIL",
    "GradReverse", "grad_rev", "get_model",
    "BCRDataset", "collate_fn", "get_emb_path",
    "cox_loss",
    "train_one_epoch", "evaluate", "predict_risks", "run_cv_oof",
    "fit_fusion_oof", "fit_clinical_oof",
    "cindex", "bootstrap_ci", "paired_bootstrap_test",
]
