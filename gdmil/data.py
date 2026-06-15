"""
Dataset and collation for BCR survival MIL.

Operates on precomputed tile embeddings stored as per-patient .pt files. Each
.pt is a dict with at least:
    features   : FloatTensor [N, D]   tile embeddings
    coords     : FloatTensor [N, 2]   tile (x, y) pixel coordinates
plus clinical/label fields supplied separately via the cohort CSV.
"""

import os
import torch
from torch.utils.data import Dataset


def get_emb_path(patient_id, emb_dir):
    """Resolve the .pt embedding file for a patient_id inside emb_dir."""
    for f in os.listdir(emb_dir):
        if f.startswith(patient_id) and f.endswith(".pt"):
            return os.path.join(emb_dir, f)
    return None


class BCRDataset(Dataset):
    """Bag-level dataset for biochemical-recurrence survival.

    Parameters
    ----------
    patient_ids : list[str]
    cohort_df   : pandas.DataFrame with columns
                  patient_id, bcr_event, survival_time
                  (+ grade_group, t_stage, age for the fusion stage)
    emb_dir     : directory of per-patient .pt embedding bags
    """

    def __init__(self, patient_ids, cohort_df, emb_dir, verbose=False):
        self.samples = []
        for pid in patient_ids:
            pt_path = get_emb_path(pid, emb_dir)
            if pt_path is None:
                continue
            row = cohort_df[cohort_df["patient_id"] == pid].iloc[0]
            self.samples.append(
                {
                    "patient_id": pid,
                    "pt_path": pt_path,
                    "bcr_event": float(row["bcr_event"]),
                    "survival_time": float(row["survival_time"]),
                }
            )
        if verbose:
            print(f"  Dataset built: {len(self.samples)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = torch.load(s["pt_path"], map_location="cpu", weights_only=False)
        feats = data["features"].float()                       # [N, D]
        coords = data.get("coords", torch.zeros(feats.shape[0], 2)).float()
        return (
            feats,
            coords,
            torch.tensor(s["bcr_event"], dtype=torch.float32),
            torch.tensor(s["survival_time"], dtype=torch.float32),
            s["patient_id"],
        )


def collate_fn(batch):
    """One-slide-per-step collation (bags have variable N)."""
    feats, coords, events, times, pids = zip(*batch)
    return feats, coords, torch.stack(events), torch.stack(times), pids
