"""
Training and evaluation engine.

Implements one-epoch training for plain MIL aggregators and for the
grade-adversarial GD-MIL backbone, plus out-of-fold (OOF) cross-validated
risk prediction with strict inner-validation early stopping.
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .losses import cox_loss
from .stats import cindex


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------
def train_one_epoch(model, bags, events, times, optimizer, device,
                    grades=None, lam=0.5, clip=1.0):
    """One pass over the training cohort (one slide per step).

    If `grades` is provided AND the model returns a (risk, grade) tuple, the
    grade-adversarial loss is added: L = L_cox(risk) + lam * MSE(grade, g).
    """
    model.train()
    risks, evs, tms = [], [], []
    adv_terms = []

    for i, x in enumerate(bags):
        x = x.to(device)
        out = model(x)

        if isinstance(out, tuple) and grades is not None:
            risk, grade_pred = out[0], out[1]
            g = torch.tensor(float(grades[i]), device=device)
            adv_terms.append((grade_pred - g) ** 2)
        else:
            risk = out[0] if isinstance(out, tuple) else out

        risks.append(risk.unsqueeze(0))
        evs.append(events[i])
        tms.append(times[i])

    risks = torch.cat(risks)
    evs = torch.stack(evs).to(device)
    tms = torch.stack(tms).to(device)

    loss = cox_loss(risks, evs, tms)
    if adv_terms:
        loss = loss + lam * torch.stack(adv_terms).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_risks(model, bags, device):
    """Return a 1-D numpy array of scalar risk scores (adversary discarded)."""
    model.eval()
    out = []
    for x in bags:
        x = x.to(device)
        r = model(x)
        r = r[0] if isinstance(r, tuple) else r
        out.append(float(r.item()))
    return np.asarray(out)


@torch.no_grad()
def evaluate(model, bags, events, times, device):
    """C-index on a held-out set."""
    risks = predict_risks(model, bags, device)
    return cindex(np.asarray(times), risks, np.asarray(events))


# ---------------------------------------------------------------------------
# Cross-validated OOF risk
# ---------------------------------------------------------------------------
def run_cv_oof(make_model, load_bag, cohort, in_dim, device,
               n_folds=5, epochs=30, lr=3e-4, weight_decay=1e-5,
               inner_val=0.15, lam=0.5, seed=42, use_grade_adv=False):
    """Five-fold stratified CV producing one OOF risk per patient.

    Parameters
    ----------
    make_model : callable () -> nn.Module
    load_bag   : callable (patient_id) -> FloatTensor [N, D]
    cohort     : DataFrame (patient_id, bcr_event, survival_time[, grade_group])
    use_grade_adv : if True, trains with the GD-MIL grade adversary

    Returns
    -------
    dict {patient_id: oof_risk}
    """
    pids = cohort["patient_id"].tolist()
    y = cohort["bcr_event"].astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = {}
    for tr_idx, te_idx in skf.split(pids, y):
        tr_pids = [pids[i] for i in tr_idx]
        te_pids = [pids[i] for i in te_idx]

        # inner validation split for early stopping
        tr_core, val_pids = train_test_split(
            tr_pids, test_size=inner_val, random_state=seed,
            stratify=cohort.set_index("patient_id").loc[tr_pids, "bcr_event"],
        )

        def pack(plist):
            sub = cohort.set_index("patient_id").loc[plist]
            bags = [load_bag(p) for p in plist]
            ev = [torch.tensor(float(sub.loc[p, "bcr_event"])) for p in plist]
            tm = [torch.tensor(float(sub.loc[p, "survival_time"])) for p in plist]
            gr = ([float(sub.loc[p, "grade_group"]) for p in plist]
                  if use_grade_adv and "grade_group" in sub.columns else None)
            return bags, ev, tm, gr

        tr_bags, tr_ev, tr_tm, tr_gr = pack(tr_core)
        val_bags, val_ev, val_tm, _ = pack(val_pids)
        te_bags, _, _, _ = pack(te_pids)

        model = make_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_c, best_state = -1.0, None
        for _ in range(epochs):
            train_one_epoch(model, tr_bags, tr_ev, tr_tm, opt, device,
                            grades=tr_gr, lam=lam)
            sched.step()
            c = evaluate(model, val_bags, val_ev, val_tm, device)
            if c > best_c:
                best_c = c
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        risks = predict_risks(model, te_bags, device)
        for p, r in zip(te_pids, risks):
            oof[p] = float(r)

    return oof
