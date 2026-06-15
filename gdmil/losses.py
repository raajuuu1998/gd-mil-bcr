"""Survival losses for MIL training."""

import torch


def cox_loss(risk_scores, events, times):
    """Negative Cox partial log-likelihood (Breslow approximation).

    Parameters
    ----------
    risk_scores : FloatTensor [B]   predicted risk (higher = higher hazard)
    events      : FloatTensor [B]   1 = BCR event occurred, 0 = censored
    times       : FloatTensor [B]   follow-up time in days

    Returns
    -------
    scalar loss. Encourages patients with earlier events to receive higher
    risk than their comparable (later or censored) pairs, which directly
    optimises the concordance objective.
    """
    order = torch.argsort(times, descending=True)
    risk = risk_scores[order]
    ev = events[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    ll = -(risk - log_cumsum)[ev.bool()]

    if ll.numel() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
    return ll.mean()
