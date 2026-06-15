"""
Statistical utilities: bootstrap confidence intervals and paired bootstrap
significance tests for the censored concordance index.

The C-index is computed with lifelines' `concordance_index`, which expects
risk to be passed with a sign such that higher risk -> earlier event. We pass
`-risk` to match that convention.
"""

import numpy as np
from lifelines.utils import concordance_index


def cindex(times, risks, events):
    """Censored concordance index for a risk score (higher risk = worse)."""
    return concordance_index(times, -np.asarray(risks), events)


def bootstrap_ci(times, risks, events, n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for the C-index over patient-level resamples.

    Returns
    -------
    point, ci_lo, ci_hi
    """
    times = np.asarray(times)
    risks = np.asarray(risks)
    events = np.asarray(events)
    rng = np.random.default_rng(seed)
    n = len(times)

    point = cindex(times, risks, events)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # skip degenerate resamples with no admissible pairs / no events
        if events[idx].sum() == 0:
            continue
        boots.append(cindex(times[idx], risks[idx], events[idx]))

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def paired_bootstrap_test(times, risk_a, risk_b, events, n_boot=2000, seed=42):
    """Paired bootstrap test of H0: C(b) - C(a) = 0.

    `risk_a` is the comparator, `risk_b` the method of interest. Returns the
    observed delta, its 95% CI, and a two-sided p-value (fraction of bootstrap
    deltas that fall on the opposite side of zero, doubled).

    Returns
    -------
    dict with keys: c_a, c_b, delta, ci_lo, ci_hi, p_value
    """
    times = np.asarray(times)
    risk_a = np.asarray(risk_a)
    risk_b = np.asarray(risk_b)
    events = np.asarray(events)
    rng = np.random.default_rng(seed)
    n = len(times)

    c_a = cindex(times, risk_a, events)
    c_b = cindex(times, risk_b, events)
    delta = c_b - c_a

    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if events[idx].sum() == 0:
            continue
        da = cindex(times[idx], risk_a[idx], events[idx])
        db = cindex(times[idx], risk_b[idx], events[idx])
        boot_deltas.append(db - da)
    boot_deltas = np.asarray(boot_deltas)

    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    # two-sided p: reflect around zero
    frac = np.mean(boot_deltas <= 0) if delta > 0 else np.mean(boot_deltas >= 0)
    p_value = float(min(1.0, 2 * frac))

    return {
        "c_a": float(c_a),
        "c_b": float(c_b),
        "delta": float(delta),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
    }
