# GD-MIL: Method Details

This document describes the GD-MIL pipeline in detail. It complements the
top-level README and the paper.

---

## 1. Problem setup

After radical prostatectomy, biochemical recurrence (BCR) is a right-censored
survival endpoint. For each patient we observe a follow-up time `t` and an
event indicator `δ ∈ {0, 1}` (1 = BCR occurred, 0 = censored). The goal is a
risk score that ranks patients by recurrence hazard, measured by the censored
concordance index (C-index).

A whole-slide image is tiled into `N` patches, each embedded by a **frozen**
foundation model `φ` into a feature `f_i ∈ R^d` (`d = 1536` for UNI2-h,
`1280` for Virchow2, `512` for ResNet50). Up to 2000 tiles per slide are kept.

---

## 2. Gated-attention MIL backbone

Each tile feature is projected to a hidden representation `h_i ∈ R^256`. Gated
attention computes a normalized weight per tile:

```
a_i = softmax_i( w^T ( tanh(V h_i) ⊙ σ(U h_i) ) )
```

with `V, U ∈ R^{128×256}` and `w ∈ R^{128}`. The two gates play complementary
roles — `tanh` scores relevance, `σ` scores inclusion — and together suppress
uninformative tiles more effectively than a single gate. The slide
representation is the layer-normalized attention-weighted sum:

```
z = LayerNorm( Σ_i a_i h_i ) ∈ R^256
```

---

## 3. Grade-adversarial disentanglement

Much of the morphological signal an imaging model learns is a proxy for Gleason
grade — the variable clinicians already record. To steer the encoder toward
**complementary** signal, we attach a grade adversary through a Gradient
Reversal Layer (GRL).

- **Main branch.** A linear Cox head maps `z` to an imaging risk
  `r_img = φ(z)`, trained with the Cox partial likelihood.
- **Adversarial branch (training only).** A two-layer MLP `ψ` predicts the
  standardized ISUP grade `g` from `z`, but through a GRL `R_λ`:

```
ĝ = ψ( R_λ(z) ),     ∂R_λ / ∂z = −λ I
```

On the forward pass `R_λ` is the identity; on the backward pass it negates and
scales the gradient by `−λ`. The encoder therefore receives two opposing
signals: keep `z` predictive of BCR, while making `z` uninformative about
grade. The combined objective is:

```
L = L_cox(r_img) + λ · || ĝ − g ||²₂ ,   λ = 0.5
```

This **adversarially discourages** grade information from `z` rather than
formally removing it; residual grade information is not verified at inference
(see Limitations in the paper).

At inference the adversarial branch is discarded.

---

## 4. Cox partial likelihood

Given risk scores `r_i`, events `δ_i`, and times `t_i`:

```
L_cox = − (1 / Σ_i δ_i) · Σ_{i: δ_i = 1} [ r_i − log Σ_{j: t_j ≥ t_i} exp(r_j) ]
```

Implemented stably with `logcumsumexp` after sorting by descending time
(see `gdmil/losses.py`).

---

## 5. Late multimodal fusion

The grade-disentangled imaging risk `r_img` is concatenated with the clinical
variables (grade group, T-stage, age) and an L2-penalized Cox model
(`penalizer = 0.1`) produces the final GD-MIL risk score.

**Leakage-free fusion.** The fusion Cox is fitted **only on out-of-fold
imaging risks** — never on in-fold predictions — so the leakage-free guarantee
extends through the entire pipeline (see `gdmil/fusion.py`).

---

## 6. Leakage-free evaluation

A common pitfall inflates reported concordance: selecting the model checkpoint
on the same fold used for final scoring. We avoid this entirely.

- 5-fold stratified cross-validation (stratified by event status).
- Within each training fold, a 15% inner split is held out **only** for early
  stopping; the outer test fold is never touched for selection.
- Each patient gets a single out-of-fold (OOF) risk from the fold in which it
  was held out; the C-index is computed once over all 487 OOF predictions.
- Repeated over 5 seeds; we report mean ± std.
- Pairwise significance: paired bootstrap, 2000 patient-level resamples.

See `gdmil/engine.py::run_cv_oof` and `gdmil/stats.py`.

---

## 7. Descriptor / tensor shapes

| Symbol | Shape        | Meaning                                  |
| ------ | ------------ | ---------------------------------------- |
| `f_i`  | `[d]`        | frozen tile embedding                    |
| `h_i`  | `[256]`      | projected hidden representation          |
| `a_i`  | `[1]`        | tile attention weight (Σ a_i = 1)        |
| `z`    | `[256]`      | slide representation (LayerNorm'd)       |
| `r_img`| scalar       | imaging risk (main branch)               |
| `ĝ`    | scalar       | grade-adversary logit (training only)    |
