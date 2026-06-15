"""
MIL aggregators and the GD-MIL grade-adversarial backbone.

All models take a bag of tile features [N, D] and produce a scalar Cox risk
score. GAdvMIL additionally produces a grade-adversary logit (training only).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin et al., JMLR 2016)
# ---------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    """Identity on the forward pass; negates and scales the gradient on the
    backward pass. This is the mechanism that drives grade disentanglement."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_rev(x, lam=0.5):
    return GradReverse.apply(x, lam)


# ---------------------------------------------------------------------------
# 1. ABMIL  (Ilse et al., ICML 2018)
# ---------------------------------------------------------------------------
class ABMIL(nn.Module):
    """Attention-based MIL with gated attention pooling."""

    def __init__(self, in_dim, h=256, d=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(d))
        self.V = nn.Sequential(nn.Linear(h, 128), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(h, 128), nn.Sigmoid())
        self.w = nn.Linear(128, 1)
        self.head = nn.Linear(h, 1)

    def forward(self, x, return_attn=False):
        h = self.enc(x)
        a = torch.softmax(self.w(self.V(h) * self.U(h)), 0)  # [N, 1]
        z = (a * h).sum(0)
        risk = self.head(z).squeeze()
        if return_attn:
            return risk, a.squeeze()
        return risk


# ---------------------------------------------------------------------------
# 2. CLAM  (Lu et al., Nature BME 2021)
# ---------------------------------------------------------------------------
class CLAM(nn.Module):
    """Clustering-constrained attention MIL, adapted for Cox survival."""

    def __init__(self, in_dim, h=256, d=0.25, n_inst=8):
        super().__init__()
        self.n_inst = n_inst
        self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(d))
        self.V = nn.Sequential(nn.Linear(h, 128), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(h, 128), nn.Sigmoid())
        self.w = nn.Linear(128, 1)
        self.head = nn.Linear(h, 1)
        self.inst = nn.Linear(h, 2)

    def forward(self, x, return_inst=False):
        h = self.enc(x)
        a = torch.softmax(self.w(self.V(h) * self.U(h)), 0)
        z = (a * h).sum(0)
        risk = self.head(z).squeeze()
        if return_inst:
            af = a.squeeze()
            top = torch.topk(af, min(self.n_inst, af.numel())).indices
            bot = torch.topk(af, min(self.n_inst, af.numel()), largest=False).indices
            return risk, a.squeeze(), self.inst(h[top]), self.inst(h[bot])
        return risk


# ---------------------------------------------------------------------------
# 3. TransMIL  (Shao et al., NeurIPS 2021)
# ---------------------------------------------------------------------------
class TransLayer(nn.Module):
    def __init__(self, h=256, n_heads=8, d=0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, n_heads, dropout=d, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(h, h * 4), nn.GELU(), nn.Dropout(d),
            nn.Linear(h * 4, h), nn.Dropout(d),
        )

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class TransMIL(nn.Module):
    """Transformer-based correlated MIL with a learnable class token."""

    def __init__(self, in_dim, h=256, n_heads=8, n_layers=2, d=0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(d)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, h))
        self.layers = nn.ModuleList([TransLayer(h, n_heads, d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        x = self.proj(x).unsqueeze(0)               # [1, N, h]
        cls = self.cls_token.expand(1, -1, -1)      # [1, 1, h]
        x = torch.cat([cls, x], dim=1)              # [1, N+1, h]
        for layer in self.layers:
            x = layer(x)
        z = self.norm(x)[0, 0]                       # class-token output
        return self.head(z).squeeze()


# ---------------------------------------------------------------------------
# 4. PatchGCN  (Chen et al., MICCAI 2021) — lightweight spatial-graph variant
# ---------------------------------------------------------------------------
class PatchGCN(nn.Module):
    """Spatial-graph MIL. A compact message-passing variant that aggregates
    over k-nearest spatial neighbours; adapted for Cox survival."""

    def __init__(self, in_dim, h=256, d=0.25, k=8):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(d))
        self.gcn = nn.Linear(h, h)
        self.V = nn.Sequential(nn.Linear(h, 128), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(h, 128), nn.Sigmoid())
        self.w = nn.Linear(128, 1)
        self.head = nn.Linear(h, 1)

    def forward(self, x, coords=None):
        h = self.enc(x)
        if coords is not None and h.shape[0] > self.k:
            # one round of mean message passing over k-NN spatial neighbours
            with torch.no_grad():
                dist = torch.cdist(coords.float(), coords.float())
                nn_idx = dist.topk(self.k + 1, largest=False).indices[:, 1:]
            msg = h[nn_idx].mean(1)
            h = torch.relu(self.gcn(h + msg))
        a = torch.softmax(self.w(self.V(h) * self.U(h)), 0)
        z = (a * h).sum(0)
        return self.head(z).squeeze()


# ---------------------------------------------------------------------------
# 5. GAdvMIL  —  the GD-MIL imaging backbone (ours)
# ---------------------------------------------------------------------------
class GAdvMIL(nn.Module):
    """Grade-adversarial gated-attention MIL.

    Produces a slide-level imaging risk while a gradient-reversal grade
    adversary adversarially discourages grade information from the slide
    representation z. At inference the adversary is unused.

    forward returns (risk, grade_adv_logit). For inference use the risk only.
    """

    def __init__(self, in_dim, h=256, d=0.25, lam=0.5):
        super().__init__()
        self.lam = lam
        self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(d))
        self.V = nn.Sequential(nn.Linear(h, 128), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(h, 128), nn.Sigmoid())
        self.w = nn.Linear(128, 1)
        self.zn = nn.LayerNorm(h)
        self.risk = nn.Linear(h, 1)
        self.adv = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, return_attn=False):
        h = self.enc(x)
        a = torch.softmax(self.w(self.V(h) * self.U(h)), 0)
        z = self.zn((a * h).sum(0))
        risk = self.risk(z).squeeze()
        grade = self.adv(grad_rev(z, self.lam)).squeeze()
        if return_attn:
            return risk, grade, a.squeeze()
        return risk, grade


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_model(name, in_dim, **kwargs):
    name = name.lower()
    table = {
        "abmil": ABMIL,
        "clam": CLAM,
        "transmil": TransMIL,
        "patchgcn": PatchGCN,
        "gadvmil": GAdvMIL,
    }
    if name not in table:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(table)}")
    return table[name](in_dim, **kwargs)
