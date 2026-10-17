"""Doc 35 §38ac: hybrid quantum-classical physics-informed KAN head (QPI-KAN) for the Fusion shape targets.

User (Oct 5): try NAS + Optuna, classical and quantum, and especially a quantum-classical PI-KAN (cf. QCPIKAN, arXiv
2606.20326: Chebyshev-polynomial KAN layers + parameterized quantum circuits + physics constraints in the loss).

Model: x (standardised inputs) -> [Chebyshev KAN layer -> LayerNorm] x depth (or residual MLP blocks: classical NAS choice)
-> linear read-out to the 7 consistency scalars + 64 LCFS rays, plus a linear skip from x; the input-dependent 30-atom
Rydberg register (paper/qlayer.py MultiRegister, 3 x 10, per-input detunings and participation gates) reads the hidden
state and adds alpha * W q (alpha zero-init, so the hybrid starts as its classical part).
Physics-informed loss (geometric self-consistency): the shape the 64 predicted rays describe about the predicted axis must
agree with the predicted kappa and volume (smooth extrema; Pappus volume of the ray polygon), after a fixed calibration
of the ray-polygon bias measured on the TRAIN labels. When the heads' rays and scalars disagree, the psi warps fight
(doc 35 §38z cross-talk); this term asks the head to produce mutually consistent targets.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn

import fx

sys.path.insert(0, str(fx.ROOT / "paper"))
N_ANG = 64


class ChebyKAN(nn.Module):
    """y_j = sum_i sum_k c_ijk T_k(tanh(x_i)) + b_j  (Chebyshev KAN layer, order K)."""

    def __init__(self, fin, fout, order=4):
        super().__init__()
        self.order = order
        self.coef = nn.Parameter(torch.randn(fin, fout, order + 1) / math.sqrt(fin * (order + 1)))
        self.bias = nn.Parameter(torch.zeros(fout))

    def forward(self, x):
        t = torch.tanh(x); T = [torch.ones_like(t), t]
        for _ in range(2, self.order + 1):
            T.append(2 * t * T[-1] - T[-2])
        B = torch.stack(T[: self.order + 1], -1)                                  # (N, fin, K+1)
        return torch.einsum("nik,iok->no", B, self.coef) + self.bias


class QPIKAN(nn.Module):
    def __init__(self, fin, out, width=1024, depth=2, order=4, block="kan", drop=0.0, q_layers=4, ansatz="rydberg",
                 link="independent", q_in=64, register=True, alpha_init=0.0):
        super().__init__()
        self.inp = ChebyKAN(fin, width, order) if block == "kan" else nn.Sequential(nn.Linear(fin, width), nn.GELU())
        mk = (lambda: ChebyKAN(width, width, order)) if block == "kan" else (lambda: nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)))
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), mk(), nn.Dropout(drop)) for _ in range(depth)])
        self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, out); self.skip = nn.Linear(fin, out)
        self.register = register
        if register:
            from qlayer import MultiRegister
            self.q_in = nn.Linear(width, q_in)                                     # controller input for the atoms
            self.reg = MultiRegister(q_in, 10, 3, layers=q_layers, ansatz=ansatz, link=link, gate=True)
            self.q_out = nn.Linear(30, out); self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        h = self.norm(h); y = self.out(h) + self.skip(x)
        if self.register:
            q = self.reg(torch.tanh(self.q_in(h)).float())
            y = y + self.alpha * self.q_out(q.to(y.dtype))
        return y


def ray_shape(axis, rays, beta=60.0):
    """Smooth kappa and volume of the polygon of `rays` (B, 64) about `axis` (B, 2) [R, Z] (beta: soft-extremum sharpness)."""
    th = 2 * math.pi * torch.arange(rays.shape[1], device=rays.device, dtype=rays.dtype) / rays.shape[1]
    R = axis[:, :1] + rays * torch.cos(th); Z = axis[:, 1:2] + rays * torch.sin(th)
    smax = lambda v: torch.logsumexp(beta * v, 1) / beta                          # noqa: E731
    kappa = (smax(Z) + smax(-Z)) / (smax(R) + smax(-R)).clamp(min=1e-3)
    Rn, Zn = torch.roll(R, -1, 1), torch.roll(Z, -1, 1); cross = R * Zn - Rn * Z
    vol = 2 * math.pi * ((R + Rn) * cross).sum(1).abs() / 6.0
    return kappa, vol
