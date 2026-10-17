"""Doc 35 §42t diagnostic: D_LCFS ceiling of a CONTINUOUS X-point correction. §42q's oracle aimed at the min-|grad psi| vertex of the
true LCFS (quantised to grid edges) with a 3 cm cap, while the large-D frames are off by one grid cell (dZ 5 cm / dR 2.7 cm). Here both
saddles are Newton-refined on the cubic spline (grad psi = 0, finite-difference Hessian), the warp cap is 8 cm, and the correction is
gated on the predicted-vs-true saddle distance. Same deployed-style VB chain and TRAIN-only components as §42q (d_anatomy.py).
    .venv/bin/python paper/strat/fusion/xp_oracle.py --n 60
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

import fx
import gpu_pipe as GP
from d_anatomy import VB, pred_xpoint, true_xpoint

GATES = (0.0, 0.005, 0.01, 0.02, 0.03)


def saddle(gp, psi, x0, n_it=10, h=0.003, step_cap=0.02):
    """(B, 2) Newton-refined critical point of the cubic-spline psi near x0, plus |grad| / scale and det(H) < 0 flag."""
    c = gp.sp.coeffs(psi.to(gp.dev, torch.float64)); x = x0.clone().to(gp.dev, torch.float64)
    dR = torch.tensor([0, h, -h, 0, 0, h, h, -h, -h], device=gp.dev, dtype=torch.float64)
    dZ = torch.tensor([0, 0, 0, h, -h, h, -h, h, -h], device=gp.dev, dtype=torch.float64)
    for _ in range(n_it + 1):
        Rq = x[:, :1] + dR[None]; Zq = x[:, 1:] + dZ[None]
        f = gp.sp.sample(c, (Zq - gp.Z0) / gp.dZ, (Rq - gp.R0) / gp.dR)
        gr = (f[:, 1] - f[:, 2]) / (2 * h); gz = (f[:, 3] - f[:, 4]) / (2 * h)
        hrr = (f[:, 1] - 2 * f[:, 0] + f[:, 2]) / h ** 2; hzz = (f[:, 3] - 2 * f[:, 0] + f[:, 4]) / h ** 2
        hrz = (f[:, 5] - f[:, 6] - f[:, 7] + f[:, 8]) / (4 * h ** 2); det = hrr * hzz - hrz ** 2
        sR = -(hzz * gr - hrz * gz) / det; sZ = -(-hrz * gr + hrr * gz) / det
        s = torch.stack([sR, sZ], 1); s = torch.nan_to_num(s); n = s.norm(dim=1, keepdim=True)
        s = s * torch.clamp(step_cap / n.clamp(min=1e-12), max=1.0)
        if _ < n_it:
            x = x + s
    scale = (hrr.abs() + hzz.abs()) * 0.01 + 1e-12                      # |grad| that a 1 cm offset would produce
    return x, (torch.hypot(gr, gz) / scale), det < 0


def warp_to(gp, psi, x0, target, sigma, cap, n_it=4):
    """Move the saddle x0 -> target with a Gaussian-localised displacement (resample); x re-found by Newton each iteration."""
    psi = psi.to(gp.dev, torch.float64)
    g = torch.exp(-((gp.RR - x0[:, :1, None]) ** 2 + (gp.ZZ - x0[:, 1:, None]) ** 2) / (2 * sigma ** 2))
    delta = torch.zeros_like(x0); out = psi; x1 = x0; capt = torch.tensor([cap, cap], device=gp.dev, dtype=psi.dtype)
    for _ in range(n_it):
        delta = torch.maximum(torch.minimum(delta + (target - x1), capt), -capt)
        out = gp.sample(psi, gp.RR - delta[:, :1, None] * g, gp.ZZ - delta[:, 1:, None] * g)
        x1, _, _ = saddle(gp, out, x1 + (target - x1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--psi", default="psiF")
    ap.add_argument("--sigmas", default="0.08,0.15"); a0 = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, ens_select as ES, axis_shift as AX
    from gpu_post import GPUPost
    d = sota.Data(); allv = [int(s) for s in np.unique(d.shot[d.split != 0])]
    rng = np.random.default_rng(0); shots = sorted(rng.choice(allv, a0.n, replace=False).tolist())
    J = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7)); Q = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))
    a = argparse.Namespace(psi=a0.psi, heads=J, kw="", qb="qh50", axis="head", qbheads=Q, qbw=0.5, shrink="", guard="")
    _pa = GP.post_all; GP.post_all = lambda *x, **k: _pa(*x, **{**k, "size": 1024})
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); sig = [float(s) for s in a0.sigmas.split(",")]; rows = []
    def sink(tag, gshots, psi_g, q, b):
        Cref, rgeo, _, has = GP.refs(d, gshots); allr = np.concatenate([fx.scored_rows(d, s) for s in gshots])
        for c in GP.chunks(len(psi_g), 1024):
            ct, rg = torch.tensor(Cref[c]), torch.tensor(rgeo[c]); pp = psi_g[c].to(gp.dev, torch.float64)
            o = gp.g.derive(pp, ct, rg)
            pt = torch.tensor(d.psi_of(allr[c]).reshape(-1, 65, 65), device=gp.dev, dtype=torch.float64)
            xt, rt, st = saddle(gp, pt, true_xpoint(gp, pt, ct)); xp, rp, sp_ = saddle(gp, pp, pred_xpoint(gp, pp, o))
            ok = st & sp_ & (rt < 0.2) & (rp < 0.2) & ((xt - xp).norm(dim=1) < 0.15)
            dist = (xt - xp).norm(dim=1); Dw = []
            for s in sig:
                ow = gp.g.derive(warp_to(gp, pp, xp, xt, s, 0.08), ct, rg); Dw.append(torch.where(ok, ow["d"], o["d"]))
            for j in range(c.stop - c.start):
                if has[c.start + j]:
                    rows.append([float(o["d"][j]), float(dist[j]), float(ok[j])] + [float(x[j]) for x in Dw]
                                + [float(xt[j, 0]), float(xt[j, 1]), float(xp[j, 0]), float(xp[j, 1])])
    GB.predict(d, shots, a, sota, ES, arms=[("VB", json.dumps(VB))], sink=sink)
    A = np.array(rows); D, dist, ok = A[:, 0], A[:, 1], A[:, 2] > 0
    print(f"[xpo] {len(A)} frames, {len(shots)} shots: mean D {D.mean():.5f}; saddles found on both sides {ok.mean():.3f}; "
          f"continuous |dX| median {np.median(dist[ok]) * 1e3:.1f} mm, p75 {np.quantile(dist[ok], 0.75) * 1e3:.1f}, p90 {np.quantile(dist[ok], 0.9) * 1e3:.1f}", flush=True)
    for i, s in enumerate(sig):
        Dw = A[:, 3 + i]
        for gt in GATES:
            m = ok & (dist > gt); Dg = np.where(m, Dw, D)
            print(f"[xpo] sigma {s:.2f} gate {gt * 1e3:4.0f} mm: frames moved {m.mean():.3f}, mean D {D.mean():.5f} -> {Dg.mean():.5f} "
                  f"(moved frames {D[m].mean() if m.any() else 0:.4f} -> {Dw[m].mean() if m.any() else 0:.4f})", flush=True)
    (fx.OUT / "xp_oracle.json").write_text(json.dumps({"cols": ["D", "dist", "ok"] + [f"Dw_{s}" for s in sig] + ["xt_R", "xt_Z", "xp_R", "xp_Z"],
                                                      "rows": A.tolist()}))


if __name__ == "__main__":
    main()


def xh_step(gp, psi, xh, gate=0.01, sigma=0.08, cap=0.08, n_it=4, far=0.15, w=1.0, axes="RZ"):
    """§42t chain step: per frame, the predicted active X-point (LCFS min-|grad| vertex -> Newton saddle) moves to the X-point head of
    its side (xh = [R_l, Z_l, R_u, Z_u]) when they differ by more than gate (and less than far); limited frames are left alone."""
    psi64 = psi.to(gp.dev, torch.float64); o = gp.g.derive(psi64); x0 = pred_xpoint(gp, psi64, o); xs, r, s = saddle(gp, psi64, x0)
    lower = (xs[:, 1] < 0)[:, None]; tgt = torch.where(lower, xh[:, 0:2], xh[:, 2:4]).to(psi64.dtype)
    tgt = w * tgt + (1 - w) * xs if w != 1.0 else tgt                   # blend with the psi saddle (w = 1: the head alone)
    if axes == "R":                                                       # R only: tri follows the X-point column, kappa its row (kw)
        tgt = torch.stack([tgt[:, 0], xs[:, 1]], 1)
    elif axes == "Z":
        tgt = torch.stack([xs[:, 0], tgt[:, 1]], 1)
    dist = (tgt - xs).norm(dim=1)
    ok = s & (r < 0.2) & ((xs - x0).norm(dim=1) < 0.03) & torch.isfinite(tgt).all(1) & (dist > gate) & (dist < far)
    if not bool(ok.any()):
        return psi
    out = warp_to(gp, psi64, xs, torch.where(ok[:, None], tgt, xs), sigma, cap, n_it)
    gp.xh_diag = (int(ok.sum()), len(ok))
    return torch.where(ok[:, None, None], out, psi64).to(psi.dtype)
