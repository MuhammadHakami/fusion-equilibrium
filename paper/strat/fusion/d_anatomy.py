"""Doc 35 §42q diagnostic: where does D_LCFS (symmetric Hausdorff / rgeo = ONE worst point per frame) come from? On a stratified subset
of VALID + TEST shots (TRAIN-only components, deployed-style VB chain), per frame: D, the location of the Hausdorff maximum (d_at), its
distance to the TRUE X-point (minimum |grad psi_true| on the true LCFS contour), and the poloidal angle of d_at about the true axis.
If the worst point sits at the X-point in most frames, D is an X-point placement problem (lever: an X-point head + saddle warp).
    .venv/bin/python paper/strat/fusion/d_anatomy.py --n 60 [--psi psiF]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

import fx
import gpu_pipe as GP

VB = {"_steps": ["shift", "bw", "kw", "xp", "jgn", "li"], "xp": {"sigma": 0.15, "cap": 0.10}, "li": {"cap": 0.9}}


def true_xpoint(gp, psi_t, ref):
    """(B, 2) R, Z of the min-|grad psi| point on the true LCFS contour (bilinear on the grid gradient)."""
    g = gp.g; phi = g.sign * psi_t
    gz, gr = torch.gradient(phi, spacing=(g.Z, g.R), dim=(1, 2)); gm = torch.sqrt(gr ** 2 + gz ** 2)
    Q = ref.to(gp.dev, torch.float64); Qn = torch.nan_to_num(Q, nan=float(g.R[0]))
    c = ((Qn[..., 0] - g.R[0]) / g.dR).clamp(0, gm.shape[2] - 1.001); r = ((Qn[..., 1] - g.Z[0]) / g.dZ).clamp(0, gm.shape[1] - 1.001); r0, c0 = r.floor().long(), c.floor().long(); fr, fc = r - r0, c - c0
    b = torch.arange(len(psi_t), device=gp.dev)[:, None]
    v = (gm[b, r0, c0] * (1 - fr) * (1 - fc) + gm[b, r0 + 1, c0] * fr * (1 - fc) + gm[b, r0, c0 + 1] * (1 - fr) * fc + gm[b, r0 + 1, c0 + 1] * fr * fc)
    v = torch.where(torch.isfinite(Q).all(-1), v, torch.full_like(v, np.inf)); k = v.argmin(1)
    return Q[b[:, 0], k]


def pred_xpoint(gp, psi_p, o):
    """(B, 2) R, Z of the min-|grad psi| vertex of the predicted LCFS (derive's extraction)."""
    g = gp.g; phi = g.sign * psi_p.to(gp.dev, torch.float64)
    gz, gr = torch.gradient(phi, spacing=(g.Z, g.R), dim=(1, 2)); gm = torch.sqrt(gr ** 2 + gz ** 2)
    _, ro, co, seg, last = o["poly"]; valid = seg > 0
    r = ro.clamp(0, gm.shape[1] - 1.001); c = co.clamp(0, gm.shape[2] - 1.001); r0, c0 = r.floor().long(), c.floor().long(); fr, fc = r - r0, c - c0
    b = torch.arange(len(phi), device=gp.dev)[:, None]
    v = (gm[b, r0, c0] * (1 - fr) * (1 - fc) + gm[b, r0 + 1, c0] * fr * (1 - fc) + gm[b, r0, c0 + 1] * (1 - fr) * fc + gm[b, r0 + 1, c0 + 1] * fr * fc)
    v = torch.where(valid, v, torch.full_like(v, np.inf)); k = v.argmin(1); bb = b[:, 0]
    return torch.stack([g.R[0] + co[bb, k] * g.dR, g.Z[0] + ro[bb, k] * g.dZ], 1)


def saddle_warp(gp, psi, target, sigma=0.08, cap=0.03, n_it=3):
    """Move the predicted X-point (min-|grad psi| LCFS vertex) toward target (B, 2) with a Gaussian-localised displacement (resample)."""
    psi = psi.to(gp.dev, torch.float64); x0 = pred_xpoint(gp, psi, gp.g.derive(psi))
    g = torch.exp(-((gp.RR - x0[:, :1, None]) ** 2 + (gp.ZZ - x0[:, 1:, None]) ** 2) / (2 * sigma ** 2))
    delta = torch.zeros_like(x0); out = psi; capt = torch.tensor([cap, cap], device=gp.dev, dtype=psi.dtype)
    for _ in range(n_it):
        x1 = pred_xpoint(gp, out, gp.g.derive(out))
        delta = torch.maximum(torch.minimum(delta + (target - x1), capt), -capt)
        out = gp.sample(psi, gp.RR - delta[:, :1, None] * g, gp.ZZ - delta[:, 1:, None] * g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--psi", default="psiF"); a0 = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, ens_select as ES, axis_shift as AX
    from gpu_post import GPUPost
    d = sota.Data(); allv = [int(s) for s in np.unique(d.shot[d.split != 0])]
    rng = np.random.default_rng(0); shots = sorted(rng.choice(allv, a0.n, replace=False).tolist())
    J = "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7)); Q = "+".join(f"head_qb_QH_s{s}" for s in range(1, 7))
    a = argparse.Namespace(psi=a0.psi, heads=J, kw="", qb="qh50", axis="head", qbheads=Q, qbw=0.5, shrink="", guard="")
    _pa = GP.post_all; GP.post_all = lambda *x, **k: _pa(*x, **{**k, "size": 1024})      # 1/4 the peak memory of the build chunking
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); rows_all = []
    def sink(tag, gshots, psi_g, q, b):
        Cref, rgeo, _, has = GP.refs(d, gshots); allr = np.concatenate([fx.scored_rows(d, s) for s in gshots])
        for c in GP.chunks(len(psi_g), 2048):
            o = gp.g.derive(psi_g[c], torch.tensor(Cref[c]), torch.tensor(rgeo[c]))
            pt = torch.tensor(d.psi_of(allr[c]).reshape(-1, 65, 65), device=gp.dev, dtype=torch.float64)
            ot = gp.g.derive(pt); xp = true_xpoint(gp, pt, torch.tensor(Cref[c]))
            dat = o["d_at"]; dist = torch.linalg.norm(dat - xp, dim=1); xq = pred_xpoint(gp, psi_g[c], o); dx = xq - xp
            ow = gp.g.derive(saddle_warp(gp, psi_g[c], xp), torch.tensor(Cref[c]), torch.tensor(rgeo[c]))        # ORACLE: true X-point
            ang = torch.rad2deg(torch.atan2(dat[:, 1] - ot["Z_axis"], dat[:, 0] - ot["R_axis"]))
            angx = torch.rad2deg(torch.atan2(xp[:, 1] - ot["Z_axis"], xp[:, 0] - ot["R_axis"]))
            for j in range(c.stop - c.start):
                if has[c.start + j]:
                    rows_all.append((float(o["d"][j]), float(dist[j]), float(ang[j]), float(angx[j]), float(xp[j, 1]), float(dx[j, 0]), float(dx[j, 1]), float(ow["d"][j])))
    GB.predict(d, shots, a, sota, ES, arms=[("VB", json.dumps(VB))], sink=sink)
    A = np.array(rows_all); D, dist, ang, angx, zx, dxr, dxz, Dor = A.T; tot = D.sum()
    print(f"[danat] ORACLE saddle warp to the true X-point (cap 3 cm, sigma 8 cm): mean D {D.mean():.5f} -> {np.nanmean(Dor):.5f}; frames improved {(Dor < D).mean():.3f}", flush=True)
    print(f"[danat] {len(A)} frames, {len(shots)} shots: mean D {D.mean():.5f}, median {np.median(D):.5f}", flush=True)
    for thr in (0.02, 0.05, 0.10, 0.20):
        m = dist < thr
        print(f"[danat] worst point within {thr:.2f} m of the true X-point: {m.mean():.3f} of frames, {D[m].sum() / tot:.3f} of the D sum, "
              f"mean D there {D[m].mean() if m.any() else 0:.5f} vs elsewhere {D[~m].mean():.5f}", flush=True)
    dang = (ang - angx + 180) % 360 - 180
    for lo, hi in ((-20, 20), (-45, 45), (-90, 90)):
        m = (dang >= lo) & (dang < hi)
        print(f"[danat] worst point within {hi} deg of the X-point direction: {m.mean():.3f} of frames, {D[m].sum() / tot:.3f} of the D sum", flush=True)
    print(f"[danat] X-point below the axis (LSN-like): {(zx < 0).mean():.3f} of frames", flush=True)
    ex = np.hypot(dxr, dxz)
    print(f"[danat] predicted vs true X-point: |dR| median {np.median(np.abs(dxr)) * 1e3:.1f} mm, |dZ| median {np.median(np.abs(dxz)) * 1e3:.1f} mm, "
          f"|d| p90 {np.quantile(ex, 0.9) * 1e3:.1f} mm; corr(D, |d|) {np.corrcoef(D, ex)[0, 1]:.3f}, corr(D, |dR|) {np.corrcoef(D, np.abs(dxr))[0, 1]:.3f}, "
          f"corr(D, |dZ|) {np.corrcoef(D, np.abs(dxz))[0, 1]:.3f}", flush=True)
    for q in (0.5, 0.8):
        m = ex <= np.quantile(ex, q)
        print(f"[danat] frames with X-point error <= p{int(q * 100)} ({np.quantile(ex, q) * 1e3:.1f} mm): mean D {D[m].mean():.5f} vs the rest {D[~m].mean():.5f}", flush=True)
    (fx.OUT / "d_anatomy.json").write_text(json.dumps({"n_frames": len(A), "frames": A.tolist()[:20000]}))


if __name__ == "__main__":
    main()
