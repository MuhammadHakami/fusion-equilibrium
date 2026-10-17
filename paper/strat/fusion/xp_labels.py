"""Doc 35 §42t: X-point labels from the TRUE psi for every labelled frame (all splits; heads choose their rows by split).
Per frame: the scorer's LCFS on the true psi, its min-|grad psi| vertex, Newton-refined to the saddle on the cubic spline (xp_oracle.saddle).
A frame is diverted when the saddle is a true saddle (det H < 0), converged, and within 3 cm of the LCFS vertex; its (R, Z) goes to the
lower (Z < 0) or upper pair of columns, the other pair NaN. Limited frames are NaN in all four columns.
    .venv/bin/python paper/strat/fusion/xp_labels.py          -> runs/strat/fusion/xp_labels.npz (rows, y = [R_l, Z_l, R_u, Z_u])
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import fx
import gpu_pipe as GP
from d_anatomy import pred_xpoint
from xp_oracle import saddle


def main() -> None:
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import axis_shift as AX
    from gpu_post import GPUPost
    d = sota.Data(); R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    comps = d.meta["comps"].to(gp.dev, torch.float64); mean = d.meta["mean"].to(gp.dev, torch.float64); K = comps.shape[0]

    def label(pt):
        o = gp.g.derive(pt); x0 = pred_xpoint(gp, pt, o); xs, r, s = saddle(gp, pt, x0)
        ok = (s & (r < 0.2) & ((xs - x0).norm(dim=1) < 0.03) & torch.isfinite(xs).all(1)).cpu().numpy(); xs = xs.cpu().numpy()
        yy = np.full((len(xs), 4), np.nan, np.float32); lo = ok & (xs[:, 1] < 0); up = ok & (xs[:, 1] >= 0)
        yy[lo, 0:2] = xs[lo]; yy[up, 2:4] = xs[up]; return yy

    def pca_psi(rr):                     # TRAIN rows keep only the PCA coefficients (K = 256 components: D_LCFS floor 0.0020)
        return (torch.tensor(np.asarray(d.C[rr][:, :K], np.float64), device=gp.dev) @ comps + mean).reshape(-1, 65, 65)

    rows = np.arange(len(d.split)); y = np.full((len(rows), 4), np.nan, np.float32)
    for k, c in enumerate(GP.chunks(len(rows), 2048)):
        rr = rows[c]; held = d.split[rr] != 0; yy = np.full((len(rr), 4), np.nan, np.float32)
        if held.any():
            yy[held] = label(torch.tensor(d.psi_of(rr[held]).reshape(-1, 65, 65), device=gp.dev, dtype=torch.float64))
        if (~held).any():
            yy[~held] = label(pca_psi(rr[~held]))
        y[c] = yy
        if k % 50 == 0:
            print(f"[xpl] {c.stop}/{len(rows)}: diverted {np.isfinite(yy).any(1).mean():.3f} (lower {np.isfinite(yy[:, 0]).mean():.3f})", flush=True)
    vt = np.nonzero(d.split != 0)[0][::20]                         # label check: full psi vs its K-component reconstruction
    yp = np.concatenate([label(pca_psi(vt[c])) for c in GP.chunks(len(vt), 2048)]); yt = y[vt]
    e = np.concatenate([np.hypot(*(yp[:, 0:2] - yt[:, 0:2]).T), np.hypot(*(yp[:, 2:4] - yt[:, 2:4]).T)]); e = e[np.isfinite(e)]
    print(f"[xpl] PCA-{K} vs full psi labels ({len(vt)} VT frames): |dX| median {np.median(e) * 1e3:.2f} mm, p90 {np.quantile(e, 0.9) * 1e3:.2f} mm; "
          f"side agreement {np.mean(np.isfinite(yp[:, 0]) == np.isfinite(yt[:, 0])):.4f}", flush=True)
    np.savez(fx.OUT / "xp_labels.npz", rows=rows, y=y)
    for sp in (0, 1, 2):
        m = d.split == sp
        print(f"[xpl] split {sp}: {m.sum()} frames, lower {np.isfinite(y[m, 0]).mean():.3f}, upper {np.isfinite(y[m, 2]).mean():.3f}, "
              f"Z_l mean {np.nanmean(y[m, 1]):.3f} sd {np.nanstd(y[m, 1]):.3f}, R_l sd {np.nanstd(y[m, 0]):.3f}", flush=True)
    print("XPL_DONE", flush=True)


if __name__ == "__main__":
    main()
