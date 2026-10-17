"""Doc 35 §42s: plasma-shape features for the q95 / betaN heads (physics stacking).

q95 ~ a^2 B0 / (R Ip) * (1 + kappa^2 (1 + 2 delta^2)) / 2 and B0 ~ I_TF / R, so log q95 ~ log(bcoil / Ip) + [2 log a - 2 log R_geo +
log((1 + kappa^2 (1 + 2 delta^2)) / 2)]; the q/b heads already see log(bcoil / Ip) (qphys), the bracket needs the plasma shape. Shape
comes from shape heads without label leakage: TRAIN rows take the 5-fold out-of-fold J1 predictions (§38u, J1F0-4), VALID / TEST rows the
TRAIN-only J1 heads (s1-6 mean); public / private folds (production) the all-data J1AH heads. 10 columns: kappa, tri_top, tri_bot,
log volume, li, R_axis, Z_axis, a, R_geo, q-shape term.
    .venv/bin/python paper/strat/fusion/qshape_feats.py [--fold public]     -> runs/strat/fusion/qshape_{all|<fold>}.npy
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import fx

NF = 10


def feats(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, np.float64); ra, za, ka, tt, tb, vol, li = P[:, :7].T; r = P[:, 7:71]
    th = 2 * np.pi * np.arange(r.shape[1]) / r.shape[1]; Rb = ra[:, None] + r * np.cos(th)[None]
    a = np.clip((Rb.max(1) - Rb.min(1)) / 2, 0.05, None); rg = np.clip((Rb.max(1) + Rb.min(1)) / 2, 0.5, None)
    dl = 0.5 * (tt + tb); sf = np.log(np.clip((1 + ka ** 2 * (1 + 2 * dl ** 2)) / 2, 1e-3, None))
    F = np.stack([ka, tt, tb, np.log(np.clip(vol, 1e-3, None)), li, ra, za, a, rg, 2 * np.log(a) - 2 * np.log(rg) + sf], 1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", default=""); a0 = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    if a0.fold:                                                    # production: all-data J1AH heads on a test fold
        import submit_public as SP
        d = sota.Data(); pub = SP.PublicData(d.meta); rows = np.arange(len(pub.X))
        H = "+".join(f"head_boundary_v8_J1AH_s{s}" for s in range(1, 9))
        out = feats(GB.heads_gpu(pub, rows, H)); np.save(fx.OUT / f"qshape_{a0.fold}.npy", out)
        print(f"[qshape] {a0.fold}: {out.shape}", flush=True); return
    d = sota.Data(); out = np.zeros((len(d.split), NF), np.float32)
    for k in range(5):                                             # TRAIN rows: out-of-fold J1 (§38u)
        z = np.load(fx.OUT / f"head_boundary_v8_J1F{k}_s1_oof.npz"); out[z["rows_oof"]] = feats(z["pred_oof"])
    vt = np.nonzero(d.split != 0)[0]                               # VALID / TEST rows: TRAIN-only J1 heads
    out[vt] = feats(GB.heads_gpu(d, vt, "+".join(f"head_boundary_v8_J1_s{s}" for s in range(1, 7))))
    np.save(fx.OUT / "qshape_all.npy", out)
    print(f"[qshape] all: {out.shape}, nonzero rows {(np.abs(out).sum(1) > 0).mean():.4f}, means {np.round(out.mean(0), 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
