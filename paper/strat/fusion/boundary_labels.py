"""Doc 35 §37p step 1: scorer-derived shape labels for TRAIN frames (CPU; TRAIN psi from its 256-component PCA, the
stored form; S ceiling of that projection 0.9996) and for the 180-shot VALID sample (true psi).

Per frame, with the vendored scorer's own extraction (axis_sign -1, DIII-D): the 7 consistency scalars (R_axis, Z_axis,
kappa, tri_top, tri_bot, volume, li) and the LCFS radius r(theta) from the magnetic axis at N_ANG equally spaced angles
(ray / polygon intersection). NaN where extraction fails.

    [STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v2] .venv/bin/python paper/strat/fusion/boundary_labels.py <train_stride>
        ->  runs/strat/fusion/boundary_labels{SUFFIX}.npz
"""
from __future__ import annotations

import sys

import numpy as np

import fx  # noqa: I001  (CPU only)
import strategies as ST

import os
N_ANG = int(os.environ.get("STRAT_NANG", "32"))
SUFFIX = os.environ.get("STRAT_LABEL_SUFFIX", "")
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def ray_radii(C, r0, z0, n=N_ANG):
    """Distance from (r0, z0) to the closed polygon C (N, 2) along n rays (largest hit per ray)."""
    th = 2 * np.pi * np.arange(n) / n; dr, dz = np.cos(th), np.sin(th)
    P = C - np.array([r0, z0]); Q = np.roll(P, -1, 0) - P                     # segments P_j -> P_j + Q_j
    den = dr[:, None] * Q[None, :, 1] - dz[:, None] * Q[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (P[None, :, 0] * Q[None, :, 1] - P[None, :, 1] * Q[None, :, 0]) / den          # ray parameter
        u = (P[None, :, 0] * dz[:, None] - P[None, :, 1] * dr[:, None]) / den              # segment parameter
    ok = (u >= 0) & (u <= 1) & (t > 0) & np.isfinite(t)
    return np.where(ok, t, -np.inf).max(1)


def _one(args):
    psis, = args
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    import local_score as LS
    m = np.load(fx.STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = m["grid_R"], m["grid_Z"]; mc = m["mask_coarse"].astype(bool); mf = mc.astype(float)
    out = np.full((len(psis), len(NAMES) + N_ANG), np.nan, np.float32)
    for i, p in enumerate(psis):
        p = p.reshape(65, 65).astype(np.float64)
        C, _ = extract_lcfs_with_sign(p, R, Z, LS.MACHINE, mc, mf, n_iter=LS.N_ITER, axis_sign=-1.0)
        v = derive_frame(p, R, Z, LS.MACHINE, mc, mf, contour=C, with_li=True, axis_sign=-1.0)
        out[i, :len(NAMES)] = [v[k] for k in NAMES]
        if C is not None and len(C) >= 8 and np.isfinite(v["R_axis"]):
            rr = ray_radii(np.asarray(C, float), v["R_axis"], v["Z_axis"])
            out[i, len(NAMES):] = np.where(np.isfinite(rr), rr, np.nan)
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sota, d = fx.data()
    comps = d.meta["comps"].numpy().astype(np.float32); mean = d.meta["mean"].numpy().astype(np.float32)
    tr = np.nonzero(d.split == 0)[0][::stride]
    shots, _ = ST.sample(); va = np.concatenate([fx.valid_rows(d, s) for s in shots])
    def chunk(i):
        if i < len(tr):
            r = tr[i:i + 2000]; return (np.asarray(d.C[r], np.float32) @ comps + mean,)
        j = i - len(tr); r = va[j:j + 2000]
        return (d.psi_of(r).reshape(len(r), -1).astype(np.float32),)
    starts = list(range(0, len(tr), 2000)) + [len(tr) + j for j in range(0, len(va), 2000)]
    res = []
    with ProcessPoolExecutor(max_workers=fx.N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        for b in range(0, len(starts), 2 * fx.N_WORKERS):                       # bounded memory: ~2 chunks per worker
            res += list(ex.map(_one, [chunk(i) for i in starts[b:b + 2 * fx.N_WORKERS]]))
            print(f"[s37p-labels] {min(b + 2 * fx.N_WORKERS, len(starts))}/{len(starts)} chunks", flush=True)
    Y = np.concatenate(res); ntr = len(tr)
    np.savez(fx.OUT / f"boundary_labels{SUFFIX}.npz", rows_train=tr, y_train=Y[:ntr], rows_valid=va, y_valid=Y[ntr:], names=np.array(NAMES), n_ang=N_ANG)
    print(f"S37P_LABELS train {ntr} valid {len(va)} nan-frac train {np.isnan(Y[:ntr, 2]).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
