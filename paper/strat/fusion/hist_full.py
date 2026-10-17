"""Fusion §41q: full-rate, sign-invariant current-history features (inputs only). The §41c history (hist.npy) was built from
the cached RAW frames: EFIT times only (the limited ramp-up, where the current profile forms, is absent) and the broken raw Ip
time base on ~70% of shots. Here every feature comes from the full-rate signals from breakdown on, with the organizers' Ip
time fix (data_fixes.fix_d3d_ip_times), on a 1 ms grid; s = sign of the shot's mean Ip makes them orientation-invariant.
Per EFIT frame t (tb = first time s*Ip > 100 kA):
  t - tb; int_tb^t s*Ip dt; s*Ip at t - 0.25 / 0.5 / 1 / 2 s; max_{<=t} s*Ip; time since s*Ip first reached 90% of s*Ip(t);
  EWMA of s*dIp/dt with tau 0.1 / 0.3 / 1 / 3 s; EWMA of s*Ip with the same taus minus s*Ip(t); s*(ECOILA(t) - ECOILA(tb));
  s*ECOILA at t - 0.5 / 1 s minus now; core Thomson Te mean over [t - 0.5, t] and [t - 1, t]; resistive clock
  int_tb^t Te_core^-1.5 dt (Te in keV, floored at 0.05) and the same over the last 1 s; mean ramp rate over the first 0.5 s.
Writes runs/strat/fusion/H_full_all.npy (cache row order) and H_full_public.npy (public frame order) + H_full_names.txt.

    .venv/bin/python paper/strat/fusion/hist_full.py --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import fold as FD  # noqa: E402,I001
sys.path.insert(0, str(HERE))
import mag_fix as MF  # noqa: E402

sys.path.insert(0, str(MF.ROOT / "baseline" / "sophelio-starter"))
from data_fixes import fix_d3d_ip_times  # noqa: E402

TAUS = (100.0, 300.0, 1000.0, 3000.0)
LAGS = (250.0, 500.0, 1000.0, 2000.0)
COLS = ["efit_times", "magnetics_time", "magnetics_plasma_current_times", "magnetics_plasma_current", "magnetics_ECOILA",
        "thomson_core_times", "thomson_core_Te"]
NAMES = (["hf_since_bd", "hf_int_ip"] + [f"hf_ip_lag{int(l)}" for l in LAGS] + ["hf_ip_max", "hf_t90"]
         + [f"hf_dip_ewma{int(t)}" for t in TAUS] + [f"hf_ip_ewma{int(t)}" for t in TAUS]
         + ["hf_dE_bd", "hf_E_lag500", "hf_E_lag1000", "hf_te05", "hf_te1", "hf_rclock", "hf_rclock1", "hf_ramp0"])


def ewma(x, tau, dt=1.0):
    a = dt / tau; out = np.empty_like(x); acc = 0.0
    for i, v in enumerate(x):                                   # 1 ms grid, a few 1e4 points per shot
        acc += a * (v - acc); out[i] = acc
    return out


def feats(row) -> np.ndarray:
    te_ = np.asarray(row["efit_times"], np.float64); mt = np.asarray(row["magnetics_time"], np.float64)
    ipt = np.asarray(fix_d3d_ip_times(row), np.float64); ip = np.asarray(row["magnetics_plasma_current"], np.float64)
    m = np.isfinite(ip); ipt, ip = ipt[m], ip[m]
    s = 1.0 if np.mean(np.interp(te_, ipt, ip)) >= 0 else -1.0
    g = np.arange(min(te_[0] - 3000.0, ipt[0]), te_[-1] + 1.0, 1.0)               # 1 ms grid
    I = s * np.interp(g, ipt, ip, left=0.0, right=0.0) / 1000.0                    # MA, canonical sign
    e = np.asarray(row["magnetics_ECOILA"], np.float64); me = np.isfinite(e)
    E = s * np.interp(g, mt[me], e[me]) if me.sum() > 2 else np.zeros_like(g)
    above = np.nonzero(I > 0.1)[0]; ib = int(above[0]) if len(above) else 0; tb = g[ib]
    Ib = np.where(np.arange(len(g)) >= ib, I, 0.0); cint = np.cumsum(Ib) * 1e-3     # MA s
    dI = np.gradient(Ib) * 1e3                                                     # MA/s
    cmax = np.maximum.accumulate(Ib)
    tt = np.asarray(row["thomson_core_times"], np.float64) if row["thomson_core_times"] is not None else np.zeros(0)
    if len(tt) > 2:
        T = np.stack([np.asarray(x, np.float64) for x in row["thomson_core_Te"]]); T = np.where(np.isfinite(T) & (T > 0), T, np.nan)
        with np.errstate(all="ignore"):
            tm = np.nanmean(T, 1) / 1000.0
        ok = np.isfinite(tm); tegrid = np.interp(g, tt[ok], tm[ok]) if ok.sum() > 2 else np.full_like(g, 0.5)
    else:
        tegrid = np.full_like(g, 0.5)
    tegrid = np.where(np.arange(len(g)) >= ib, np.maximum(tegrid, 0.05), 0.05)
    rc = np.cumsum(np.where(np.arange(len(g)) >= ib, tegrid ** -1.5, 0.0)) * 1e-3
    cte = np.cumsum(tegrid) * 1e-3
    k = np.clip(np.round(te_ - g[0]).astype(int), 0, len(g) - 1)
    def at(arr, lag):
        return arr[np.clip(k - int(lag), 0, len(g) - 1)]
    out = [te_ - tb, cint[k]] + [at(Ib, l) for l in LAGS] + [cmax[k]]
    # time since s*Ip first reached 90% of its current value (diffusion progress at this current level)
    first = np.array([np.argmax(Ib[: kk + 1] >= 0.9 * Ib[kk]) if Ib[kk] > 0 else kk for kk in k])
    out += [g[k] - g[first]]
    out += [ewma(dI, t)[k] for t in TAUS] + [ewma(Ib, t)[k] - Ib[k] for t in TAUS]
    out += [E[k] - E[ib], at(E, 500) - E[k], at(E, 1000) - E[k]]
    out += [(cte[k] - at(cte, 500)) / 0.5, (cte[k] - at(cte, 1000)) / 1.0, rc[k], rc[k] - at(rc, 1000)]
    out += [np.full(len(k), (I[min(ib + 500, len(I) - 1)] - I[ib]) / 0.5)]
    return np.stack(out, 1).astype(np.float32)


def _one(args):
    path, name = args
    row = pd.read_parquet(path / f"{name}.parquet", columns=COLS).iloc[0]
    return name, feats(row)


def build(src: pathlib.Path, names, counts, dst: pathlib.Path, workers: int) -> None:
    pos = dict(zip(names, np.r_[0, np.cumsum(counts)[:-1]])); H = np.zeros((int(np.sum(counts)), len(NAMES)), np.float32)
    with mp.Pool(workers) as pool:
        for i, (nm, f) in enumerate(pool.imap_unordered(_one, [(src, n) for n in names], chunksize=4)):
            c = counts[names.index(nm)] if len(names) < 2000 else None
            s = pos[nm]; H[s:s + len(f)] = f
            if (i + 1) % 1000 == 0:
                print(f"[hfull] {dst.name} {i + 1}/{len(names)}", flush=True)
    assert np.isfinite(H).all(), "non-finite history feature"
    np.save(dst, H); print(f"[hfull] {dst.name} {H.shape}", flush=True)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--fold-only", action="store_true"); a = ap.parse_args()
    meta = torch.load(MF.CACHE / "meta.pt", weights_only=False); names = list(meta["names"]); shot = np.load(MF.CACHE / "shot.npy")
    counts = np.bincount(shot, minlength=len(names))
    if a.n:                                                                        # smoke: first n shots, checks frame counts
        for nm, c in zip(names[:a.n], counts[:a.n]):
            _, f = _one((MF.RAW, nm)); assert len(f) == c, (nm, len(f), c); print(nm, f.shape, np.round(f[len(f) // 2], 3).tolist())
        return
    if not a.fold_only:                                                           # Final day: test fold only
        (MF.OUT / "H_full_names.txt").write_text("\n".join(NAMES))
        build(MF.RAW, names, counts, MF.OUT / "H_full_all.npy", a.workers)
    proc = MF.ROOT / "data" / "sophelio_proc" / FD.FOLD; pn = [f.stem for f in sorted(proc.glob("*.npz"))]
    pc = np.asarray([len(np.load(proc / f"{nm}.npz")["t"]) for nm in pn])
    build(MF.RAW.parent / FD.FOLD, pn, pc, MF.OUT / f"H_full_{FD.TAG}.npy", a.workers)
    print("HFULL_DONE", flush=True)


if __name__ == "__main__":
    main()
