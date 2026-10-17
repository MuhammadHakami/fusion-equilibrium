"""Doc 35 §41t: future-context inputs ("fut" block; inputs only, the whole shot's magnetics are given).

The shot's last frames carry the error (§41s: last 10 frames per shot = 4.5% of frames hold li 45%, Z_axis 22%, volume 20% of the
SSE; the last frame 35% of betaN SSE) while the inputs look at most +20 ms ahead (LAGS_MS). Per EFIT frame t, on a 1 ms grid with the
organizers' Ip time fix (data_fixes.fix_d3d_ip_times), s = sign of the shot's mean Ip, I = s * Ip [MA] (0 after the record ends):
  I(t + L) / max(I(t), 0.05) for L = 30 / 50 / 100 / 200 / 500 ms; min over [t, t + 50] and [t, t + 200] ms of I / max(I(t), 0.05);
  time until I < 0.1 MA and until I < 0.8 I(t) (capped at 2 s); mean dI/dt over [t, t + 100] and [t + 100, t + 300] ms;
  the 20 coil currents (ECOILA, bcoil, F1A .. F9B) at t + 50 / 100 / 200 ms minus at t;
  time to the shot's last EFIT frame and the elapsed fraction (EFIT times).
Writes runs/strat/fusion/F_fut_all.npy (cache row order) and F_fut_public.npy (public frame order).

    .venv/bin/python paper/strat/fusion/fut_feats.py --workers 8 [--n 3]
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

COILS = ["ECOILA", "bcoil"] + [f"F{i}{ab}" for i in range(1, 10) for ab in "AB"]
FL = (30, 50, 100, 200, 500)
CL = (50, 100, 200)
COLS = ["efit_times", "magnetics_time", "magnetics_plasma_current_times", "magnetics_plasma_current"] + [f"magnetics_{c}" for c in COILS]
NAMES = ([f"fu_ip{l}" for l in FL] + ["fu_min50", "fu_min200", "fu_t_end", "fu_t_80", "fu_dip100", "fu_dip300"]
         + [f"fu_{c}_{l}" for l in CL for c in COILS] + ["fu_to_last", "fu_frac"])


def feats(row) -> np.ndarray:
    te_ = np.asarray(row["efit_times"], np.float64); mt = np.asarray(row["magnetics_time"], np.float64)
    ipt = np.asarray(fix_d3d_ip_times(row), np.float64); ip = np.asarray(row["magnetics_plasma_current"], np.float64)
    m = np.isfinite(ip); ipt, ip = ipt[m], ip[m]
    s = 1.0 if np.mean(np.interp(te_, ipt, ip)) >= 0 else -1.0
    g = np.arange(te_[0] - 5.0, te_[-1] + 2001.0, 1.0)
    I = s * np.interp(g, ipt, ip, left=0.0, right=0.0) / 1000.0
    k = np.clip(np.round(te_ - g[0]).astype(int), 0, len(g) - 1); n = len(g)
    I0 = np.maximum(I[k], 0.05); out = [I[np.clip(k + l, 0, n - 1)] / I0 for l in FL]
    for L in (50, 200):
        out.append(np.array([I[a:a + L + 1].min() for a in k]) / I0)
    def until(cond_fn):
        r = np.empty(len(k))
        for i, a in enumerate(k):
            hit = np.nonzero(cond_fn(I[a:a + 2001], I[a]))[0]; r[i] = float(hit[0]) if len(hit) else 2000.0
        return r
    out += [until(lambda x, x0: x < 0.1), until(lambda x, x0: x < 0.8 * max(x0, 0.05))]
    out += [(I[np.clip(k + 100, 0, n - 1)] - I[k]) / 0.1, (I[np.clip(k + 300, 0, n - 1)] - I[np.clip(k + 100, 0, n - 1)]) / 0.2]
    for l in CL:
        for c in COILS:
            v = np.asarray(row[f"magnetics_{c}"], np.float64); mv = np.isfinite(v)
            if mv.sum() > 2:
                f = np.interp(g[np.clip(k + l, 0, n - 1)], mt[mv], v[mv]) - np.interp(g[k], mt[mv], v[mv])
            else:
                f = np.zeros(len(k))
            out.append(f)
    out += [te_[-1] - te_, (te_ - te_[0]) / max(te_[-1] - te_[0], 1e-9)]
    return np.stack(out, 1).astype(np.float32)


def _one(args):
    path, name = args
    row = pd.read_parquet(path / f"{name}.parquet", columns=COLS).iloc[0]
    return name, feats(row)


def build(src: pathlib.Path, names, counts, dst: pathlib.Path, workers: int) -> None:
    pos = dict(zip(names, np.r_[0, np.cumsum(counts)[:-1]])); H = np.zeros((int(np.sum(counts)), len(NAMES)), np.float32)
    with mp.Pool(workers) as pool:
        for i, (nm, f) in enumerate(pool.imap_unordered(_one, [(src, n) for n in names], chunksize=4)):
            s = pos[nm]; H[s:s + len(f)] = f
            if (i + 1) % 1000 == 0:
                print(f"[fut] {dst.name} {i + 1}/{len(names)}", flush=True)
    assert np.isfinite(H).all(), "non-finite future feature"
    np.save(dst, H); print(f"[fut] {dst.name} {H.shape}", flush=True)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--fold-only", action="store_true"); a = ap.parse_args()
    meta = torch.load(MF.CACHE / "meta.pt", weights_only=False); names = list(meta["names"]); shot = np.load(MF.CACHE / "shot.npy")
    counts = np.bincount(shot, minlength=len(names))
    if a.n:                                                                        # smoke: first n shots, checks frame counts
        for nm, c in zip(names[:a.n], counts[:a.n]):
            _, f = _one((MF.RAW, nm)); assert len(f) == c, (nm, len(f), c)
            print(nm, f.shape, "last frame", dict(zip(NAMES[:11], np.round(f[-1, :11], 3).tolist())))
        return
    if not a.fold_only:                                                           # Final day: test fold only
        (MF.OUT / "F_fut_names.txt").write_text("\n".join(NAMES))
        build(MF.RAW, names, counts, MF.OUT / "F_fut_all.npy", a.workers)
    proc = MF.ROOT / "data" / "sophelio_proc" / FD.FOLD; pn = [f.stem for f in sorted(proc.glob("*.npz"))]
    pc = np.asarray([len(np.load(proc / f"{nm}.npz")["t"]) for nm in pn])
    build(MF.RAW.parent / FD.FOLD, pn, pc, MF.OUT / f"F_fut_{FD.TAG}.npy", a.workers)
    print("FUT_DONE", flush=True)


if __name__ == "__main__":
    main()
