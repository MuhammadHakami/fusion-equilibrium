"""Doc 35 §41f: Thomson profiles on fixed physical coordinates (inputs only).

extract.py resamples each Thomson profile by channel index after dropping invalid channels, but the core chord layout varies
by shot (top channel Z 0.68-0.83 m; a given index moves by 3-5 cm, the pedestal width) and dropped channels erase where the
plasma ends. Here, per EFIT frame (nearest Thomson frame in time, as extract.py):
- core (vertical chord, R = 1.94 m): Te and ne interpolated in Z onto 28 fixed points, Z = 0.02 .. 0.83 m (0 outside the
  shot's coverage), plus a coverage mask;
- tangential (midplane, R 1.67-2.06 m): Te and ne onto 8 fixed R points;
- edge markers on the core chord: highest Z with Te above 100 / 300 / 1000 eV, and the Te / ne gradients over the top 10 cm.
Rows align with the trunk cache (data/sophelio_proc/cache_k256/X.npy); shots not processed stay 0.
    .venv/bin/python paper/strat/fusion/thz_feats.py [--cv]     (--cv: only the 900 cv_trunk.py shots)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import fx  # noqa: I001
import fold as FD  # noqa: E402,I001
sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio"))
import sota  # noqa: E402

ZC = np.linspace(0.02, 0.83, 28); RT = np.linspace(1.67, 2.06, 8); SRC = fx.ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data"


def nearest(tt, tq):
    i = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
    return np.where(np.abs(tt[i - 1] - tq) < np.abs(tt[i] - tq), i - 1, i)


def profile(vals, pos, grid):
    """Interpolate valid (finite, > 0) channel values in physical position onto the grid; 0 outside the valid coverage."""
    out = np.zeros((len(vals), len(grid))); o = np.argsort(pos); p = pos[o]
    for k, v in enumerate(vals[:, o]):
        g = np.isfinite(v) & (v > 0)
        if g.sum() >= 2:
            out[k] = np.interp(grid, p[g], v[g], left=0.0, right=0.0)
    return out


def shot_feats(path, tq):
    r = pd.read_parquet(path).iloc[0]; F = np.zeros((len(tq), 2 * len(ZC) + len(ZC) + 2 * len(RT) + 5), np.float32)
    nm = np.asarray(r["thomson_chord_name"]).astype(str); z = np.asarray(r["thomson_chord_Z"], float); R = np.asarray(r["thomson_chord_R"], float)
    core, tang = np.char.startswith(nm, "TS_core"), np.char.startswith(nm, "TS_tangential")
    if r["thomson_core_times"] is not None and len(r["thomson_core_times"]) > 1:
        i = nearest(np.asarray(r["thomson_core_times"], float), tq)
        Te = np.stack([np.asarray(x, float) for x in r["thomson_core_Te"]])[i]; ne = np.stack([np.asarray(x, float) for x in r["thomson_core_ne"]])[i]
        zc = z[core][: Te.shape[1]]; pt, pn = profile(Te, zc, ZC), profile(ne / 1e19, zc, ZC)
        F[:, :len(ZC)] = pt; F[:, len(ZC):2 * len(ZC)] = pn; F[:, 2 * len(ZC):3 * len(ZC)] = (ZC >= zc.min() - 1e-3) & (ZC <= zc.max() + 1e-3)
        base = 3 * len(ZC) + 2 * len(RT)
        for j, thr in enumerate((100.0, 300.0, 1000.0)):
            F[:, base + j] = np.array([ZC[np.nonzero(row > thr)[0].max()] if (row > thr).any() else 0.0 for row in pt])
        top = ZC > ZC[-1] - 0.10
        F[:, base + 3] = (pt[:, top][:, -1] - pt[:, top][:, 0]) / 1e3; F[:, base + 4] = pn[:, top][:, -1] - pn[:, top][:, 0]
    if r["thomson_edge_times"] is not None and len(r["thomson_edge_times"]) > 1 and tang.any():
        i = nearest(np.asarray(r["thomson_edge_times"], float), tq)
        Te = np.stack([np.asarray(x, float) for x in r["thomson_edge_Te"]])[i]; ne = np.stack([np.asarray(x, float) for x in r["thomson_edge_ne"]])[i]
        rt = R[tang][: Te.shape[1]]; b = 3 * len(ZC)
        F[:, b:b + len(RT)] = profile(Te, rt, RT); F[:, b + len(RT):b + 2 * len(RT)] = profile(ne / 1e19, rt, RT)
    return F


NF = 2 * len(ZC) + len(ZC) + 2 * len(RT) + 5


def _work(args):
    """One worker: a chunk of (row start, row end, parquet path, frame times) written into the shared memmap."""
    out, jobs = args; F = np.load(out, mmap_mode="r+")
    for a, b, path, tq in jobs:
        F[a:b] = shot_feats(path, tq)
    F.flush(); return len(jobs)


def public() -> None:
    """Rows in submit_public.PublicData order (sorted public parquet files, efit_times frames) -> <proc>/thz_public.npy."""
    import glob
    import pyarrow.parquet as pq
    raw = sorted(glob.glob(str(SRC / FD.FOLD / "*.parquet"))); jobs, a = [], 0
    for r in raw:
        t = np.asarray(pq.read_table(r, columns=["efit_times"]).column(0)[0].as_py(), float); jobs.append((a, a + len(t), r, t)); a += len(t)
    out = fx.ROOT / "data" / "sophelio_proc" / f"thz_{FD.TAG}.npy"
    np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(a, NF)).flush(); run(out, jobs)


def run(out, jobs, workers=8):
    from multiprocessing import Pool
    chunks = [jobs[i::workers] for i in range(workers)]
    with Pool(workers) as p:
        print(f"[thz] {sum(p.map(_work, [(out, c) for c in chunks]))} shots -> {out}", flush=True)


def main() -> None:
    if "--public" in sys.argv:
        return public()
    d = sota.Data(); names = d.meta["names"]; fn = d.fn; it = fn.index("t")
    if "--all" in sys.argv:
        out = sota.CACHE / "thz.npy"; starts = np.r_[0, np.nonzero(np.diff(d.shot))[0] + 1, len(d.shot)]
        jobs = [(int(a), int(b), SRC / "diii_d_train" / f"{names[int(d.shot[a])]}.parquet", np.asarray(d.X[a:b, it], float))
                for a, b in zip(starts[:-1], starts[1:])]
        return run(out, jobs)
    if "--cv" in sys.argv:
        import cv_trunk
        shots, _ = cv_trunk.folds(d, sota)
    else:
        shots = np.arange(len(names))
    out = sota.CACHE / "thz.npy"
    F = np.load(out, mmap_mode="r+") if out.exists() else np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(d.shot), 2 * len(ZC) + len(ZC) + 2 * len(RT) + 5))
    starts = np.r_[0, np.nonzero(np.diff(d.shot))[0] + 1, len(d.shot)]; first = {int(d.shot[a]): (a, b) for a, b in zip(starts[:-1], starts[1:])}
    for n, s in enumerate(shots):
        a, b = first[int(s)]; stem = names[int(s)]; split = {0: "diii_d_train", 1: "diii_d_train", 2: "diii_d_train"}[int(d.split[a])]
        path = SRC / split / f"{stem}.parquet"
        if not path.exists():
            hits = list(SRC.glob(f"*/{stem}.parquet")); path = hits[0] if hits else None
        if path is None:
            continue
        F[a:b] = shot_feats(path, np.asarray(d.X[a:b, it], float))
        if (n + 1) % 100 == 0:
            print(f"[thz] {n + 1}/{len(shots)}", flush=True)
    F.flush(); print(f"[thz] done {len(shots)} shots -> {out}", flush=True)


if __name__ == "__main__":
    main()
