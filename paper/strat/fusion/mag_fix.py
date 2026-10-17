"""Doc 35 §37c (exploratory, TRAIN/VALID only): corrected magnetics block for the Fusion Equilibrium cache.

Two input errors found in §37 (inputs only, no label read):
  1. Our extraction (paper/sophelio/extract.py) sampled the 13 "lags" at t + l/1000 with t in ms, so the intended
     -200 ... +20 ms offsets are -0.2 ... +0.02 ms: the lag features are copies of the current frame.
  2. The challenge ships one shared plasma-current time vector (-858.19 + 0.5 i ms) for every shot. On 20 kHz shots it
     is wrong: read that way |Ip| < 5% of its peak at 57-100% of EFIT frames (impossible; EFIT frames need plasma).
     Anchored at magnetics_time[0] - 2.25 ms (runs/strat/fusion/ip_align.json, ECOILA drive correlation), no EFIT
     frame is near zero. 2 kHz shots pass the same check as shipped and are left unchanged.
This writes runs/strat/fusion/X_magfix.npy (N_frames, 21 * 14) in the cache's column order (mag_<s>, then
mag_<s>_lag<l>ms for l in LAGS_MS), with Ip re-anchored on 20 kHz shots and lags in true ms. TEST rows stay zero and
their shots are never read.

Checks (the script stops on failure): (a) with the old rules it reproduces the cached X columns exactly on 20 shots;
(b) frame counts per shot match the cache; (c) with the new Ip time base |Ip| at t = -50 ms (before breakdown) is < 5%
of the shot's peak on 20 kHz shots.

    .venv/bin/python paper/strat/fusion/mag_fix.py [--workers 6]
    .venv/bin/python paper/strat/fusion/mag_fix.py --target test     -> X_magfix_all.npy (X_magfix + TEST rows; §38g)
    .venv/bin/python paper/strat/fusion/mag_fix.py --target public   -> X_magfix_public.npy (public test frames, file order)
Inputs only in every case; no label is read. Check (a) runs on each target's own shots first.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
import extract as EX  # noqa: E402

RAW = ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data" / "diii_d_train"
CACHE = ROOT / "data" / "sophelio_proc" / "cache_k256"
OUT = ROOT / "runs" / "strat" / "fusion"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import fold as FD  # noqa: E402,I001
IP_OFF_20K = -2.25
COLS = ["efit_times", "magnetics_time", "magnetics_plasma_current_times"] + [f"magnetics_{s}" for s in EX.SIGNALS]


def is_20k(mt: np.ndarray) -> bool:
    return float(np.median(np.diff(mt[:1000]))) < 0.1


def mag_block(row, fixed: bool) -> np.ndarray:
    """(T, 21 * 14) magnetics features at the EFIT frames; fixed=False reproduces the cache's rule."""
    t = np.asarray(row["efit_times"], np.float64)
    r = row.copy()
    if fixed and is_20k(np.asarray(row["magnetics_time"], np.float64)):
        n = len(np.asarray(row["magnetics_plasma_current_times"]))
        r["magnetics_plasma_current_times"] = float(np.asarray(row["magnetics_time"])[0]) + IP_OFF_20K + 0.5 * np.arange(n)
    scale = 1.0 if fixed else 1.0 / 1000.0
    return np.concatenate([EX._mag(r, t)] + [EX._mag(r, t + l * scale) for l in EX.LAGS_MS], 1)


def one(name: str, raw: pathlib.Path = RAW):
    row = pd.read_parquet(raw / f"{name}.parquet", columns=COLS).iloc[0]
    mt = np.asarray(row["magnetics_time"], np.float64)
    fast = is_20k(mt)
    blk = mag_block(row, True)
    pre = None
    if fast:                                                # check (c): before breakdown Ip must be ~0
        r = row.copy(); n = len(np.asarray(row["magnetics_plasma_current_times"]))
        r["magnetics_plasma_current_times"] = float(mt[0]) + IP_OFF_20K + 0.5 * np.arange(n)
        ip = np.asarray(row["magnetics_plasma_current"], np.float64)
        pre = float(abs(EX._mag(r, np.array([-50.0]))[0, -1]) / max(np.nanmax(np.abs(ip)), 1e-9))
    return name, blk.astype(np.float32), fast, pre


def _one_public(name: str):
    return one(name, RAW.parent / FD.FOLD)


def build_public(workers: int) -> None:
    """Corrected magnetics block for the public DIII-D test frames, in PublicData's file order (sorted names)."""
    import torch
    fn = torch.load(CACHE / "meta.pt", weights_only=False)["feature_names"]
    cols = [fn.index(f"mag_{s}") for s in EX.SIGNALS] + [fn.index(f"mag_{s}_lag{l:+d}ms") for l in EX.LAGS_MS for s in EX.SIGNALS]
    proc = ROOT / "data" / "sophelio_proc" / FD.FOLD; raw = RAW.parent / FD.FOLD
    names = [f.stem for f in sorted(proc.glob("*.npz"))]
    assert names == [f.stem for f in sorted(raw.glob("*.parquet"))]
    for nm in names[:20]:                                                     # (a) old rule = the processed public features
        row = pd.read_parquet(raw / f"{nm}.parquet", columns=COLS).iloc[0]
        ref = np.load(proc / f"{nm}.npz")["X"][:, cols]; old = mag_block(row, False)
        err = float(np.abs(old - ref).max() / (np.abs(ref).max() + 1e-9)); assert old.shape == ref.shape and err < 1e-5, (nm, err)
    print("[magfix] public check (a) passed on 20 shots", flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = dict((n, (b, f, p)) for n, b, f, p in pool.imap_unordered(_one_public, names, chunksize=4))
    for nm in names:                                                          # (b) frame counts
        assert len(res[nm][0]) == len(np.load(proc / f"{nm}.npz")["X"]), nm
    pre = np.asarray([res[n][2] for n in names if res[n][2] is not None])
    rep = {"n_shots": len(names), "n_20k": int(sum(res[n][1] for n in names)), "pre_breakdown_ip_frac": {
        "median": float(np.median(pre)), "p95": float(np.percentile(pre, 95)), "frac_over_5pct": float((pre > 0.05).mean())},
        "seconds": round(time.time() - t0)}
    if rep["n_20k"] >= 50:                                                    # (c) statistical; a smoke fold only reports it
        assert rep["pre_breakdown_ip_frac"]["frac_over_5pct"] < 0.05, rep
    else:
        print(f"[magfix] check (c) skipped on {rep['n_20k']} 20 kHz shots: {rep['pre_breakdown_ip_frac']}", flush=True)
    np.save(OUT / f"X_magfix_{FD.TAG}.npy", np.concatenate([res[n][0] for n in names]).astype(np.float32))
    (OUT / f"magfix_{FD.TAG}.json").write_text(json.dumps(rep, indent=1)); print(f"[magfix] public done {json.dumps(rep)}", flush=True)


def build_test(workers: int) -> None:
    """X_magfix_all.npy = X_magfix.npy with the held-out TEST rows filled by the same rule (inputs only)."""
    import torch
    meta = torch.load(CACHE / "meta.pt", weights_only=False); names = meta["names"]
    shot = np.load(CACHE / "shot.npy"); split = np.load(CACHE / "split.npy"); X = np.load(CACHE / "X.npy", mmap_mode="r")
    fn = meta["feature_names"]
    cols = [fn.index(f"mag_{s}") for s in EX.SIGNALS] + [fn.index(f"mag_{s}_lag{l:+d}ms") for l in EX.LAGS_MS for s in EX.SIGNALS]
    starts = np.r_[0, np.nonzero(np.diff(shot))[0] + 1]; counts = np.bincount(shot, minlength=len(names))
    todo = [j for j in range(len(names)) if split[starts[j]] == 2]
    for j in todo[:20]:                                                       # (a) on TEST shots' inputs
        row = pd.read_parquet(RAW / f"{names[j]}.parquet", columns=COLS).iloc[0]
        old = mag_block(row, False); ref = np.asarray(X[starts[j]:starts[j] + counts[j]])[:, cols]
        assert old.shape == ref.shape and float(np.abs(old - ref).max() / (np.abs(ref).max() + 1e-9)) < 1e-5, names[j]
    print("[magfix] test check (a) passed on 20 shots", flush=True)
    out = np.lib.format.open_memmap(OUT / "X_magfix_all.tmp.npy", mode="w+", dtype=np.float32, shape=(len(shot), len(cols)))
    out[:] = np.load(OUT / "X_magfix.npy", mmap_mode="r")
    j_of = {names[j]: j for j in todo}; t0 = time.time()
    with mp.Pool(workers) as pool:
        for name, blk, fast, p in pool.imap_unordered(one, [names[j] for j in todo], chunksize=4):
            j = j_of[name]; assert len(blk) == counts[j], name; out[starts[j]:starts[j] + counts[j]] = blk
    out.flush(); del out
    (OUT / "X_magfix_all.tmp.npy").rename(OUT / "X_magfix_all.npy")
    print(f"[magfix] test rows done: {len(todo)} shots {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--target", default="trainvalid", choices=["trainvalid", "test", "public"])
    a = ap.parse_args()
    if a.target == "test":
        return build_test(a.workers)
    if a.target == "public":
        return build_public(a.workers)
    import torch
    meta = torch.load(CACHE / "meta.pt", weights_only=False)
    names, fn = meta["names"], meta["feature_names"]
    shot = np.load(CACHE / "shot.npy"); split = np.load(CACHE / "split.npy")
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    cols = [fn.index(f"mag_{s}") for s in EX.SIGNALS] + [fn.index(f"mag_{s}_lag{l:+d}ms") for l in EX.LAGS_MS for s in EX.SIGNALS]
    starts = np.r_[0, np.nonzero(np.diff(shot))[0] + 1]; counts = np.bincount(shot, minlength=len(names))
    shot_split = split[starts]
    # (a) the old rule reproduces the cache exactly (20 TRAIN shots)
    rng = np.random.default_rng(0)
    for j in rng.choice(np.nonzero(shot_split == 0)[0], 20, replace=False):
        row = pd.read_parquet(RAW / f"{names[j]}.parquet", columns=COLS).iloc[0]
        old = mag_block(row, False); ref = np.asarray(X[starts[j]:starts[j] + counts[j]])[:, cols]
        err = float(np.abs(old - ref).max() / (np.abs(ref).max() + 1e-9))
        assert old.shape == ref.shape and err < 1e-5, (names[j], old.shape, ref.shape, err)
    print("[magfix] check (a) passed: the old rule reproduces the cached magnetics columns on 20 shots", flush=True)
    todo = [j for j in range(len(names)) if shot_split[j] != 2]
    out = np.lib.format.open_memmap(OUT / "X_magfix.tmp.npy", mode="w+", dtype=np.float32, shape=(len(shot), len(cols)))
    t0 = time.time(); n20 = 0; pre = []
    with mp.Pool(a.workers) as pool:
        for i, (name, blk, fast, p) in enumerate(pool.imap_unordered(one, [names[j] for j in todo], chunksize=4)):
            j = NAME2J[name]
            assert len(blk) == counts[j], (name, len(blk), counts[j])            # (b)
            out[starts[j]:starts[j] + counts[j]] = blk
            n20 += fast
            if p is not None:
                pre.append(p)
            if (i + 1) % 500 == 0:
                print(f"[magfix] {i + 1}/{len(todo)} shots {time.time() - t0:.0f}s", flush=True)
    out.flush(); del out
    pre = np.asarray(pre)
    rep = {"n_shots": len(todo), "n_20k": int(n20), "pre_breakdown_ip_frac": {"median": float(np.median(pre)),
           "p95": float(np.percentile(pre, 95)), "frac_over_5pct": float((pre > 0.05).mean())},
           "columns": [fn[c] for c in cols], "seconds": round(time.time() - t0)}
    assert rep["pre_breakdown_ip_frac"]["frac_over_5pct"] < 0.05, rep["pre_breakdown_ip_frac"]   # (c)
    (OUT / "X_magfix.tmp.npy").rename(OUT / "X_magfix.npy")
    (OUT / "magfix.json").write_text(json.dumps(rep, indent=1))
    print(f"[magfix] done {json.dumps({k: v for k, v in rep.items() if k != 'columns'})}", flush=True)


if __name__ == "__main__":
    import torch as _t  # noqa: F401
    _meta = _t.load(CACHE / "meta.pt", weights_only=False)
    NAME2J = {n: j for j, n in enumerate(_meta["names"])}
    main()
