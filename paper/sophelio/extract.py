"""Full-data feature extraction for the Sophelio DIII-D equilibrium task (all 7,041 training shots).

    .venv/bin/python paper/sophelio/extract.py --split diii_d_train --workers 8   ->  data/sophelio_proc/<split>/<shot>.npz

Per EFIT frame (inputs only; no efit_* label is ever read into X):
  * 21 magnetics signals (coil currents incl. Ip) at the frame time and at 13 time lags (-200 … +20 ms);
    offline reconstruction has the whole discharge, so symmetric context is legal and captures dI/dt (eddy currents);
  * time into the discharge;
  * Thomson core / edge Te, ne and pe = ne·Te: validity count, mean, max, 10/50/90th percentiles, and profiles
    resampled to fixed length along the chord order (core 24, edge 8 points);
Targets (training split only): psi (65x65), and the six EFIT scalar labels (beta_n, li, q95, r_axis, z_axis, dsep).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data"
DST = ROOT / "data" / "sophelio_proc"
SIGNALS = ["ECOILA", "bcoil", "F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B", "F6A", "F6B",
           "F7A", "F7B", "F8A", "F8B", "F9A", "F9B", "plasma_current"]
LAGS_MS = [-200, -100, -50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20]
LABELS = ["efit_beta_n", "efit_li", "efit_q95", "efit_r_axis", "efit_z_axis", "magnetics_dsep"]
PROFILE_LEN = {"thomson_core": 24, "thomson_edge": 8}


def feature_names() -> list[str]:
    names = [f"mag_{s}" for s in SIGNALS]
    names += [f"mag_{s}_lag{l:+d}ms" for l in LAGS_MS for s in SIGNALS]
    names += ["t"]
    for sysk, n in PROFILE_LEN.items():
        for q in ("Te", "ne", "pe"):
            names += [f"{sysk}_{q}_{k}" for k in ("count", "mean", "max", "p10", "p50", "p90")]
            names += [f"{sysk}_{q}_prof{i}" for i in range(n)]
    return names


def _mag(row, t_query: np.ndarray) -> np.ndarray:
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_t = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    out = np.zeros((len(t_query), len(SIGNALS)), dtype=np.float32)
    for j, s in enumerate(SIGNALS):
        col = f"magnetics_{s}"
        if col not in row.index or row[col] is None:
            continue
        v = np.asarray(row[col], dtype=np.float64); t = ip_t if s == "plasma_current" else shared
        m = np.isfinite(v)
        if m.sum() >= 2:
            out[:, j] = np.interp(t_query, t[m], v[m])
    return out


def _thomson(row, key: str, t_query: np.ndarray) -> np.ndarray:
    n = PROFILE_LEN[key]; T = len(t_query)
    feats = np.zeros((T, 3 * (6 + n)), dtype=np.float32)
    if f"{key}_times" not in row.index or row[f"{key}_times"] is None:
        return feats
    tt = np.asarray(row[f"{key}_times"], dtype=np.float64)
    Te = np.stack([np.asarray(x, dtype=np.float64) for x in row[f"{key}_Te"]]) if len(tt) else None
    ne = np.stack([np.asarray(x, dtype=np.float64) for x in row[f"{key}_ne"]]) if len(tt) else None
    if Te is None or Te.ndim != 2 or len(tt) < 2:
        return feats
    # nearest Thomson frame in time for each EFIT frame (profiles are sparse in time; no smearing across frames)
    idx = np.clip(np.searchsorted(tt, t_query), 1, len(tt) - 1)
    idx = np.where(np.abs(tt[idx - 1] - t_query) < np.abs(tt[idx] - t_query), idx - 1, idx)
    for q, arr in enumerate((Te, ne, Te * ne)):
        a = arr[idx]                                                    # (T, channels)
        good = np.isfinite(a) & (a > 0)
        cnt = good.sum(1)
        with np.errstate(all="ignore"):
            am = np.where(good, a, np.nan)
            stats = np.stack([cnt, np.nanmean(am, 1), np.nanmax(am, 1), *np.nanpercentile(am, [10, 50, 90], axis=1)], 1)
            prof = np.zeros((T, n))
            for k in range(T):
                g = np.nonzero(good[k])[0]
                if len(g) >= 2:
                    prof[k] = np.interp(np.linspace(0, len(a[k]) - 1, n), g, a[k, g])
        base = q * (6 + n)
        feats[:, base:base + 6] = np.nan_to_num(stats)
        feats[:, base + 6:base + 6 + n] = np.nan_to_num(prof)
    return feats


def process(path: pathlib.Path, out_dir: pathlib.Path, with_targets: bool) -> tuple[str, int, str]:
    dst = out_dir / (path.stem + ".npz")
    if dst.exists():
        return path.stem, -1, "exists"
    try:
        row = pd.read_parquet(path).iloc[0]
        t = np.asarray(row["efit_times"], dtype=np.float64)
        parts = [_mag(row, t)] + [_mag(row, t + l / 1000.0) for l in LAGS_MS] + [t[:, None].astype(np.float32)]
        parts += [_thomson(row, k, t) for k in PROFILE_LEN]
        X = np.concatenate(parts, axis=1).astype(np.float32)
        save = dict(X=X, t=t, R=np.asarray(row["efit_grid_R"], np.float32), Z=np.asarray(row["efit_grid_Z"], np.float32))
        if with_targets:
            save["psi"] = np.stack([np.stack([np.asarray(r, np.float32) for r in fr]) for fr in row["efit_psirz"]])
            save["S"] = np.stack([np.asarray(row[l], np.float64).ravel()[:len(t)] if l in row.index and row[l] is not None
                                  else np.full(len(t), np.nan) for l in LABELS], 1).astype(np.float32)
            save["S"][:, LABELS.index("magnetics_dsep")][save["S"][:, LABELS.index("magnetics_dsep")] == -1.0] = np.nan
        np.savez(dst, **save)
        return path.stem, len(t), "ok"
    except Exception as e:  # noqa: BLE001 — record and continue; a failed shot is reported, never silently dropped
        return path.stem, 0, f"FAIL {type(e).__name__}: {e}"


def _job(args):
    return process(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="diii_d_train"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted((SRC / a.split).glob("*.parquet"))[: a.limit or None]
    out = DST / a.split; out.mkdir(parents=True, exist_ok=True)
    (DST / "feature_names.txt").write_text("\n".join(feature_names()) + "\n")
    with_targets = a.split.endswith("train")
    t0, ok, fails, frames = time.time(), 0, [], 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (name, n, status) in enumerate(pool.imap_unordered(_job, [(f, out, with_targets) for f in files], chunksize=4)):
            if status == "ok":
                ok += 1; frames += n
            elif status.startswith("FAIL"):
                fails.append(f"{name}\t{status}")
            if (i + 1) % 250 == 0:
                print(f"[extract] {a.split} {i+1}/{len(files)} ok {ok} fail {len(fails)} frames {frames} {time.time()-t0:.0f}s", flush=True)
    (DST / f"{a.split}_failures.txt").write_text("\n".join(fails) + ("\n" if fails else ""))
    print(f"[extract] {a.split} done: {ok} new, {len(fails)} failed, {frames} frames, {time.time()-t0:.0f}s", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
