"""Fusion: rebuild the 14 Ip columns of the corrected magnetics block with the ORGANIZERS' Ip time-axis fix (starter kit
data_fixes.fix_d3d_ip_times, erratum #6), replacing our own re-anchoring (mag_fix.py: 20 kHz shots, mt[0] - 2.25 ms + 0.5 ms
steps), which differs from it by 1-4.5% of Ip on the affected shots (25-shot check, 2026-10-10). Every other column is kept.
Writes X_ipfix_all.npy (cache row order) and X_ipfix_public.npy (PublicData order) next to the X_magfix files; select them
with QW_MAGFIX=X_ipfix (inputs_fix.py).

    .venv/bin/python paper/strat/fusion/ip_fix.py --workers 8
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

EX = MF.EX
STARTER = MF.ROOT / "baseline" / "sophelio-starter"
sys.path.insert(0, str(STARTER))
from data_fixes import fix_d3d_ip_times  # noqa: E402

IPCOL = EX.SIGNALS.index("plasma_current")
IPCOLS = [IPCOL + 21 * k for k in range(1 + len(EX.LAGS_MS))]     # lag-0 block, then one 21-block per lag


def ip_block(raw_dir: pathlib.Path, name: str):
    row = pd.read_parquet(raw_dir / f"{name}.parquet", columns=MF.COLS).iloc[0]
    r = row.copy(); r["magnetics_plasma_current_times"] = fix_d3d_ip_times(row)
    t = np.asarray(row["efit_times"], np.float64)
    blk = np.concatenate([EX._mag(r, t)] + [EX._mag(r, t + l) for l in EX.LAGS_MS], 1)
    return name, blk[:, IPCOLS].astype(np.float32)


def _train(name):
    return ip_block(MF.RAW, name)


def _public(name):
    return ip_block(MF.RAW.parent / FD.FOLD, name)


def patch(src: pathlib.Path, dst: pathlib.Path, names, counts, fn, workers):
    X = np.load(src, mmap_mode="r"); out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=X.shape)
    out[:] = X[:]; starts = np.r_[0, np.cumsum(counts)[:-1]]; pos = dict(zip(names, zip(starts, counts))); diffs = []
    with mp.Pool(workers) as pool:
        for i, (nm, blk) in enumerate(pool.imap_unordered(fn, names, chunksize=4)):
            s, c = pos[nm]; assert len(blk) == c, (nm, len(blk), c)
            old = np.asarray(X[s:s + c][:, IPCOLS]); out[s:s + c, IPCOLS] = blk
            diffs.append(float(np.abs(old - blk).max() / (np.abs(blk).max() + 1e-9)))
            if (i + 1) % 1000 == 0:
                print(f"[ipfix] {dst.name} {i + 1}/{len(names)}", flush=True)
    out.flush(); d = np.asarray(diffs)
    print(f"[ipfix] {dst.name}: shots changed (>0.1%) {int((d > 1e-3).sum())}/{len(d)}, max rel change {d.max():.4f}, "
          f"median of changed {np.median(d[d > 1e-3]) if (d > 1e-3).any() else 0:.4f}", flush=True)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--fold-only", action="store_true"); a = ap.parse_args()
    meta = torch.load(MF.CACHE / "meta.pt", weights_only=False); names = list(meta["names"]); shot = np.load(MF.CACHE / "shot.npy")
    if not a.fold_only:                                                           # Final day: test fold only
        patch(MF.OUT / "X_magfix_all.npy", MF.OUT / "X_ipfix_all.npy", names, np.bincount(shot, minlength=len(names)), _train, a.workers)
    proc = MF.ROOT / "data" / "sophelio_proc" / FD.FOLD; pn = [f.stem for f in sorted(proc.glob("*.npz"))]
    pc = [len(np.load(proc / f"{nm}.npz")["t"]) for nm in pn]
    patch(MF.OUT / f"X_magfix_{FD.TAG}.npy", MF.OUT / f"X_ipfix_{FD.TAG}.npy", pn, np.asarray(pc), _public, a.workers)
    print("IPFIX_DONE", flush=True)


if __name__ == "__main__":
    main()
