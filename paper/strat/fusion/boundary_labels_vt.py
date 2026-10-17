"""Fusion §41q all-data refit: the v8 shape labels (7 consistency scalars + 64 LCFS rays, boundary_labels.py rules) for EVERY
VALID and TEST frame (true psi), so the final heads can train on all 7,041 shots. -> runs/strat/fusion/boundary_labels_v8_vt.npz
(rows, y). CPU.

    STRAT_NANG=64 .venv/bin/python paper/strat/fusion/boundary_labels_vt.py
"""
from __future__ import annotations

import os

os.environ.setdefault("STRAT_NANG", "64")
import numpy as np  # noqa: E402

import boundary_labels as BL  # noqa: E402
import fx  # noqa: E402


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    assert BL.N_ANG == 64, BL.N_ANG
    sota, d = fx.data(); rows = np.nonzero(d.split != 0)[0]; starts = list(range(0, len(rows), 2000)); res = []
    chunk = lambda i: (d.psi_of(rows[i:i + 2000]).reshape(-1, 4225).astype(np.float32),)       # noqa: E731
    with ProcessPoolExecutor(max_workers=fx.N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        for b in range(0, len(starts), 2 * fx.N_WORKERS):
            res += list(ex.map(BL._one, [chunk(i) for i in starts[b:b + 2 * fx.N_WORKERS]]))
            print(f"[labels-vt] {min(b + 2 * fx.N_WORKERS, len(starts))}/{len(starts)} chunks", flush=True)
    Y = np.concatenate(res); np.savez(fx.OUT / "boundary_labels_v8_vt.npz", rows=rows, y=Y)
    old = np.load(fx.OUT / "boundary_labels_v8.npz"); pos = {r: i for i, r in enumerate(rows)}
    k = np.array([pos[r] for r in old["rows_valid"]]); a, b = Y[k], old["y_valid"]; m = np.isfinite(a) & np.isfinite(b)
    print(f"LABELS_VT_DONE rows {len(rows)} nan-frac kappa {np.isnan(Y[:, 2]).mean():.4f} | max |diff| vs v8 VALID sample {np.abs(a - b)[m].max():.2e}", flush=True)


if __name__ == "__main__":
    main()
