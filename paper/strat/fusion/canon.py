"""Fusion §41q: Ip-sign canonicalisation (exact Grad-Shafranov symmetry). Negating every toroidal current (Ip, the 18
F-coils, ECOILA; not bcoil - the toroidal field does not enter psi) negates psi and leaves every flux surface, the 7
consistency scalars, q95 and betaN unchanged. Each shot gets s = sign(mean corrected Ip over its frames) (inputs only);
canonical inputs are the current columns (every lag) times s, so reversed-Ip shots (8.6% of the cache shots, 8.7% of
the public test) look like normal shots. Every target is invariant: the dataset's psi is orientation-normalised (the axis
is a psi minimum on normal AND reversed shots, checked on VALID), so heads and psi trunks change only their inputs.

    import canon; canon.use(data)                  # sota.Data (cache rows)
    canon.use(public_data, public=True)            # submit_public.PublicData
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import fold as FD  # noqa: E401,E402,I001
CURRENT = ("mag_plasma_current", "mag_ECOILA") + tuple(f"mag_F{i}{ab}" for i in range(1, 10) for ab in "AB")


def is_current(name: str) -> bool:
    return name.split("_lag")[0] in CURRENT


def shot_sign(data, public: bool = False) -> np.ndarray:
    """Per-row sign of the shot's mean corrected Ip (X_ipfix block; the cached raw Ip has a broken time base)."""
    cols = json.loads((OUT / "magfix.json").read_text())["columns"]
    mf = np.load(OUT / (f"X_ipfix_{FD.TAG}.npy" if public else "X_ipfix_all.npy"), mmap_mode="r")
    assert len(mf) == len(data.X), (len(mf), len(data.X))
    ip = np.asarray(mf[:, cols.index("mag_plasma_current")], np.float64); sh = np.asarray(data.shot)
    st = np.r_[0, np.nonzero(np.diff(sh))[0] + 1]; n = np.diff(np.r_[st, len(sh)])
    assert len(np.unique(sh[st])) == len(st), "rows must be grouped by shot"
    s = np.sign(np.add.reduceat(ip, st)); s[s == 0] = 1.0
    return np.repeat(s, n).astype(np.float32)


def use(data, public: bool = False, indicator: bool = False):
    """Patch data.inputs in place: current columns times the shot's Ip sign (+ the sign itself as a last column when
    indicator, so the Ip-Bt helicity stays visible). Returns data."""
    sgn = shot_sign(data, public); fn = list(data.fn); inner = data.inputs

    def inputs(rows, fm, edge=False, hist=False, thz=False):
        x = inner(rows, fm, edge, hist=hist, thz=thz) if (hist or thz) else inner(rows, fm, edge)
        sel = np.nonzero(fm)[0] if np.asarray(fm).dtype == bool else np.asarray(fm)
        cur = np.array([is_current(fn[i]) for i in sel] + [False] * (x.shape[1] - len(sel)))
        x = np.array(x, dtype=np.float32, copy=True); x[:, cur] *= sgn[rows][:, None]
        return np.concatenate([x, sgn[rows][:, None]], 1) if indicator else x
    data.inputs = inputs; data.canon_sign = sgn
    return data
