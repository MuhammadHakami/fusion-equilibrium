"""Doc 35 §38g: corrected magnetics block (Ip re-anchored on 20 kHz shots + lags in true ms; mag_fix.py) as an input
transformation for any model. Replaces the 294 magnetics columns of the cached feature matrix; nothing else changes.

    import inputs_fix as IF; IF.use(data)                 # sota.Data: X_magfix_all.npy (all splits) or X_magfix.npy
    IF.use(public_data, public=True)                     # submit_public.PublicData: X_magfix_public.npy
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"   # no fx import: it would hide the GPU
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import fold as FD  # noqa: E401,E402,I001


_NAME = __import__("os").environ.get("QW_MAGFIX", "X_magfix")      # X_ipfix: organizers' Ip time fix (ip_fix.py)


def use(data, public: bool = False, name: str | None = None):
    """Patch data.inputs in place to read the corrected magnetics block (name: X_magfix / X_ipfix, default QW_MAGFIX); returns data."""
    name = name or _NAME
    if public:
        mf = np.load(OUT / f"{name}_{FD.TAG}.npy", mmap_mode="r"); has_test = True
    else:
        full = OUT / f"{name}_all.npy"; has_test = full.exists()
        mf = np.load(full if has_test else OUT / "X_magfix.npy", mmap_mode="r")
    assert len(mf) == len(data.X), (len(mf), len(data.X))
    cols = np.array([data.fn.index(c) for c in json.loads((OUT / "magfix.json").read_text())["columns"]])

    def inputs(rows, fm, edge=False, hist=False, thz=False):
        if public and (thz or hist):                         # §42: the thz / hist branches below read TRAIN-cache rows
            raise RuntimeError("inputs_fix: thz / hist trunk inputs are cache-only; use hfull_inputs (kind thz) for public folds")
        if not has_test:
            assert (data.split[rows] != 2).all(), "TEST rows need X_magfix_all.npy (mag_fix.py --target test)"
        x = np.array(data.X[rows], dtype=np.float32); x[:, cols] = mf[rows]; x = x[:, fm]
        x = np.concatenate([x, data.edge_feats()[rows]], 1) if edge else x
        if thz:                                              # §41f fixed-coordinate Thomson (cache or public rows)
            if getattr(data, "_thz", None) is None:
                import sota
                data._thz = np.load(sota.CACHE / "thz.npy", mmap_mode="r")
            x = np.concatenate([x, np.asarray(data._thz[rows])], 1)
        if hist:                                             # §41c long-window input history (cache rows only)
            if getattr(data, "_hist", None) is None:
                import sota
                data._hist = np.load(sota.CACHE / "hist.npy", mmap_mode="r")
            x = np.concatenate([x, np.asarray(data._hist[rows])], 1)
        return x
    data.inputs = inputs
    return data
