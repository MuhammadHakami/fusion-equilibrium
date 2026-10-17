"""Fusion §41q: append the full-rate current-history features (hist_full.py: H_full_all.npy / H_full_public.npy, 24 columns,
sign-invariant) to a data object's inputs. Composes with inputs_fix / canon (wrap after them).

    import hfull_inputs as HF; HF.use(data)            # sota.Data (cache rows)
    HF.use(public_data, public=True)                   # submit_public.PublicData
"""
from __future__ import annotations

import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import fold as FD  # noqa: E401,E402,I001


THZ = OUT.parents[2] / "data" / "sophelio_proc"


def use(data, public: bool = False, kind: str = "hfull"):
    """kind "hfull": full-rate history (24 columns); "thz": §41f fixed-coordinate Thomson (thz_feats.py) - appended at the end."""
    if kind == "thz":
        H = np.load(THZ / (f"thz_{FD.TAG}.npy" if public else "cache_k256/thz.npy"), mmap_mode="r")
    elif kind in ("ctx", "thzs", "liphys", "qshape", "subf"):   # subf: §45m sub-frame block (subf_feats.py)                                   # §42p: li current-profile proxies (liphys_feats.py)                                             # §41q shot context / smoothed thz (ctx_feats.py)
        H = np.load(OUT / f"{kind}_{FD.TAG if public else 'all'}.npy", mmap_mode="r")
    elif kind == "fut":                                                       # §41t future context (fut_feats.py)
        H = np.load(OUT / (f"F_fut_{FD.TAG}.npy" if public else "F_fut_all.npy"), mmap_mode="r")
    elif kind.startswith("thp_"):                                             # §41r Thomson on psi_N of a psi trunk set (thp_feats.py)
        H = np.load(OUT / f"thp_{FD.TAG if public else 'all'}_{kind[4:]}.npy", mmap_mode="r")
    else:
        H = np.load(OUT / (f"H_full_{FD.TAG}.npy" if public else "H_full_all.npy"), mmap_mode="r")
    assert len(H) == len(data.X), (len(H), len(data.X))
    inner = data.inputs

    def inputs(rows, fm, edge=False, hist=False, thz=False):
        x = inner(rows, fm, edge, hist=hist, thz=thz) if (hist or thz) else inner(rows, fm, edge)
        return np.concatenate([np.asarray(x, np.float32), np.asarray(H[rows], np.float32)], 1)
    data.inputs = inputs
    return data
