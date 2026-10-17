"""Batched GPU Fusion evaluation over many shots at once: flat frame arrays, chunked post-processing and scoring,
per-shot parts compatible with fx.shot_sums / fx.composite / ST.compare.

    psi, sc, ph, idx = frames(...)          # (N, 65, 65), per-frame scalars / heads, shot slices
    psi = post_all(gp, psi, axis, r_t, kappa_t, li_t)
    parts = parts_all(gp, psi, q, b, d, shots, idx)
"""
from __future__ import annotations

import numpy as np
import torch

import fx  # noqa: I001

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def chunks(n, size):
    return [slice(i, min(n, i + size)) for i in range(0, n, size)]


def post_all(gp, psi, axis, r_t, kappa_t, li_t, size=4096, steps=("shift", "bw", "kw", "li"), kw=None, tri_t=None, heads=None, xh=None):
    """Frozen §37 chain (or a subset given by steps) on (N, 65, 65) torch psi, chunked. kw: per-step keyword overrides,
    e.g. {"li": {"cap": 0.9}, "kw": {"cap": 0.2}} (defaults = the frozen values)."""
    kw = kw or {}
    out = torch.empty_like(psi); passes = int(kw.get("_passes", 1))      # §38z: repeat the chain (caps re-applied per pass)
    for c in chunks(len(psi), size):
        a = psi[c]
        for _ in range(passes):
            a = _chain(gp, a, c, axis, r_t, kappa_t, li_t, tri_t, steps, kw, heads, xh)
        out[c] = a
    return out


def _chain(gp, a, c, axis, r_t, kappa_t, li_t, tri_t, steps, kw, heads=None, xh=None):
        if "shift" in steps:
            a = gp.axis_shift(a, axis[c], **kw.get("shift", {}))
        if "bw" in steps:
            a = gp.boundary_warp(a, r_t[c], **kw.get("bw", {}))
        if "xhe" in steps:                                              # §42t: the xh step early (before kw / xp / jgn re-fit the shape)
            import xp_oracle as XO
            a = XO.xh_step(gp, a, xh[c], **kw.get("xhe", {}))
        if "kw" in steps:
            a = gp.kappa_warp(a, kappa_t[c], **kw.get("kw", {}))
        if "xp" in steps:
            a = gp.xp_warp(a, tri_t[c], **kw.get("xp", {}))
        if "tsnap" in steps:                                            # §38aa: tri grid-column search
            import joint_warp as JW
            a = JW.tri_snap(gp, a, tri_t[c], **kw.get("tsnap", {}))
        if "jgn" in steps:                                              # §38aa: joint fit to the 6 geometric heads
            a = _jgn(gp, a, c, axis, heads, dict(kw.get("jgn", {})))
        if "xh" in steps:                                               # §42t: active X-point (Newton saddle) -> the X-point heads
            import xp_oracle as XO
            a = XO.xh_step(gp, a, xh[c], **kw.get("xh", {}))
        if "li" in steps:
            a = gp.li_warp(a, li_t[c], **kw.get("li", {}))
        if "li2" in steps:                                              # §41y: second li pass (one secant step stops short: public
            a = gp.li_warp(a, li_t[c], **kw.get("li", {}))              # frames with |li - head| > 0.05: rms 0.405 -> 0.197)
        if "kw2" in steps:                                              # §42y: second kappa pass (= post_kw2.py), then li twice
            a = gp.kappa_warp(a, kappa_t[c], **kw.get("kw", {}))
            a = gp.li_warp(gp.li_warp(a, li_t[c], **kw.get("li", {})), li_t[c], **kw.get("li", {}))
        if "trs" in steps:                                              # §45t: Rgeo shift absorbs the column-quantised tri residual
            a = gp.tri_shift(a, tri_t[c], **kw.get("trs", {}))
        if "trsli" in steps:                                            # §45t: li passes after the shift
            a = gp.li_warp(gp.li_warp(a, li_t[c], **kw.get("li", {})), li_t[c], **kw.get("li", {}))
        return a


def _jgn(gp, a, c, axis, heads, o):
    """o: {"axis": "head" | "chain" | "free", "fit": [6 bools], n_it, mu, lam}; heads (N, >=6) in NAMES order."""
    import joint_warp as JW
    t = heads[c][:, :6].clone().to(a.dtype); mode = o.pop("axis", "head")
    if mode == "chain":
        t[:, :2] = axis[c]
    elif mode == "free":
        t[:, :2] = float("nan")
    jw = JW.JointWarp(gp); out = jw.fit(a, t, mask=o.pop("fit", None), **o)
    gp.jgn_diag = jw.diag
    return out


def refs(d, shots):
    """Per-frame reference contours (N, 512, 2), rgeo (N,), consistency targets (N, 7) and contour presence (N,)."""
    C, rg, cg, has = [], [], [], []
    for s in shots:
        z = np.load(fx.REF / f"ref_{d.meta['names'][s]}.npz", allow_pickle=True)
        for c in z["contours"]:
            has.append(c is not None)
            c = np.asarray(c, float) if c is not None else np.full((512, 2), np.nan)
            C.append(c[:512] if len(c) >= 512 else np.pad(c, ((0, 512 - len(c)), (0, 0)), mode="edge"))
        n = len(z["contours"]); rg.append(np.full(n, float(z["rgeo"]))); cg.append(np.where(z["cmask"], z["cons"], np.nan))
    return np.stack(C), np.concatenate(rg), np.concatenate(cg), np.array(has)


def parts_all(gp, psi, q, b, d, shots, size=2048, ref=None):
    """psi (N, 65, 65) torch in the order of fx.scored_rows over shots; q, b (N,) numpy -> {shot: parts}."""
    sota = fx.data()[0]; L = sota.LABELS
    rows = [fx.scored_rows(d, s) for s in shots]; allr = np.concatenate(rows)
    Cref, rgeo, cgt, has = ref if ref is not None else refs(d, shots)
    cp = np.empty((len(psi), 7)); dd = np.empty(len(psi)); pres, psum, psq = (np.empty(len(psi)) for _ in range(3))
    for c in chunks(len(psi), size):
        o = gp.g.derive(psi[c], torch.tensor(Cref[c]), torch.tensor(rgeo[c]))
        cp[c] = torch.stack([o[k] for k in NAMES], 1).cpu().numpy(); dd[c] = o["d"].cpu().numpy()
        g = torch.tensor(d.psi_of(allr[c]).reshape(-1, 65, 65), device=gp.dev, dtype=torch.float64)
        pres[c] = ((g - psi[c]) ** 2).flatten(1).sum(1).cpu().numpy(); psum[c] = g.flatten(1).sum(1).cpu().numpy()
        psq[c] = (g ** 2).flatten(1).sum(1).cpu().numpy()
    dd = np.where(has, dd, np.nan)
    qg, bg = d.S[allr, L.index("efit_q95")].astype(np.float64), d.S[allr, L.index("efit_beta_n")].astype(np.float64)
    parts, i = {}, 0
    for s, r in zip(shots, rows):
        j = slice(i, i + len(r)); i += len(r)
        parts[s] = dict(psi_res=pres[j], psi_sum=psum[j], psi_sq=psq[j], npix=4225, q_res=(qg[j] - q[j]) ** 2,
                        b_res=(bg[j] - b[j]) ** 2, q_gt=qg[j], b_gt=bg[j], d=dd[j], cres=(cgt[j] - cp[j]) ** 2,
                        cpred=cp[j], cgt=cgt[j], fails=0)
    return parts
