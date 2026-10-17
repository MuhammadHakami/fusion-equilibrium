"""Fusion §45j: vacuum-field inputs - the measured F-coil currents mapped through exact Green's functions (the shipped DIII-D coil
geometry, fb/geom.py) to the field the plasma sees. All of it is linear in the currents (or in currents / |Ip|), so an MLP could learn
it, but only along coil-current directions present in TRAIN; the error lives on shots with unusual coil patterns (§42w: OOD mean-7
0.955-0.965 vs 0.988), where different coil combinations can still make an ordinary field at the plasma. Appended block (lag 0):
  81 points (R 1.05-2.25 x Z -1.2..1.2, 9 x 9): sign(Ip) psi_vac minus its mean over the points [Wb/rad], and the same per |Ip| [Wb/rad/MA];
  vertical / radial vacuum field at the axis region (1.69, 0) and at the two X-point regions (1.35, +-1.1), x 2 pi R / (mu0 |Ip|);
  decay index -(R / B_Z) dB_Z / dR at (1.69, 0).
The E-coil is left out (unknown turn count; its field in the plasma region is a near-uniform flux, irrelevant to shape).
    import vac_inputs as VI; VI.use(data)        # wrap after the other input blocks (qphys last before this)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fb"))
MU0 = 4e-7 * np.pi
COILS = [f"F{i}{s}" for i in range(1, 10) for s in "AB"]
_T = {}


def tables():
    """Green's tables (18 coils x points) per kA-turn: psi at the 81 grid points and at the 3 field points (+- dR, dZ stencils)."""
    if "G" in _T:
        return _T
    import geom
    r = geom.first_row("d3d"); names = [str(c) for c in r["coil_name"]]
    gr, gz = np.meshgrid(np.linspace(1.05, 2.25, 9), np.linspace(-1.2, 1.2, 9), indexing="xy")
    fp = np.array([(1.69, 0.0), (1.35, 1.1), (1.35, -1.1)]); h = 0.01
    st = np.concatenate([fp + [[dr, dz]] for dr, dz in ((0, 0), (h, 0), (-h, 0), (0, h), (0, -h), (2 * h, 0), (-2 * h, 0))])
    pts = np.concatenate([np.stack([gr.ravel(), gz.ravel()], 1), st])
    G = np.zeros((len(COILS), len(pts)))
    for k, c in enumerate(COILS):
        i = names.index(c)
        pr, pz = geom.coil_points(r["coil_R"][i], r["coil_Z"][i], r["coil_width"][i], r["coil_height"][i], r["coil_angle1"][i], r["coil_angle2"][i], 6)
        G[k] = sum(geom.filament_psi(a, b, pts[:, 0], pts[:, 1]) for a, b in zip(pr, pz)) / len(pr) * 1e3      # per kA-turn
    _T.update(G=G, n_grid=81, fp=fp, h=h)
    return _T


def feats(x: np.ndarray, cols: list[str]) -> np.ndarray:
    T = tables(); ix = {n: i for i, n in enumerate(cols)}
    I = np.stack([np.asarray(x[:, ix[f"mag_{c}"]], np.float64) for c in COILS], 1)            # kA-turn
    ipr = np.asarray(x[:, ix["mag_plasma_current"]], np.float64); s = np.where(ipr < 0, -1.0, 1.0)
    ip = np.abs(ipr) + 100.0                                                                     # kA floor keeps ramps bounded
    P = s[:, None] * (I @ T["G"])                                                                # plasma-oriented vacuum flux
    g = P[:, :T["n_grid"]]; g = g - g.mean(1, keepdims=True)
    st = P[:, T["n_grid"]:].reshape(len(P), 7, 3); h = T["h"]; R = T["fp"][:, 0][None]
    dpdR = (st[:, 1] - st[:, 2]) / (2 * h); dpdZ = (st[:, 3] - st[:, 4]) / (2 * h)
    BZ, BR = dpdR / R, -dpdZ / R
    BZp = (st[:, 1] - st[:, 0]) / h / (R + h / 2); BZm = (st[:, 0] - st[:, 2]) / h / (R - h / 2)  # B_Z at R +- h/2
    dBZ = (BZp[:, 0] - BZm[:, 0]) / h
    norm = 2 * np.pi * R / (MU0 * ip[:, None] * 1e3)
    dec = -(R[:, 0] / np.where(np.abs(BZ[:, 0]) > 1e-6, BZ[:, 0], 1e-6)) * dBZ
    F = np.concatenate([g, g / (ip[:, None] / 1e3), BZ * norm, BR * norm, np.clip(dec, -20, 20)[:, None]], 1)
    return np.clip(np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3).astype(np.float32)


def use(data):
    inner, fn = data.inputs, list(data.fn)

    def inputs(rows, fm, edge=False, hist=False, thz=False):
        x = inner(rows, fm, edge, hist=hist, thz=thz) if (hist or thz) else inner(rows, fm, edge)
        cols = [n for n, m in zip(fn, fm) if m]; base = np.asarray(x[:, :len(cols)])
        return np.concatenate([np.asarray(x, np.float32), feats(base, cols)], 1)
    data.inputs = inputs
    return data
