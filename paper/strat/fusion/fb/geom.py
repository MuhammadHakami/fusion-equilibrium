"""Doc 35 §46: machine geometry for the free-boundary Grad-Shafranov solver (DIII-D and MAST, from the shipped columns only).

Per machine, from one shot row of the challenge data:
- the 65 x 65 EFIT grid (psirz layout: Z rows, R columns);
- per-circuit coil flux tables G_c (n_circ, nZ * nR) in Wb/rad per kA of the circuit's input column. MAST p-coil columns are
  kA-turn totals of the circuit, spread evenly over its conductor elements; the MAST solenoid column is per-turn kA on 656
  single-turn elements (x sol_factor: the organizers' open 2x caveat); DIII-D F-coil columns are kA-turn of one lumped element;
  DIII-D ECOILA (kA, no turn count) gets an explicit turn count argument. Elements are integrated over their (parallelogram)
  cross-section with n x n sub-filaments;
- the plasma Green's matrix Gpp (N, N), psi at grid point i per A in the grid cell j, with the singular self term replaced by the
  cell average (sub-sampled source cell, offset so no sub-point sits on the target);
- the limiter mask: grid points where plasma may exist (a geometric box minus coil neighbourhoods; no LCFS data of any kind).
No equilibrium labels are read here.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "sophelio", "fusion-equilibrium-challenge", "data")


def filament_psi(a, b, R, Z):
    """Poloidal flux per radian [Wb/rad] at (R, Z) of a 1 A circular filament at (a, b)."""
    m = np.clip(4 * a * R / ((a + R) ** 2 + (Z - b) ** 2), 1e-12, 1 - 1e-12); k = np.sqrt(m)
    return MU0 * np.sqrt(a * R) / (np.pi * k) * ((1 - m / 2) * ellipk(m) - ellipe(m))


def coil_points(R, Z, w, h, a1, a2, n=6):
    """Sub-filament centres of an EFIT coil element (w x h rectangle sheared by angle1 / angle2, EFIT convention)."""
    u = (np.arange(n) + 0.5) / n - 0.5; du, dv = np.meshgrid(u, u, indexing="ij"); x, y = du * w, dv * h
    if a1 and abs(a1) != 90:
        x = x + y / np.tan(np.radians(a1))
    if a2 and abs(a2) != 90:
        y = y + x * np.tan(np.radians(a2))
    return (R + x).ravel(), (Z + y).ravel()


@dataclass
class Machine:
    name: str
    R: np.ndarray                 # (nR,)
    Z: np.ndarray                 # (nZ,)
    circuits: list                # input column names, order of G_c rows
    Gc: np.ndarray                # (n_circ, nZ * nR) Wb/rad per kA of the column
    Gpp: np.ndarray               # (N, N) Wb/rad per A
    limiter: np.ndarray           # (nZ, nR) bool, plasma allowed
    R0: float                     # reference major radius for the profile
    el_R: np.ndarray              # coil element centres (for the limiter / plots)
    el_Z: np.ndarray


def first_row(machine: str) -> pd.Series:
    split = "mast_public_test" if machine == "mast" else "diii_d_train"
    f = sorted(glob.glob(f"{DATA}/{split}/*.parquet"))[0]
    return pd.read_parquet(f).iloc[0]


def coil_tables(r: pd.Series, RR: np.ndarray, ZZ: np.ndarray, sol_factor: float = 1.0, ecoil_turns: float = 1.0,
                p2_outer_sign: float = 1.0, n_sub: int = 6):
    cols = [str(c) for c in r["coil_input_column"]]; names = [str(c) for c in r["coil_name"]]
    circuits = list(dict.fromkeys(cols)); count = {c: cols.count(c) for c in circuits}
    G = {c: np.zeros(RR.size) for c in circuits}
    for i, c in enumerate(cols):
        pr, pz = coil_points(r["coil_R"][i], r["coil_Z"][i], r["coil_width"][i], r["coil_height"][i], r["coil_angle1"][i], r["coil_angle2"][i], n_sub)
        g = sum(filament_psi(a, b, RR, ZZ) for a, b in zip(pr, pz)).ravel() / len(pr)          # per A in the element
        if c == "magnetics_sol_current":
            w = 1e3 * sol_factor                                                               # per-turn kA, single-turn elements
        elif c == "magnetics_ECOILA":
            w = 1e3 * ecoil_turns
        else:
            w = 1e3 / count[c]                                                                 # kA-turn total spread over elements
            if "p2_outer" in names[i]:
                w *= p2_outer_sign
        G[c] += w * g
    return circuits, np.stack([G[c] for c in circuits])


def plasma_green(R: np.ndarray, Z: np.ndarray, n_self: int = 8) -> np.ndarray:
    RR, ZZ = np.meshgrid(R, Z, indexing="xy"); r, z = RR.ravel(), ZZ.ravel(); N = r.size
    G = np.empty((N, N))
    for j in range(N):                                                                         # column j: source cell j
        G[:, j] = filament_psi(r[j], z[j], r, z)
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    u = (np.arange(n_self) + 0.5) / n_self - 0.5                                               # never 0 for even n_self
    du, dv = np.meshgrid(u * dR, u * dZ, indexing="ij")
    for j in range(N):
        G[j, j] = filament_psi(r[j] + du.ravel(), z[j] + dv.ravel(), r[j], z[j]).mean()
    return G


def limiter_mask(machine: str, R: np.ndarray, Z: np.ndarray, el_R: np.ndarray, el_Z: np.ndarray, clear: float = 0.08) -> np.ndarray:
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    if machine == "mast":                       # centre-column armour ~0.2 m; in-vessel coils excluded below; vessel |Z| < 1.8
        box = (RR > 0.19) & (RR < 1.95) & (np.abs(ZZ) < 1.8)
    else:                                       # DIII-D: the EFIT grid is the vessel interior; keep one cell off the edge
        box = (RR > R[1]) & (RR < R[-2]) & (ZZ > Z[1]) & (ZZ < Z[-2])
    d2 = np.min((RR.ravel()[:, None] - el_R[None]) ** 2 + (ZZ.ravel()[:, None] - el_Z[None]) ** 2, 1).reshape(RR.shape)
    return box & (d2 > clear ** 2)


def build(machine: str, **kw) -> Machine:
    r = first_row(machine)
    R, Z = np.asarray(r["efit_grid_R"], np.float64), np.asarray(r["efit_grid_Z"], np.float64)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    circuits, Gc = coil_tables(r, RR, ZZ, **kw)
    el_R, el_Z = np.asarray(r["coil_R"], np.float64), np.asarray(r["coil_Z"], np.float64)
    lim = limiter_mask(machine, R, Z, el_R, el_Z)
    return Machine(machine, R, Z, circuits, Gc, plasma_green(R, Z), lim, 0.85 if machine == "mast" else 1.69, el_R, el_Z)
