"""Exact torch port of scipy.ndimage.map_coordinates(order=3, mode="nearest", prefilter=True) on a fixed 2-D grid.
scipy pads 12 cells with edge values, applies the cubic B-spline prefilter (mode "nearest") and evaluates the spline
on the padded array. Both linear steps are built once as per-axis matrices from scipy itself, so the GPU result
matches the CPU warps to round-off (checked in __main__).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import spline_filter1d

NPAD = 12


def _axis_matrix(n: int) -> np.ndarray:
    """(n + 2 NPAD, n): edge-pad then prefilter, as scipy's _prepad_for_spline_filter + spline_filter."""
    pad = np.zeros((n + 2 * NPAD, n)); pad[np.arange(n + 2 * NPAD), np.clip(np.arange(n + 2 * NPAD) - NPAD, 0, n - 1)] = 1
    return spline_filter1d(pad, order=3, axis=0, mode="nearest", output=np.float64)


class Spline3:
    def __init__(self, nz: int, nr: int, device="cuda"):
        self.dev = torch.device(device); self.nz, self.nr = nz, nr
        self.Az = torch.tensor(_axis_matrix(nz), device=self.dev); self.Ar = torch.tensor(_axis_matrix(nr), device=self.dev)

    def coeffs(self, f):
        """f (B, nz, nr) -> spline coefficients (B, nz+24, nr+24)."""
        return self.Az @ f.to(self.Az.dtype) @ self.Ar.T

    @staticmethod
    def _w(t):
        """cubic B-spline weights for offsets -1..2 given the fractional part t."""
        t2, t3 = t * t, t * t * t
        return torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], -1)

    def sample(self, c, zi, ri):
        """c from coeffs; zi, ri (B, ...) fractional row/col indices on the original grid -> values (B, ...)."""
        B = c.shape[0]; sh = zi.shape; zi = zi.reshape(B, -1) + NPAD; ri = ri.reshape(B, -1) + NPAD
        Pz, Pr = c.shape[1], c.shape[2]
        zi = zi.clamp(0, Pz - 1); ri = ri.clamp(0, Pr - 1)
        z0 = torch.floor(zi); r0 = torch.floor(ri); wz = self._w(zi - z0); wr = self._w(ri - r0)
        iz = (z0.long()[..., None] + torch.arange(-1, 3, device=c.device)).clamp(0, Pz - 1)
        ir = (r0.long()[..., None] + torch.arange(-1, 3, device=c.device)).clamp(0, Pr - 1)
        flat = c.reshape(B, -1); idx = (iz[..., :, None] * Pr + ir[..., None, :]).reshape(B, -1)
        v = torch.gather(flat, 1, idx).reshape(B, zi.shape[1], 4, 4)
        return (v * wz[..., :, None] * wr[..., None, :]).sum((-1, -2)).reshape(sh)

    def map(self, f, zi, ri):
        return self.sample(self.coeffs(f), zi, ri)


if __name__ == "__main__":
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(0); sp = Spline3(65, 65)
    yy, xx = np.mgrid[0:65, 0:65] / 64.0
    f = np.stack([np.sin(3 * xx + k) * np.cos(2 * yy - k) + 0.1 * rng.standard_normal((65, 65)) for k in range(4)])
    zi = rng.uniform(-3, 67, (4, 65, 65)); ri = rng.uniform(-3, 67, (4, 65, 65))
    zi[:, :20] = rng.uniform(0, 64, (4, 20, 65)); ri[:, :20] = rng.uniform(0, 64, (4, 20, 65))
    ref = np.stack([map_coordinates(f[k], [zi[k], ri[k]], order=3, mode="nearest") for k in range(4)])
    got = sp.map(torch.tensor(f, device="cuda"), torch.tensor(zi, device="cuda"), torch.tensor(ri, device="cuda")).cpu().numpy()
    err = np.abs(got - ref)
    print(f"[spline] in-range max err {err[:, :20].max():.2e}; incl. out-of-range max err {err.max():.2e}")
