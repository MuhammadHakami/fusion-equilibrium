"""GPU twin of the Fusion scorer's psi-derived quantities (user, Oct 3: no heavy CPU runs). Batched over frames, torch.

Mirrors the vendored functions:
- o_point.find_o_point: interior local max of phi = sign * psi inside the mask eroded by 2 cells, strongest one,
  3-point parabolic sub-pixel refinement.
- lcfs._extract_one_sign: bisection between min(border phi) and phi(axis) for the lowest level whose axis-enclosing
  contour is closed and inside the envelope.
- derive._shape_from_contour and internal_inductance; D_LCFS (symmetric Hausdorff / rgeo).
GPU form of the bisection test: the 4-connected component of {phi > level} that holds the axis must not touch the grid
border, and all its boundary crossing points (the marching-squares vertices) must have a bilinear mask value > 0.5.
Contour = those crossing points, ordered by angle about the axis. This approximates the CPU scorer (skimage contours,
512-point resampling); it is validated against the official per-frame values before use, and the official CPU scorer
stays the arbiter for once-off TEST numbers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GPUScorer:
    def __init__(self, R, Z, mask, sign=-1.0, n_iter=22, device="cuda"):
        self.dev = torch.device(device)
        self.R = torch.tensor(np.asarray(R, np.float64), device=self.dev); self.Z = torch.tensor(np.asarray(Z, np.float64), device=self.dev)
        self.dR, self.dZ = float(R[1] - R[0]), float(Z[1] - Z[0]); self.sign, self.n_iter = sign, n_iter
        m = torch.tensor(np.asarray(mask, bool), device=self.dev); self.mask_f = m.double()
        er = m.clone()
        for _ in range(2):                                                     # 4-connectivity erosion by 2 cells
            e = er.clone(); e[1:] &= er[:-1]; e[:-1] &= er[1:]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]; er = e
        inter = torch.zeros_like(m); inter[1:-1, 1:-1] = True
        self.cand_mask = er & inter
        self._edge_sets(m)
        self.RR, self.ZZ = torch.meshgrid(self.Z, self.R, indexing="ij")[1], torch.meshgrid(self.Z, self.R, indexing="ij")[0]

    # ------------------------------------------------------------------ axis
    def o_point(self, phi):
        B, nz, nr = phi.shape
        p = torch.where(torch.isfinite(phi), phi, torch.full_like(phi, -np.inf))
        nb = F.max_pool2d(F.pad(p[:, None], (1, 1, 1, 1), value=-np.inf), 3, 1)[:, 0]
        lm = (p >= nb) & self.cand_mask
        cand = torch.where(lm, p, torch.full_like(p, -np.inf)).flatten(1)
        has = lm.flatten(1).any(1)
        idx = torch.where(has, cand.argmax(1), p.flatten(1).argmax(1))
        iz, ir = idx // nr, idx % nr
        b = torch.arange(B, device=self.dev)

        def d1(a, c0, c):
            den = a - 2 * c0 + c
            return torch.where(den == 0, torch.zeros_like(den), (0.5 * (a - c) / den).clamp(-1, 1))
        izm, izp = (iz - 1).clamp(0, nz - 1), (iz + 1).clamp(0, nz - 1); irm, irp = (ir - 1).clamp(0, nr - 1), (ir + 1).clamp(0, nr - 1)
        dz = torch.where((iz > 0) & (iz < nz - 1), d1(p[b, izm, ir], p[b, iz, ir], p[b, izp, ir]), torch.zeros(B, device=self.dev, dtype=p.dtype))
        dr = torch.where((ir > 0) & (ir < nr - 1), d1(p[b, iz, irm], p[b, iz, ir], p[b, iz, irp]), torch.zeros(B, device=self.dev, dtype=p.dtype))
        return self.R[ir] + dr * self.dR, self.Z[iz] + dz * self.dZ, iz, ir

    # ------------------------------------------------------------------ LCFS
    def reach(self, phi, iz, ir):
        """Max-min path value from the axis pixel (4-connected): comp(level) = {reach > level} equals the flood fill of
        {phi > level} seeded at the axis, for every level at once (one reconstruction instead of one fill per level)."""
        B = phi.shape[0]; h = torch.full_like(phi, -np.inf); b = torch.arange(B, device=self.dev); h[b, iz, ir] = phi[b, iz, ir]
        for it in range(400):
            g = torch.minimum(torch.maximum(F.max_pool2d(h[:, None], (3, 1), 1, (1, 0)), F.max_pool2d(h[:, None], (1, 3), 1, (0, 1)))[:, 0], phi)
            if it % 8 == 7 and torch.equal(g, h):
                break
            h = g
        return h

    def _crossings(self, phi, comp, level):
        """Marching-squares vertices of the component's boundary: (B, P) row/col fractional coords + valid mask."""
        rows, cols, val = [], [], []
        for axis in (1, 2):
            a = comp.narrow(axis, 0, comp.shape[axis] - 1); b = comp.narrow(axis, 1, comp.shape[axis] - 1)
            pa = phi.narrow(axis, 0, phi.shape[axis] - 1); pb = phi.narrow(axis, 1, phi.shape[axis] - 1)
            e = a ^ b                                                          # one end in the component
            pin = torch.where(a, pa, pb); pout = torch.where(a, pb, pa)
            t = ((pin - level[:, None, None]) / (pin - pout)).clamp(0, 1)
            frac = torch.where(a, t, 1 - t)                                    # position from the lower index
            ii, jj = torch.meshgrid(torch.arange(a.shape[1], device=self.dev), torch.arange(a.shape[2], device=self.dev), indexing="ij")
            z0 = torch.zeros_like(frac); r = ii[None].double() + (frac if axis == 1 else z0); c = jj[None].double() + (frac if axis == 2 else z0)
            rows.append(r.flatten(1)); cols.append(c.flatten(1)); val.append(e.flatten(1))
        return torch.cat(rows, 1), torch.cat(cols, 1), torch.cat(val, 1)

    def _edge_sets(self, m):
        """Grid edges whose crossing can fail the mask test: both ends outside the mask (always fails) and mixed edges
        (fails unless the crossing lies on the inside half). Edges with both ends inside always pass."""
        nz, nr = m.shape; idx = torch.arange(nz * nr, device=self.dev).view(nz, nr)
        A, Bq, side, Z0a, Z0b = [], [], [], [], []
        for a, b in ((idx[:-1], idx[1:]), (idx[:, :-1], idx[:, 1:])):
            a, b = a.flatten(), b.flatten(); ma, mb = m.flatten()[a], m.flatten()[b]
            Z0a.append(a[~ma & ~mb]); Z0b.append(b[~ma & ~mb]); A.append(a[ma ^ mb]); Bq.append(b[ma ^ mb]); side.append(ma[ma ^ mb])
        self.z0a, self.z0b = torch.cat(Z0a), torch.cat(Z0b)
        self.mxa, self.mxb, self.mx_lo_in = torch.cat(A), torch.cat(Bq), torch.cat(side)
        bd = torch.zeros(nz, nr, dtype=torch.bool, device=self.dev); bd[0] = bd[-1] = True; bd[:, 0] = bd[:, -1] = True
        self.border_idx = idx[bd]

    def lcfs(self, phi, iz, ir):
        """Bisection on the level; the per-level test reads h (reach) only at the border and the mask-boundary edges,
        which is the same predicate as filling the component and testing every crossing (old _valid)."""
        B = phi.shape[0]; b = torch.arange(B, device=self.dev); h = self.reach(phi, iz, ir)
        hf, pf = h.flatten(1), phi.flatten(1)
        hbord = hf[:, self.border_idx].max(1).values; hax = h[b, iz, ir]
        za, zb = hf[:, self.z0a], hf[:, self.z0b]; zlo, zhi = torch.minimum(za, zb), torch.maximum(za, zb)
        ha, hb_, pa, pb = hf[:, self.mxa], hf[:, self.mxb], pf[:, self.mxa], pf[:, self.mxb]

        def valid(level):
            L = level[:, None]
            zc = ((zlo <= L) & (L < zhi)).any(1)
            ain = ha > L; e = ain ^ (hb_ > L)
            pin = torch.where(ain, pa, pb); pout = torch.where(ain, pb, pa)
            t = ((pin - L) / (pin - pout)).clamp(0, 1); frac = torch.where(ain, t, 1 - t)
            good = torch.where(self.mx_lo_in[None], frac < 0.5, frac > 0.5)
            return (hbord <= level) & (hax > level) & ~zc & ~(e & ~good).any(1)
        border = torch.cat([phi[:, 0], phi[:, -1], phi[:, :, 0], phi[:, :, -1]], 1)
        lo = border.min(1).values; hi = phi[b, iz, ir] - 1e-9
        ok = valid(hi) & (hi > lo)
        for _ in range(self.n_iter):
            mid = 0.5 * (lo + hi); v = valid(mid)
            hi = torch.where(v, mid, hi); lo = torch.where(v, lo, mid)
        comp = h > hi[:, None, None]
        rows, cols, val = self._crossings(phi, comp, hi)
        return ok, hi, comp, rows, cols, val

    # ------------------------------------------------------------------ all quantities
    def _li(self, phi, poly):
        """Internal inductance of phi on a given LCFS: plasma region comp and the closed ordered contour (row/col)."""
        comp, ro, co, seg, last = poly; b = torch.arange(len(phi), device=self.dev)
        gz, gr = torch.gradient(phi, spacing=(self.Z, self.R), dim=(1, 2))
        Bp = torch.sqrt(gr ** 2 + gz ** 2) / self.RR
        dV = 2 * np.pi * self.RR * self.dR * self.dZ
        num = (Bp ** 2 * dV * comp).sum((1, 2)); V = (dV * comp).sum((1, 2))
        nr_, nc_ = Bp.shape[1:]; r = ro.clamp(0, nr_ - 1); c = co.clamp(0, nc_ - 1)
        r0 = r.floor().long().clamp(max=nr_ - 2); c0 = c.floor().long().clamp(max=nc_ - 2); fr = r - r0; fc = c - c0; bb = b[:, None]
        bpc = (Bp[bb, r0, c0] * (1 - fr) * (1 - fc) + Bp[bb, r0 + 1, c0] * fr * (1 - fc) + Bp[bb, r0, c0 + 1] * (1 - fr) * fc
               + Bp[bb, r0 + 1, c0 + 1] * fr * fc)
        bpn = torch.roll(bpc, -1, 1); bpn[b, last] = bpc[:, 0]
        G = (0.5 * (bpc + bpn) * seg).sum(1); lp = seg.sum(1)
        return (num / V) / (G / lp) ** 2

    def li_with(self, psi, o):
        """li of psi on the LCFS extracted in a previous derive() output o (CPU li warp: contour held fixed)."""
        return self._li(self.sign * psi.to(self.dev, torch.float64), o["poly"])

    @torch.no_grad()
    def derive(self, psi, ref_contours=None, rgeo=None):
        """psi (B, 65, 65) -> dict of (B,) tensors: R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li, ok
        (+ d = Hausdorff / rgeo if reference contours (B, 512, 2) and rgeo (B,) are given)."""
        psi = psi.to(self.dev, torch.float64); phi = self.sign * psi
        Ra, Za, iz, ir = self.o_point(phi)
        ok, lev, comp, rows, cols, val = self.lcfs(phi, iz, ir); ext = (Ra, Za, ok, lev, comp, rows, cols, val, phi)
        keep = torch.argsort((~val).to(torch.int8), dim=1, stable=True)[:, :1024]           # compact to the contour vertices
        rows, cols, val = (torch.gather(x, 1, keep) for x in (rows, cols, val))
        Rp = self.R[0] + cols * self.dR; Zp = self.Z[0] + rows * self.dZ
        inf = torch.full_like(Rp, np.inf)
        Rmax, io = torch.where(val, Rp, -inf).max(1); Rmin, ii = torch.where(val, Rp, inf).min(1)
        zt = torch.where(val, Zp, -inf); Zmax, it = zt.max(1); zb = torch.where(val, Zp, inf); Zmin, ib = zb.min(1)
        b = torch.arange(len(psi), device=self.dev)
        width = Rmax - Rmin; a = 0.5 * width; Rgeo = 0.5 * (Rmax + Rmin)
        kappa = (Zmax - Zmin) / width; tri_top = (Rgeo - Rp[b, it]) / a; tri_bot = (Rgeo - Rp[b, ib]) / a
        ang = torch.atan2(Zp - Za[:, None], Rp - Ra[:, None]); ang = torch.where(val, ang, torch.full_like(ang, 10.0))
        order = ang.argsort(1); n = val.sum(1)
        Ro = torch.gather(Rp, 1, order); Zo = torch.gather(Zp, 1, order); vo = torch.gather(val, 1, order)
        last = (n - 1).clamp(min=0)
        Rn = torch.roll(Ro, -1, 1); Zn = torch.roll(Zo, -1, 1)
        Rn[b, last] = Ro[:, 0]; Zn[b, last] = Zo[:, 0]                          # close the polygon
        cross = Ro * Zn - Rn * Zo
        M = (1 / 6) * ((Ro + Rn) * cross * vo).sum(1); volume = 2 * np.pi * M.abs()
        seg = torch.sqrt((Rn - Ro) ** 2 + (Zn - Zo) ** 2) * vo
        ro = torch.gather(rows, 1, order); co = torch.gather(cols, 1, order); poly = (comp, ro, co, seg, last)
        li = self._li(phi, poly)
        out = dict(R_axis=Ra, Z_axis=Za, kappa=kappa, tri_top=tri_top, tri_bot=tri_bot, volume=volume, li=li, ok=ok & (n >= 8), ext=ext, poly=poly,
                   Rgeo=Rgeo, a=a, R_top=Rp[b, it], Z_top=Zmax, R_bot=Rp[b, ib], Z_bot=Zmin, R_out=Rmax, Z_out=Zp[b, io], R_in=Rmin, Z_in=Zp[b, ii])
        if ref_contours is not None:                                            # resample to 512 by arc length, as the scorer
            cum = torch.cumsum(seg, 1); L = cum[b, last].clamp(min=1e-12)
            tq = torch.arange(512, device=self.dev, dtype=torch.float64)[None] * (L[:, None] / 512)
            k = torch.searchsorted(cum.contiguous(), tq.contiguous()).clamp(max=cum.shape[1] - 1)
            c0 = torch.where(k > 0, torch.gather(cum, 1, (k - 1).clamp(min=0)), torch.zeros_like(tq))
            sl = torch.gather(seg, 1, k).clamp(min=1e-12); w = ((tq - c0) / sl).clamp(0, 1)
            Pr = torch.gather(Ro, 1, k) + w * (torch.gather(Rn, 1, k) - torch.gather(Ro, 1, k))
            Pz = torch.gather(Zo, 1, k) + w * (torch.gather(Zn, 1, k) - torch.gather(Zo, 1, k))
            P = torch.stack([Pr, Pz], -1); Q = ref_contours.to(self.dev, torch.float64)
            D = torch.cdist(P, Q)
            m2, a2 = D.min(2).values.max(1); m1, a1 = D.min(1).values.max(1); d1, d2 = m2, m1
            out["d"] = (torch.maximum(d1, d2) / rgeo.to(self.dev)).clamp(max=1.0)
            out["d_at"] = torch.where((d1 >= d2)[:, None], P[b, a2], Q[b, a1])     # where the Hausdorff max sits
        return out
