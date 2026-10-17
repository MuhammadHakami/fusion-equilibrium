"""Batched GPU versions of the Fusion post-processing (doc 35 §37n axis shift, §37q boundary warp, §37w elongation,
§37o li warp), using gpu_scorer.GPUScorer for every extraction and gpu_spline.Spline3 (exact scipy map_coordinates
order 3, mode "nearest") for every resampling. Ray radii are exact ray-polygon intersections, as boundary_labels.ray_radii.
Same formulas and caps as axis_shift.py / boundary_warp.py / kappa_warp.py / li_warp.py; validated against them.
"""
from __future__ import annotations

import numpy as np
import torch

from gpu_scorer import GPUScorer
from gpu_spline import Spline3


class GPUPost:
    def __init__(self, R, Z, mask, sign=-1.0, device="cuda"):
        self.g = GPUScorer(R, Z, mask, sign, device=device); self.dev = self.g.dev; self.sign = sign
        self.R0, self.Z0, self.dR, self.dZ = float(R[0]), float(Z[0]), self.g.dR, self.g.dZ
        self.RR, self.ZZ = self.g.RR, self.g.ZZ; self.nz, self.nr = len(Z), len(R); self.sp = Spline3(len(Z), len(R), device)

    def sample(self, psi, Rs, Zs):
        """psi (B,nz,nr) sampled at physical (Rs, Zs) (B,nz,nr): cubic spline, nearest extension (= the CPU warps)."""
        return self.sp.map(psi, (Zs - self.Z0) / self.dZ, (Rs - self.R0) / self.dR)

    def axis(self, psi):
        Ra, Za, _, _ = self.g.o_point(self.sign * psi); return torch.stack([Ra, Za], 1)

    # §37n: shift the core so the scorer's O-point lands on the head axis
    def axis_shift(self, psi, head, sigma=0.15, n_it=4, cap_cells=2.0):
        x0 = self.axis(psi); cap = torch.tensor([cap_cells * self.dR, cap_cells * self.dZ], device=self.dev, dtype=psi.dtype)
        g = torch.exp(-((self.RR - x0[:, :1, None]) ** 2 + (self.ZZ - x0[:, 1:, None]) ** 2) / (2 * sigma ** 2))
        delta = torch.zeros_like(x0); out = psi
        for _ in range(n_it):
            delta = torch.maximum(torch.minimum(delta + (head - self.axis(out)), cap), -cap)
            out = self.sample(psi, self.RR - delta[:, :1, None] * g, self.ZZ - delta[:, 1:, None] * g)
        return out

    def _polar(self, psi, o=None):
        """Extraction + ordered contour; returns axis (B,2), contour angles/radii about it, component and level.
        o: a derive() output for the same psi, whose extraction is reused."""
        if o is None:
            phi = self.sign * psi; Ra, Za, iz, ir = self.g.o_point(phi)
            ok, lev, comp, rows, cols, val = self.g.lcfs(phi, iz, ir)
        else:
            Ra, Za, ok, lev, comp, rows, cols, val, phi = o["ext"]
        Rp = self.R0 + cols * self.dR; Zp = self.Z0 + rows * self.dZ
        th = torch.remainder(torch.atan2(Zp - Za[:, None], Rp - Ra[:, None]), 2 * np.pi)
        rad = torch.hypot(Rp - Ra[:, None], Zp - Za[:, None])
        th = torch.where(val, th, torch.full_like(th, 99.0)); o = th.argsort(1)
        return Ra, Za, torch.gather(th, 1, o), torch.gather(rad, 1, o), torch.gather(val, 1, o), ok, comp, lev, phi

    @staticmethod
    def _interp_periodic(th_s, r_s, v_s, q):
        """Radius at query angles q (B,Q) of the polygon through the angle-sorted vertices (theta, r) (invalid entries at
        the end): exact ray-segment intersection with the bracketing segment (= ray_radii for a star-shaped contour)."""
        n = v_s.sum(1).clamp(min=2); B = th_s.shape[0]; b = torch.arange(B, device=th_s.device)[:, None]
        last = (n - 1)[:, None]
        thx = torch.cat([th_s[b, last] - 2 * np.pi, th_s, th_s[:, :1] + 2 * np.pi], 1)
        rx = torch.cat([r_s[b, last], r_s, r_s[:, :1]], 1)
        idx = torch.arange(thx.shape[1], device=th_s.device)[None]
        valid = (idx == 0) | ((idx >= 1) & (idx <= n[:, None])) | (idx == n[:, None] + 1)
        thx = torch.where(valid, thx, torch.full_like(thx, 1e9)); thx[b[:, 0], (n + 1)] = th_s[b[:, 0], 0] + 2 * np.pi
        o = thx.argsort(1); thx = torch.gather(thx, 1, o); rx = torch.gather(rx, 1, o)
        k = torch.searchsorted(thx.contiguous(), q.contiguous()).clamp(1, thx.shape[1] - 1)
        t0, t1 = torch.gather(thx, 1, k - 1), torch.gather(thx, 1, k); r0, r1 = torch.gather(rx, 1, k - 1), torch.gather(rx, 1, k)
        den = r0 * torch.sin(q - t0) + r1 * torch.sin(t1 - q)
        return torch.where(den.abs() > 1e-15, r0 * r1 * torch.sin(t1 - t0) / den, r0)

    @staticmethod
    def _ray_interp(r, th):
        """Periodic linear interpolation of ray radii r (B, n) at equally spaced angles 2 pi k / n, at angles th (B, Q)."""
        n = r.shape[1]; ang = (2 * np.pi * torch.arange(n + 1, device=r.device, dtype=r.dtype) / n)[None].expand(len(r), -1)
        rx = torch.cat([r, r[:, :1]], 1); k = torch.searchsorted(ang.contiguous(), th.contiguous()).clamp(1, n)
        a0, a1 = torch.gather(ang, 1, k - 1), torch.gather(ang, 1, k)
        return torch.gather(rx, 1, k - 1) + (torch.gather(rx, 1, k) - torch.gather(rx, 1, k - 1)) * (th - a0) / (a1 - a0)

    # §37q: radial warp so the extracted LCFS meets the head radii r_t (B, n_ang) about the axis
    def boundary_warp(self, psi, r_t, W=0.10, cap=0.05):
        Ra, Za, th_s, r_s, v_s, ok, _, _, _ = self._polar(psi)
        n_ang = r_t.shape[1]; ang = (2 * np.pi * torch.arange(n_ang, device=self.dev, dtype=psi.dtype) / n_ang)[None].expand(len(psi), -1)
        r_p = self._interp_periodic(th_s, r_s, v_s, ang)
        r_t = r_p + (r_t.to(psi.dtype) - r_p).clamp(-cap, cap)
        dx, dy = self.RR - Ra[:, None, None], self.ZZ - Za[:, None, None]; rho = torch.hypot(dx, dy)
        th = torch.remainder(torch.atan2(dy, dx), 2 * np.pi).flatten(1)
        rp, rt = self._ray_interp(r_p, th).view_as(rho), self._ray_interp(r_t, th).view_as(rho)
        src = torch.where(rho <= rt, rho * rp / rt, rho - (rt - rp) * torch.exp(-((rho - rt) / W) ** 2))
        f = torch.where(rho > 0, src / rho.clamp(min=1e-12), torch.ones_like(rho))
        out = self.sample(psi, Ra[:, None, None] + dx * f, Za[:, None, None] + dy * f)
        good = ok & torch.isfinite(r_p).all(1) & torch.isfinite(r_t).all(1)
        return torch.where(good[:, None, None], out, psi)

    def kappa_of(self, psi):
        return self.g.derive(psi)["kappa"]

    # §37w: area-preserving elongation to the kappa head (2 secant steps, guard)
    def kappa_warp(self, psi, kt, W=0.10, cap=0.10):
        o = self.g.derive(psi); k0 = o["kappa"]; Ra, Za, th_s, r_s, v_s, ok, comp, _, _ = self._polar(psi, o)
        # geometric centre from the extracted contour extremes (twin)
        Rp = Ra[:, None] + r_s * torch.cos(th_s); Zp = Za[:, None] + r_s * torch.sin(th_s)
        inf = torch.full_like(Rp, np.inf)
        Rg = 0.5 * (torch.where(v_s, Rp, -inf).max(1).values + torch.where(v_s, Rp, inf).min(1).values)
        Zg = 0.5 * (torch.where(v_s, Zp, -inf).max(1).values + torch.where(v_s, Zp, inf).min(1).values)
        dx, dy = self.RR - Rg[:, None, None], self.ZZ - Zg[:, None, None]; rho = torch.hypot(dx, dy)
        thg = torch.remainder(torch.atan2(Zp - Zg[:, None], Rp - Rg[:, None]), 2 * np.pi); rg = torch.hypot(Rp - Rg[:, None], Zp - Zg[:, None])
        thg = torch.where(v_s, thg, torch.full_like(thg, 99.0)); oo = thg.argsort(1)
        rays = (2 * np.pi * torch.arange(64, device=self.dev, dtype=psi.dtype) / 64)[None].expand(len(psi), -1)
        rp = self._interp_periodic(torch.gather(thg, 1, oo), torch.gather(rg, 1, oo), torch.gather(v_s, 1, oo), rays)
        rpt = self._ray_interp(rp, torch.remainder(torch.atan2(dy, dx), 2 * np.pi).flatten(1)).view_as(rho)
        w = torch.where(comp | (rho <= rpt), torch.ones_like(rho), torch.exp(-((rho - rpt) / W) ** 2))
        kt = kt.to(psi.dtype)
        def apply(s):
            s = s[:, None, None]
            return self.sample(psi, self.RR + ((Rg[:, None, None] + dx * torch.sqrt(s)) - self.RR) * w,
                               self.ZZ + ((Zg[:, None, None] + dy / torch.sqrt(s)) - self.ZZ) * w)
        s = (kt / k0).clamp(1 - cap, 1 + cap); out = apply(s)
        for _ in range(2):                                                  # CPU: stop once |k1 - kt| < 1e-5 or k1 is NaN
            k1 = self.kappa_of(out); stop = ~torch.isfinite(k1) | ((k1 - kt).abs() < 1e-5)
            s = torch.where(stop, s, (s * kt / k1).clamp(1 - cap, 1 + cap)); out = apply(s)
        kf = self.kappa_of(out)
        good = ok & torch.isfinite(kf) & torch.isfinite(k0) & ((kf - kt).abs() <= (k0 - kt).abs())
        return torch.where(good[:, None, None], out, psi)

    # §38h: lateral shift of the top / bottom boundary regions toward the tri heads (Gaussian bumps at the extreme points)
    def xp_warp(self, psi, tri_t, sigma=0.15, cap=0.05, n_it=3, dk=0.003, w=1.0, sigma_z=None):
        """tri_t (B, 2) head tri_top / tri_bot. psi(R, Z) -> psi(R - dt g_t - db g_b, Z); the bumps sit at the current top
        and bottom points (fixed during the iteration); targets R* = Rgeo - tri * a; guard on the tri error and kappa.
        §38n: w blends the target, R* = R0 + w (R* - R0); sigma_z gives the bump its own vertical width (defaults: as before)."""
        o0 = self.g.derive(psi); tri_t = tri_t.to(psi.dtype)
        Rt, Rb = o0["Rgeo"] - tri_t[:, 0] * o0["a"], o0["Rgeo"] - tri_t[:, 1] * o0["a"]
        if w != 1.0:
            Rt, Rb = o0["R_top"] + w * (Rt - o0["R_top"]), o0["R_bot"] + w * (Rb - o0["R_bot"])
        def bump(Rc, Zc):
            if sigma_z is not None:
                return torch.exp(-(self.RR - Rc[:, None, None]) ** 2 / (2 * sigma ** 2) - (self.ZZ - Zc[:, None, None]) ** 2 / (2 * sigma_z ** 2))
            return torch.exp(-((self.RR - Rc[:, None, None]) ** 2 + (self.ZZ - Zc[:, None, None]) ** 2) / (2 * sigma ** 2))
        gt, gb = bump(o0["R_top"], o0["Z_top"]), bump(o0["R_bot"], o0["Z_bot"])
        def apply(dt, db):
            return self.sample(psi, self.RR - dt[:, None, None] * gt - db[:, None, None] * gb, self.ZZ.expand_as(psi))
        dt = (Rt - o0["R_top"]).clamp(-cap, cap); db = (Rb - o0["R_bot"]).clamp(-cap, cap); out = apply(dt, db)
        for _ in range(n_it - 1):
            o = self.g.derive(out)
            dt = torch.where(torch.isfinite(o["R_top"]), (dt + Rt - o["R_top"]).clamp(-cap, cap), dt)
            db = torch.where(torch.isfinite(o["R_bot"]), (db + Rb - o["R_bot"]).clamp(-cap, cap), db)
            out = apply(dt, db)
        o = self.g.derive(out)
        e0 = (o0["tri_top"] - tri_t[:, 0]).abs() + (o0["tri_bot"] - tri_t[:, 1]).abs()
        e1 = (o["tri_top"] - tri_t[:, 0]).abs() + (o["tri_bot"] - tri_t[:, 1]).abs()
        good = o0["ok"] & o["ok"] & torch.isfinite(e1) & (e1 < e0) & ((o["kappa"] - o0["kappa"]).abs() <= dk) & torch.isfinite(tri_t).all(1)
        return torch.where(good[:, None, None], out, psi)

    # §45t: tri_top / tri_bot are column-quantised (R_at_Zmax / R_at_Zmin are grid columns) but Rgeo = (Rmax + Rmin) / 2 is continuous
    def tri_shift(self, psi, tri_t, sigma=0.10, sz_out=0.25, sz_in=0.40, cap=0.02, n_it=3, dk=0.002, dv=0.003):
        """tri_t (B, 2) head tri_top / tri_bot. Moves the outboard and inboard boundary by the same dl (Gaussian bumps at the R
        extremes; a and kappa unchanged) so Rgeo absorbs the mean tri residual m = mean(tri - head): target Rgeo - m a.
        Guard per frame: squared tri residual smaller, |dkappa| <= dk, |dV / V| <= dv."""
        o0 = self.g.derive(psi); tri_t = tri_t.to(psi.dtype)
        m0 = 0.5 * ((o0["tri_top"] - tri_t[:, 0]) + (o0["tri_bot"] - tri_t[:, 1])); Rg = o0["Rgeo"] - m0 * o0["a"]
        def bump(Rc, Zc, sz):
            Rc, Zc = torch.nan_to_num(Rc), torch.nan_to_num(Zc)
            return torch.exp(-(self.RR - Rc[:, None, None]) ** 2 / (2 * sigma ** 2) - (self.ZZ - Zc[:, None, None]) ** 2 / (2 * sz ** 2))
        g = bump(o0["R_out"], o0["Z_out"], sz_out) + bump(o0["R_in"], o0["Z_in"], sz_in)
        def apply(dl):
            return self.sample(psi, self.RR - dl[:, None, None] * g, self.ZZ.expand_as(psi))
        dl = torch.nan_to_num(Rg - o0["Rgeo"]).clamp(-cap, cap); out = apply(dl)
        for _ in range(n_it - 1):
            o = self.g.derive(out)
            dl = torch.where(torch.isfinite(o["Rgeo"]), (dl + Rg - o["Rgeo"]).clamp(-cap, cap), dl); out = apply(dl)
        o = self.g.derive(out)
        e0 = (o0["tri_top"] - tri_t[:, 0]) ** 2 + (o0["tri_bot"] - tri_t[:, 1]) ** 2
        e1 = (o["tri_top"] - tri_t[:, 0]) ** 2 + (o["tri_bot"] - tri_t[:, 1]) ** 2
        good = (o0["ok"] & o["ok"] & torch.isfinite(e1) & (e1 < e0) & ((o["kappa"] - o0["kappa"]).abs() <= dk)
                & ((o["volume"] - o0["volume"]).abs() <= dv * o0["volume"].abs()) & torch.isfinite(tri_t).all(1))
        self.trs_diag = dict(dl=dl, good=good, m0=m0)
        return torch.where(good[:, None, None], out, psi)

    # §37o: interior profile remap toward the li target (head li, full=True)
    def li_warp(self, psi, lit, cap=0.5, cap_lo=None, guard=False):
        o = self.g.derive(psi); Ra, Za, th_s, r_s, v_s, ok, comp, lev, phi = self._polar(psi, o)
        phi_ax = torch.where(comp, phi, torch.full_like(phi, -np.inf)).flatten(1).max(1).values
        x = (phi_ax[:, None, None] - phi) / (phi_ax - lev).clamp(min=1e-12)[:, None, None]
        sel = comp & (x >= 0) & (x <= 1)
        def apply(beta):
            ph = torch.where(sel, phi_ax[:, None, None] - (phi_ax - lev)[:, None, None] * (x + beta[:, None, None] * x * (1 - x) ** 2), phi)
            return self.sign * ph
        lit = lit.to(psi.dtype); li0 = o["li"]
        b1 = torch.full_like(li0, 0.1); li1 = self.g.li_with(apply(b1), o)
        slope = (li1 - li0) / 0.1
        lo = -cap if cap_lo is None else cap_lo                             # monotone profile map for beta in (-1, 3)
        beta = ((lit - li0) / slope).clamp(lo, cap); li2 = self.g.li_with(apply(beta), o)
        upd = torch.isfinite(li2) & (beta != b1) & (li2 != li1)
        beta = torch.where(upd, (beta + (lit - li2) * (beta - b1) / (li2 - li1)).clamp(lo, cap), beta)
        good = ok & torch.isfinite(li0) & torch.isfinite(slope) & (slope != 0) & torch.isfinite(lit)
        if guard:                                                            # §44f: accept-if-better per frame (current-quench frames:
            li3 = self.g.li_with(apply(torch.where(good, beta, torch.zeros_like(beta))), o)   # the warp overshot li 5.99 vs head 2.64)
            good = good & torch.isfinite(li3) & ((li3 - lit).abs() < (li0 - lit).abs())
        self.li_diag = dict(beta=beta, good=good, li0=li0, li1=li1, li2=li2, lev=lev)
        return torch.where(good[:, None, None], apply(torch.where(good, beta, torch.zeros_like(beta))), psi)

    def full(self, psi, head_axis, r_t, kappa_t, li_t):
        a = self.axis_shift(psi, head_axis); a = self.boundary_warp(a, r_t); a = self.kappa_warp(a, kappa_t)
        return self.li_warp(a, li_t)
