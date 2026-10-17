"""Doc 35 §38aa: joint Gauss-Newton shape correction (GPU, inference only).

§38z: the sequential capped warps (axis shift, boundary, kappa, tri) realise only 0.98419 of the heads' 0.98711 consistency;
stronger or repeated single warps make it worse because each one moves the scalars the others fixed (cross-talk). Here one
psi deformation with six parameters theta = (dR_axis, dZ_axis, e, v, d_top, d_bot) is fitted per frame to the six geometric
head scalars at once (R_axis, Z_axis, kappa, tri_top, tri_bot, volume):

    psi'(x) = psi(src(x)),  src = x - (dR, dZ) g_axis(x) + w(x) (M (x - c) - (x - c)) - (d_top g_top(x) + d_bot g_bot(x), 0)
    M = diag(sqrt(s), 1/sqrt(s)) / (1 + v),  s = exp(e)  (kappa x s, size x (1 + v) about the geometric centre c)

Training-free linearised projection onto the head constraints (cf. arXiv 2511.17258): forward finite-difference Jacobian of
the GPU twin's extraction w.r.t. theta, Levenberg-Marquardt steps in the residual space normalised by each scalar's spread
(the consistency R2 weights), a per-frame accept-if-better guard and caps. li stays with the li warp, which runs after this
(its monotone interior remap leaves the LCFS and the axis unchanged).
"""
from __future__ import annotations

import numpy as np
import torch

import fx

GEO = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume"]
EPS = (0.004, 0.004, 0.01, 0.005, 0.004, 0.004)                 # finite-difference steps per parameter
CAP = (0.06, 0.06, 0.15, 0.05, 0.10, 0.10)                      # |theta| caps (axis / tri in m; e, v relative)
_SD = None


def spreads():
    """Per-scalar spread of the training labels (the R2 normalisation), cached."""
    global _SD
    if _SD is None:
        y = np.load(fx.OUT / "boundary_labels_v8.npz")["y_train"][:, :6]; _SD = np.nanstd(y, 0)
    return _SD


class JointWarp:
    def __init__(self, gp, sigma_axis=0.15, sigma_tri=0.15, W=0.10):
        self.gp, self.sa, self.st, self.W = gp, sigma_axis, sigma_tri, W

    def _geometry(self, psi):
        """Fields fixed during the fit, from the input psi: axis / tri bumps, centre offsets and the in-plasma weight."""
        gp = self.gp; o = gp.g.derive(psi); Ra, Za, th_s, r_s, v_s, ok, comp, _, _ = gp._polar(psi, o)
        Rp = Ra[:, None] + r_s * torch.cos(th_s); Zp = Za[:, None] + r_s * torch.sin(th_s)
        inf = torch.full_like(Rp, np.inf)
        Rg = 0.5 * (torch.where(v_s, Rp, -inf).max(1).values + torch.where(v_s, Rp, inf).min(1).values)
        Zg = 0.5 * (torch.where(v_s, Zp, -inf).max(1).values + torch.where(v_s, Zp, inf).min(1).values)
        dx, dy = gp.RR - Rg[:, None, None], gp.ZZ - Zg[:, None, None]; rho = torch.hypot(dx, dy)
        thg = torch.remainder(torch.atan2(Zp - Zg[:, None], Rp - Rg[:, None]), 2 * np.pi); rg = torch.hypot(Rp - Rg[:, None], Zp - Zg[:, None])
        thg = torch.where(v_s, thg, torch.full_like(thg, 99.0)); oo = thg.argsort(1)
        rays = (2 * np.pi * torch.arange(64, device=gp.dev, dtype=psi.dtype) / 64)[None].expand(len(psi), -1)
        rp = gp._interp_periodic(torch.gather(thg, 1, oo), torch.gather(rg, 1, oo), torch.gather(v_s, 1, oo), rays)
        rpt = gp._ray_interp(rp, torch.remainder(torch.atan2(dy, dx), 2 * np.pi).flatten(1)).view_as(rho)
        w = torch.where(comp | (rho <= rpt), torch.ones_like(rho), torch.exp(-((rho - rpt) / self.W) ** 2))
        def bump(Rc, Zc, s):
            return torch.exp(-((gp.RR - Rc[:, None, None]) ** 2 + (gp.ZZ - Zc[:, None, None]) ** 2) / (2 * s ** 2))
        good = ok & o["ok"] & torch.isfinite(Rg) & torch.isfinite(Zg) & torch.isfinite(rpt).flatten(1).all(1)
        return dict(o=o, dx=dx, dy=dy, w=w, ga=bump(Ra, Za, self.sa), gt=bump(o["R_top"], o["Z_top"], self.st),
                    gb=bump(o["R_bot"], o["Z_bot"], self.st), good=good)

    def apply(self, psi, th, G):
        e = lambda k: th[:, k, None, None]                                       # noqa: E731
        sq = torch.exp(0.5 * e(2)); m = 1.0 / (1.0 + e(3))
        Rs = self.gp.RR - e(0) * G["ga"] + G["w"] * (G["dx"] * sq * m - G["dx"]) - e(4) * G["gt"] - e(5) * G["gb"]
        Zs = self.gp.ZZ - e(1) * G["ga"] + G["w"] * (G["dy"] / sq * m - G["dy"])
        return self.gp.sample(psi, Rs, Zs)

    def scalars(self, psi):
        o = self.gp.g.derive(psi); return torch.stack([o[k] for k in GEO], 1), o["ok"]

    def fit(self, psi, target, n_it=3, mu=1e-3, lam=1e-2, mask=None):
        """psi (B,65,65); target (B,6) head scalars in GEO order (NaN = do not fit that scalar). mask (6,) bool: scalars fitted."""
        dev, dt = psi.device, psi.dtype; B = len(psi)
        sd = torch.tensor(spreads(), device=dev, dtype=dt); eps = torch.tensor(EPS, device=dev, dtype=dt)
        cap = torch.tensor(CAP, device=dev, dtype=dt); G = self._geometry(psi)
        use = torch.isfinite(target) & (torch.ones(6, dtype=torch.bool, device=dev) if mask is None else torch.as_tensor(mask, device=dev))[None]
        tgt = torch.where(use, target.to(dt), torch.zeros_like(target, dtype=dt))
        def resid(y):
            return torch.where(use, (tgt - y) / sd, torch.zeros_like(y))
        def cost(y, ok):
            c = (resid(y) ** 2).sum(1); return torch.where(ok & torch.isfinite(c), c, torch.full_like(c, np.inf))
        th = torch.zeros(B, 6, device=dev, dtype=dt); y0, ok0 = self.scalars(psi); c0 = cost(y0, ok0); y, cur = y0, c0
        for _ in range(n_it):
            r = resid(y); J = torch.zeros(B, 6, 6, device=dev, dtype=dt)
            for k in range(6):
                tk = th.clone(); tk[:, k] += eps[k]; yk, okk = self.scalars(self.apply(psi, tk, G))
                col = (resid(yk) - r) / eps[k]                               # d resid / d theta_k (= -J of the scalars)
                J[:, :, k] = torch.where(okk[:, None] & torch.isfinite(col), col, torch.zeros_like(col))
            A = J.transpose(1, 2) @ J; diag = torch.diagonal(A, dim1=1, dim2=2)
            A = A + torch.diag_embed(mu * diag + lam / cap ** 2)
            step = -torch.linalg.solve(A, (J.transpose(1, 2) @ r[..., None]))[..., 0]
            step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
            for _h in range(3):                                               # backtrack: accept the first improving step
                tn = torch.maximum(torch.minimum(th + step, cap), -cap); yn, okn = self.scalars(self.apply(psi, tn, G)); cn = cost(yn, okn)
                better = cn < cur
                th = torch.where(better[:, None], tn, th); y = torch.where(better[:, None], yn, y); cur = torch.where(better, cn, cur)
                step = torch.where(better[:, None], torch.zeros_like(step), 0.5 * step)
        keep = G["good"] & ok0 & (cur < c0)
        self.diag = dict(c0=c0, c1=cur, keep=keep, theta=th)
        return torch.where(keep[:, None, None], self.apply(psi, th, G), psi)


def tri_snap(gp, psi, tri_t, shifts=(-1.0, -0.5, 0.5, 1.0), sigma=0.15, dk=0.003, lam=0.002):
    """§38aa: tri moves in grid-column steps (the official top / bottom point sits within +-0.3 cell of a column), so pick,
    per frame, the lateral shift of the top (bottom) region - candidates in units of dR, 0 included - whose extracted tri is
    nearest the head; kappa guard dk; lam * |shift in cells| breaks ties toward no move. tri_t (B, 2) head tri_top / tri_bot."""
    o0 = gp.g.derive(psi); tri_t = tri_t.to(psi.dtype); dev = psi.device
    def bump(Rc, Zc):
        return torch.exp(-((gp.RR - Rc[:, None, None]) ** 2 + (gp.ZZ - Zc[:, None, None]) ** 2) / (2 * sigma ** 2))
    gt, gb = bump(o0["R_top"], o0["Z_top"]), bump(o0["R_bot"], o0["Z_bot"])
    def apply(dt, db):
        return gp.sample(psi, gp.RR - dt[:, None, None] * gt - db[:, None, None] * gb, gp.ZZ.expand_as(psi))
    cand = torch.tensor((0.0,) + tuple(shifts), device=dev, dtype=psi.dtype)
    et, eb = [], []
    for s in cand:
        if s == 0:
            o = o0
        else:
            v = torch.full((len(psi),), float(s) * gp.dR, device=dev, dtype=psi.dtype); o = gp.g.derive(apply(v, v))
        bad = ~(o["ok"] & ((o["kappa"] - o0["kappa"]).abs() <= dk))
        pen = lam * abs(float(s))
        e_t = (o["tri_top"] - tri_t[:, 0]).abs() + pen; e_b = (o["tri_bot"] - tri_t[:, 1]).abs() + pen
        et.append(torch.where(bad | ~torch.isfinite(e_t), torch.full_like(e_t, np.inf), e_t))
        eb.append(torch.where(bad | ~torch.isfinite(e_b), torch.full_like(e_b, np.inf), e_b))
    it, ib = torch.stack(et, 1).argmin(1), torch.stack(eb, 1).argmin(1)
    dt, db = cand[it] * gp.dR, cand[ib] * gp.dR; out = apply(dt, db); o = gp.g.derive(out)
    e0 = (o0["tri_top"] - tri_t[:, 0]).abs() + (o0["tri_bot"] - tri_t[:, 1]).abs()
    e1 = (o["tri_top"] - tri_t[:, 0]).abs() + (o["tri_bot"] - tri_t[:, 1]).abs()
    good = o0["ok"] & o["ok"] & torch.isfinite(e1) & (e1 < e0) & ((o["kappa"] - o0["kappa"]).abs() <= dk) & torch.isfinite(tri_t).all(1)
    gp.tsnap_diag = dict(moved=good & ((dt != 0) | (db != 0)), it=it, ib=ib)
    return torch.where(good[:, None, None], out, psi)
