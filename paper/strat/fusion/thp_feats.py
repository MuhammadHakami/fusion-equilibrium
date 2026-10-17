"""Doc 35 §41r: Thomson profiles in flux coordinates of the PREDICTED psi (inputs only; "thp" block).

thz_feats.py puts Te / ne on fixed physical Z / R points; the same Z is a different flux surface in a different shape, so a
head has to learn the geometry mapping itself. Here every Thomson channel (vertical core chord R = 1.94 m, midplane
tangential chords) is placed on psi_N of the frame's predicted flux map (psi trunk ensemble -> GPU twin of the scorer's
axis / LCFS search), then per frame:
  - Te, ne kernel-regressed on psi_N (Gaussian, sigma 0.04; softmax weights -> nearest channel when far) at
    G = 0, 0.1, .., 1.0, 1.05, and the distance from each G to the nearest channel (coverage);
  - volume averages over the predicted plasma (profile linearly interpolated to every grid cell inside the LCFS):
    <ne>_V, density-weighted <Te>, <ne Te>_V and the peaking factors Te(0) / <Te>, ne(0) / <ne>, p(0) / <p>;
  - flags: LCFS found, number of channels inside psi_N < 1, min psi_N seen, psi_N of the highest core channel with Te > 100 eV.
Intensive quantities only - no raw geometry (volume, minor radius) of the trunk psi, which is in-sample on training rows.
Honest screen: psi = psiF (TRAIN-only trunks) -> VALID / TEST rows are out-of-sample, as the public fold is for psiFA.

    .venv/bin/python paper/strat/fusion/thp_feats.py raw [--public]                 (CPU, 8 workers: channel positions + Te / ne)
    .venv/bin/python paper/strat/fusion/thp_feats.py feat --psi psiF [--public]     (GPU) -> runs/strat/fusion/thp_{all|public}_{psi}.npy
"""
from __future__ import annotations

import argparse
import glob
import sys
import time

import numpy as np
import pandas as pd

import fx  # noqa: I001
import fold as FD  # noqa: E402,I001

SRC = fx.ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge" / "data"
PROC = fx.ROOT / "data" / "sophelio_proc"
NCH = 64
G = np.r_[np.linspace(0.0, 1.0, 11), 1.05]
SIG = 0.04
NF = 3 * len(G) + 10


def nearest(tt, tq):
    i = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
    return np.where(np.abs(tt[i - 1] - tq) < np.abs(tt[i] - tq), i - 1, i)


def shot_raw(path, tq):
    """Channel positions (NCH, 2) [R, Z] and per-frame Te [keV] / ne [1e19 m^-3] (T, NCH); NaN where invalid / absent."""
    r = pd.read_parquet(path).iloc[0]; T = len(tq)
    pos = np.full((NCH, 2), np.nan, np.float32); Te = np.full((T, NCH), np.nan, np.float32); ne = np.full((T, NCH), np.nan, np.float32)
    nm = np.asarray(r["thomson_chord_name"]).astype(str); z = np.asarray(r["thomson_chord_Z"], float); R = np.asarray(r["thomson_chord_R"], float)
    k = 0
    for key, sel in (("core", np.char.startswith(nm, "TS_core")), ("edge", np.char.startswith(nm, "TS_tangential"))):
        tt = r[f"thomson_{key}_times"]
        if tt is None or len(tt) < 2 or not sel.any():
            continue
        i = nearest(np.asarray(tt, float), tq)
        te = np.stack([np.asarray(x, float) for x in r[f"thomson_{key}_Te"]])[i]; nn = np.stack([np.asarray(x, float) for x in r[f"thomson_{key}_ne"]])[i]
        n = min(te.shape[1], int(sel.sum()), NCH - k)
        pos[k:k + n, 0], pos[k:k + n, 1] = R[sel][:n], z[sel][:n]
        good = np.isfinite(te[:, :n]) & np.isfinite(nn[:, :n]) & (te[:, :n] > 0) & (nn[:, :n] > 0)
        Te[:, k:k + n] = np.where(good, te[:, :n] / 1e3, np.nan); ne[:, k:k + n] = np.where(good, nn[:, :n] / 1e19, np.nan)
        k += n
    return pos, Te, ne


def _work(args):
    tag, jobs = args
    Te = np.load(PROC / f"thp_Te_{tag}.npy", mmap_mode="r+"); ne = np.load(PROC / f"thp_ne_{tag}.npy", mmap_mode="r+"); P = {}
    for s, a, b, path, tq in jobs:
        try:
            pos, te, nn = shot_raw(path, tq)
        except Exception as e:                                              # a shot without usable Thomson stays NaN
            print(f"[thp] {path.name}: {type(e).__name__} {e}", flush=True); continue
        Te[a:b], ne[a:b], P[s] = te, nn, pos
    Te.flush(); ne.flush(); return P


def jobs_of(public: bool):
    if public:
        import pyarrow.parquet as pq
        raw = sorted(glob.glob(str(SRC / FD.FOLD / "*.parquet"))); out, a = [], 0
        for s, r in enumerate(raw):
            t = np.asarray(pq.read_table(r, columns=["efit_times"]).column(0)[0].as_py(), float); out.append((s, a, a + len(t), fx.pathlib.Path(r), t)); a += len(t)
        return out, a, len(raw)
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    d = sota.Data(); names = d.meta["names"]; it = d.fn.index("t"); st = np.r_[0, np.nonzero(np.diff(d.shot))[0] + 1, len(d.shot)]
    return [(int(d.shot[a]), int(a), int(b), SRC / "diii_d_train" / f"{names[int(d.shot[a])]}.parquet", np.asarray(d.X[a:b, it], float))
            for a, b in zip(st[:-1], st[1:])], len(d.shot), len(names)


def raw(public: bool, workers=8) -> None:
    from multiprocessing import Pool
    tag = FD.TAG if public else "all"; jobs, N, S = jobs_of(public)
    for q in ("Te", "ne"):
        np.lib.format.open_memmap(PROC / f"thp_{q}_{tag}.npy", mode="w+", dtype=np.float32, shape=(N, NCH))[:] = np.nan
    pos = np.full((S, NCH, 2), np.nan, np.float32); t = time.time()
    with Pool(workers) as p:
        for P in p.map(_work, [(tag, jobs[i::workers]) for i in range(workers)]):
            for s, v in P.items():
                pos[s] = v
    np.save(PROC / f"thp_pos_{tag}.npy", pos)
    Te = np.load(PROC / f"thp_Te_{tag}.npy", mmap_mode="r")
    print(f"THP_RAW_DONE {tag}: {N} frames, {S} shots, frames with any channel {np.isfinite(Te).any(1).mean():.4f} | {time.time() - t:.0f}s", flush=True)


def feats_gpu(sc, psi, pos, Te, ne):
    """psi (B, 65, 65) float64 cuda; pos (B, NCH, 2); Te / ne (B, NCH) with NaN = invalid -> (B, NF) float32."""
    import torch
    import torch.nn.functional as F
    dev = psi.device; B = len(psi); phi = sc.sign * psi
    Ra, Za, iz, ir = sc.o_point(phi); ok, lev, comp, *_ = sc.lcfs(phi, iz, ir)
    b = torch.arange(B, device=dev); pax = phi[b, iz, ir]; den = (pax - lev).clamp(min=1e-9)
    psin = ((pax[:, None, None] - phi) / den[:, None, None])                                           # (B, 65, 65)
    R0, R1, Z0, Z1 = sc.R[0], sc.R[-1], sc.Z[0], sc.Z[-1]
    gx = (2 * (pos[..., 0] - R0) / (R1 - R0) - 1); gy = (2 * (pos[..., 1] - Z0) / (Z1 - Z0) - 1)
    grid = torch.stack([gx, gy], -1)[:, :, None, :]                                                    # (B, NCH, 1, 2)
    pc = F.grid_sample(psin[:, None], torch.nan_to_num(grid, nan=-2.0), align_corners=True, padding_mode="border")[:, 0, :, 0]
    valid = torch.isfinite(Te) & torch.isfinite(ne) & torch.isfinite(pos[..., 0])
    Gt = torch.tensor(G, device=dev, dtype=psi.dtype)
    d = pc[:, None, :] - Gt[None, :, None]                                                            # (B, nG, NCH)
    logit = torch.where(valid[:, None, :], -(d ** 2) / (2 * SIG ** 2), torch.full_like(d, -np.inf))
    w = torch.softmax(logit, -1); w = torch.nan_to_num(w, nan=0.0)
    te_g = (w * torch.nan_to_num(Te)[:, None, :]).sum(-1); ne_g = (w * torch.nan_to_num(ne)[:, None, :]).sum(-1)
    cov = torch.where(valid[:, None, :], d.abs(), torch.full_like(d, 9.0)).min(-1).values.clamp(max=0.5)
    anyv = valid.any(1)
    x = (psin.clamp(0, 1) * 10); i0 = x.floor().long().clamp(0, 9); fr = x - i0                         # cells: linear interp on G[0..10]
    def at(prof):
        p = prof[:, :11]; flat = p[:, None, :].expand(-1, 65 * 65, -1)
        lo = torch.gather(flat, 2, i0.view(B, -1, 1))[..., 0]; hi = torch.gather(flat, 2, (i0 + 1).view(B, -1, 1))[..., 0]
        return (lo * (1 - fr.view(B, -1)) + hi * fr.view(B, -1)).view(B, 65, 65)
    dV = (2 * np.pi * sc.RR * sc.dR * sc.dZ)[None] * comp
    V = dV.sum((1, 2)).clamp(min=1e-9); tc, nc = at(te_g), at(ne_g)
    nV = (nc * dV).sum((1, 2)) / V; pV = (nc * tc * dV).sum((1, 2)) / V; tN = pV / nV.clamp(min=1e-9)
    inside = valid & (pc < 1.0)
    core_top = torch.where(valid & (Te > 0.1) & (torch.abs(pos[..., 0] - 1.94) < 1e-3) & (pos[..., 1] > 0.0), pc, torch.full_like(pc, -1.0)).max(1).values
    good = (ok & anyv).to(psi.dtype)[:, None]
    out = torch.cat([te_g, ne_g, cov,
                     torch.stack([nV, tN, pV, te_g[:, 0] / tN.clamp(min=1e-6), ne_g[:, 0] / nV.clamp(min=1e-6), te_g[:, 0] * ne_g[:, 0] / pV.clamp(min=1e-6),
                                  ok.to(psi.dtype), inside.sum(1).to(psi.dtype) / 54.0,
                                  torch.where(valid, pc, torch.full_like(pc, 9.0)).min(1).values.clamp(max=2.0), core_top.clamp(-1, 2)], 1)], 1)
    out = out * good
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).float().cpu().numpy()


def feat(psi_name: str, public: bool, chunk=12000) -> None:
    import torch
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    import gpu_build as GB
    import axis_shift as AX
    from gpu_scorer import GPUScorer
    d = sota.Data()
    if public:
        import submit_public as SP
        d = SP.PublicData(d.meta)
    tag = FD.TAG if public else "all"
    Te = np.load(PROC / f"thp_Te_{tag}.npy", mmap_mode="r"); ne = np.load(PROC / f"thp_ne_{tag}.npy", mmap_mode="r"); pos = np.load(PROC / f"thp_pos_{tag}.npy")
    assert len(Te) == len(d.X), (len(Te), len(d.X))
    views = []
    for p in GB.PSI[psi_name]:
        cf = torch.load(GB.SOTA / p, map_location="cpu", weights_only=False)["cfg"]
        dd = GB.fixed(d, GB.mf_name(p, {})) if cf.get("inputs") == "magfix" else d
        views.append((str(GB.SOTA / p), GB.canonical(dd) if cf.get("canon") else dd))
    comps = torch.tensor(d.meta["comps"][:ES.KMAX].numpy().astype(np.float64), device="cuda"); mean = torch.tensor(d.meta["mean"].numpy().astype(np.float64), device="cuda")
    R, Z, mask = AX.grid(); sc = GPUScorer(R, Z, mask)
    out = np.lib.format.open_memmap(fx.OUT / f"thp_{tag}_{psi_name}.npy", mode="w+", dtype=np.float32, shape=(len(d.X), NF))
    shots = np.unique(d.shot); t = time.time(); groups, cur, n = [], [], 0
    for s in shots:
        cur.append(int(s)); n += int((d.shot == s).sum()) if public else 0
        if (public and n > chunk) or (not public and len(cur) >= 60):
            groups.append(cur); cur, n = [], 0
    if cur:
        groups.append(cur)
    fold = {}                                                                 # psiOOF: TRAIN shot -> the trunk that left its fold out
    if psi_name == "psiOOF" and not public:
        import head_boundary as HB
        lab = np.load(fx.OUT / "boundary_labels_v8.npz")
        for k in range(5):
            for s_ in HB.cv_shots(d, lab["rows_train"], lab["y_train"].astype(np.float64), k, 5):
                fold[int(s_)] = k
    for gi, grp in enumerate(groups):
        pcs = [ES.member(path, dd, grp) for path, dd in views]
        for s in grp:
            rows = np.nonzero(d.shot == s)[0]
            mem = [pcs[fold[int(s)]]] if int(s) in fold else pcs                 # out-of-fold trunk, else the mean of all
            pc = torch.tensor(np.mean([m[s][0] for m in mem], 0), device="cuda", dtype=torch.float64)
            for a in range(0, len(rows), 4096):
                rr = rows[a:a + 4096]; psi = (pc[a:a + 4096] @ comps + mean).view(-1, 65, 65)
                T_ = torch.tensor(np.asarray(Te[rr]), device="cuda", dtype=torch.float64); N_ = torch.tensor(np.asarray(ne[rr]), device="cuda", dtype=torch.float64)
                P_ = torch.tensor(np.repeat(pos[s][None], len(rr), 0), device="cuda", dtype=torch.float64)
                out[rr] = feats_gpu(sc, psi, P_, T_, N_)
        if gi % 10 == 0:
            print(f"[thp] {psi_name} {tag}: group {gi + 1}/{len(groups)} | {time.time() - t:.0f}s", flush=True)
    out.flush(); F = np.load(fx.OUT / f"thp_{tag}_{psi_name}.npy", mmap_mode="r")
    print(f"THP_FEAT_DONE {tag} {psi_name}: {F.shape}, ok {float(np.mean(F[:, 3 * len(G) + 6] > 0)):.4f} | {time.time() - t:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["raw", "feat"]); ap.add_argument("--public", action="store_true")
    ap.add_argument("--psi", default="psiF"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    raw(a.public, a.workers) if a.mode == "raw" else feat(a.psi, a.public)


if __name__ == "__main__":
    main()
