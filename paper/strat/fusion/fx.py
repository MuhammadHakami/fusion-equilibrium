"""§37 fusion (Sophelio DIII-D) stratified exploration: shared CPU utilities. VALID/TRAIN only; TEST is never loaded.

- CPU inference of saved trunks (the submitted entry = runs/sophelio_sota/ens_select_mlp.json, 7 members).
- A per-FRAME decomposition of the official composite S with the vendored scorer's own functions
  (local_score.score_shot logic, fusion_scoring/*): psi SS_res, q95 / betaN residuals, D_LCFS, and the 7 consistency
  residuals, per frame. With the fold's SS_tot held fixed, 1 - S is additive over shots and frames, so any stratum's
  share of the loss (and a paired bootstrap over shots) is exact.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU only (the Flowers run owns the GPU)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
STARTER = ROOT / "baseline" / "sophelio-starter"
for p in (ROOT / "paper" / "sophelio", ROOT / "paper", STARTER, STARTER / "fusion_scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
OUT = ROOT / "runs" / "strat" / "fusion"
OUT.mkdir(parents=True, exist_ok=True)
REF = ROOT / "runs" / "sophelio_sota" / "ref_cache"
TRUNKS = ROOT / "runs" / "nas" / "sophelio" / os.environ.get("QW_ENTRY_DIR", "trunks")       # §41q: trunks_all = all-data entry
ENTRY = json.loads((ROOT / "runs" / "sophelio_sota" / "ens_select_mlp.json").read_text())["members"]
W_PSI, W_QB, W_LCFS, W_CONS = 0.55, 0.15, 0.10, 0.20
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
N_WORKERS = int(os.environ.get("STRAT_WORKERS", "8"))


def data():
    import torch
    torch.set_num_threads(8)
    import sota
    sota.DEV = torch.device("cpu")
    return sota, sota.Data()


def valid_rows(d, s: int) -> np.ndarray:
    rows = np.nonzero(d.shot == s)[0]
    assert (d.split[rows] == 1).all(), "only VALID shots are scored here"
    return rows


def scored_rows(d, s: int) -> np.ndarray:
    """Rows of a held-out shot (VALID or TEST; §41y VT-wide twin parts). TRAIN shots are never scored."""
    rows = np.nonzero(d.shot == s)[0]
    assert (d.split[rows] != 0).all(), "TRAIN shots are never scored"
    return rows


def member_predict(path, d, rows_by_shot: dict) -> dict:
    """CPU inference of one saved trunk: {shot: (coef (T, k), scalars (T, 5))}."""
    import torch
    import sota
    ck = torch.load(path, map_location="cpu", weights_only=False); c = ck["cfg"]
    m = sota.Hybrid(len(ck["mu"]), c["k"], c["width"], c["depth"], c["drop"], c["quantum"], c["qcfg"], c["k_lo"],
                    c["q_layers"], c.get("k_hi"))
    m.load_state_dict(ck["model"]); m.eval()
    out = {}
    with torch.no_grad():
        for s, rows in rows_by_shot.items():
            x = np.clip((d.inputs(rows, ck["fm"], c.get("edge", False), thz=bool(c.get("thz"))) - ck["mu"]) / ck["sd"], -10, 10)
            pc, ps = m(torch.tensor(x, dtype=torch.float32))
            out[s] = (pc.numpy().astype(np.float32), (ps.numpy() * ck["ssd"] + ck["smu"]).astype(np.float32))
    return out


# ----------------------------------------------------------------------------------------------- per-frame scorer
def _frame_parts(args):
    """One shot, per frame, exactly as local_score.score_shot (psi_sign = +1, as every VALID/TEST/public score had)."""
    key, psi_gt, q95_gt, bn_gt, psi_pred, q95_p, bn_p, mean_cons = args
    import local_score as LS
    from contour import symmetric_hausdorff
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]; mc = mask["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
    z = np.load(REF / f"ref_{key}.npz", allow_pickle=True)
    contours, cons_gt, cmask, rgeo = list(z["contours"]), z["cons"], z["cmask"], float(z["rgeo"])
    T = len(psi_gt); axis_sign = LS.AXIS_SIGN[LS.MACHINE] * 1.0
    g = psi_gt.astype(np.float64); p = psi_pred.astype(np.float64)
    psi_res = ((g - p) ** 2).reshape(T, -1).sum(1)
    psi_sum = g.reshape(T, -1).sum(1); psi_sq = (g ** 2).reshape(T, -1).sum(1)
    d = np.full(T, np.nan); cres = np.full((T, 7), np.nan); cpred = np.full((T, 7), np.nan); fails = 0
    for k in range(T):
        ref_c, need = contours[k], cmask[k]
        if ref_c is None and not need.any():
            continue
        ex, _ = extract_lcfs_with_sign(p[k], R, Z, LS.MACHINE, mc, mf, n_iter=LS.N_ITER, axis_sign=axis_sign)
        if ref_c is not None:
            d[k] = 1.0 if ex is None else min(1.0, symmetric_hausdorff(ex, np.asarray(ref_c, np.float64)) / rgeo if rgeo > 0 else 1.0)
        if need.any():
            vals = derive_frame(p[k], R, Z, LS.MACHINE, mc, mf, contour=ex, with_li=bool(need[LS.LI_IDX]), axis_sign=axis_sign)
            for j in np.nonzero(need)[0]:
                v = vals[LS.CONS_SCALARS[j]]
                if not np.isfinite(v):
                    v = mean_cons[j]; fails += 1
                cpred[k, j] = v; cres[k, j] = (cons_gt[k, j] - v) ** 2
    q_res = (q95_gt.astype(np.float64) - q95_p) ** 2; b_res = (bn_gt.astype(np.float64) - bn_p) ** 2
    return dict(psi_res=psi_res, psi_sum=psi_sum, psi_sq=psi_sq, npix=psi_gt[0].size, q_res=q_res, b_res=b_res,
                q_gt=q95_gt.astype(np.float64), b_gt=bn_gt.astype(np.float64), d=d, cres=cres, cpred=cpred,
                cgt=np.where(cmask, cons_gt, np.nan), fails=fails)


def frame_parts(d, shots, pred_fn, mean_cons) -> dict:
    """pred_fn(shot) -> (psi (T, 4225) float64, q95 (T,), betaN (T,)). Returns {shot: per-frame parts}."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import sota
    L = sota.LABELS
    jobs = []
    for s in shots:
        rows = valid_rows(d, s)
        psi, q, b = pred_fn(s)
        jobs.append((d.meta["names"][s], d.psi_of(rows).reshape(-1, 65, 65), d.S[rows, L.index("efit_q95")],
                     d.S[rows, L.index("efit_beta_n")], np.asarray(psi, np.float64).reshape(-1, 65, 65),
                     np.asarray(q, np.float64), np.asarray(b, np.float64), mean_cons))
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_frame_parts, jobs, chunksize=2))
    return dict(zip(shots, res))


# ----------------------------------------------------------------------------------------------- composite from parts
def norms(parts_gt: dict) -> dict:
    """Per-frame SS_tot normalisers of the reference fold (full VALID): pooled variance per element."""
    ps = sum(p["psi_sum"].sum() for p in parts_gt.values()); pq = sum(p["psi_sq"].sum() for p in parts_gt.values())
    npx = sum(p["npix"] * len(p["psi_sum"]) for p in parts_gt.values())
    out = dict(psi_var=(pq - ps * ps / npx) / npx, npix=list(parts_gt.values())[0]["npix"])
    for nm in ("q", "b"):
        v = np.concatenate([p[f"{nm}_gt"] for p in parts_gt.values()]); v = v[np.isfinite(v)]
        out[f"{nm}_var"] = float(v.var())
    c = np.concatenate([p["cgt"] for p in parts_gt.values()])
    out["c_var"] = np.nanvar(c, axis=0); out["c_mean"] = np.nanmean(c, axis=0)
    return out


def shot_sums(p: dict, fsel=None) -> np.ndarray:
    """Sufficient statistics of one shot (optionally a frame subset) for the fixed-normaliser composite:
    [psi_res, n_frames, q_res, q_n, b_res, b_n, d_sum, d_n, cres_1..7, cn_1..7]."""
    T = len(p["psi_res"]); f = np.ones(T, bool) if fsel is None else fsel
    qm = f & np.isfinite(p["q_gt"]); bm = f & np.isfinite(p["b_gt"]); dm = f & np.isfinite(p["d"])
    cm = f[:, None] & np.isfinite(p["cres"])
    return np.concatenate([[p["psi_res"][f].sum(), f.sum(), p["q_res"][qm].sum(), qm.sum(), p["b_res"][bm].sum(), bm.sum(),
                            p["d"][dm].sum(), dm.sum()], np.where(cm, p["cres"], 0).sum(0), cm.sum(0)])


def composite(stats: np.ndarray, nm: dict) -> dict:
    """stats: (n_shots, 22) from shot_sums. D_LCFS is the mean over shots of per-shot frame means (as the scorer)."""
    s = stats.sum(0)
    r2psi = 1 - s[0] / (s[1] * nm["npix"] * nm["psi_var"])
    r2q = 1 - s[2] / (s[3] * nm["q_var"]); r2b = 1 - s[4] / (s[5] * nm["b_var"])
    ok = stats[:, 7] > 0
    dl = float(np.mean(stats[ok, 6] / stats[ok, 7])) if ok.any() else 0.0
    r2c = 1 - s[8:15] / (s[15:22] * nm["c_var"])
    cons = float(np.mean(np.maximum(0, r2c)))
    qb = 0.5 * (r2q + r2b)
    S = W_PSI * max(0, r2psi) + W_QB * max(0, qb) + W_LCFS * (1 - min(1, dl)) + W_CONS * cons
    return dict(S=float(S), r2_psi=float(r2psi), r2_qb=float(qb), r2_q95=float(r2q), r2_betaN=float(r2b), dlcfs=dl,
                consistency=cons, cons_each=dict(zip(CONS, map(float, r2c))),
                loss=dict(psi=float(W_PSI * (1 - r2psi)), qb=float(W_QB * (1 - qb)), dlcfs=float(W_LCFS * dl),
                          cons=float(W_CONS * (1 - cons))))


def paired_boot(stA: np.ndarray, stB: np.ndarray, nm: dict, groups: np.ndarray | None = None, B: int = 2000,
                seed: int = 0) -> dict:
    """Paired bootstrap over shots (stratified by `groups` if given) of S(B) - S(A)."""
    rng = np.random.default_rng(seed); n = len(stA)
    base = composite(stB, nm)["S"] - composite(stA, nm)["S"]
    idx_g = [np.arange(n)] if groups is None else [np.nonzero(groups == g)[0] for g in np.unique(groups)]
    ds = np.empty(B)
    for b in range(B):
        ii = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in idx_g])
        ds[b] = composite(stB[ii], nm)["S"] - composite(stA[ii], nm)["S"]
    return dict(delta=float(base), lo=float(np.percentile(ds, 2.5)), hi=float(np.percentile(ds, 97.5)),
                p_le0=float((ds <= 0).mean()))
