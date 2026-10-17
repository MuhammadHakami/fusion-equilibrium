"""Doc 35 §37p step 2: direct shape heads (GPU). Do they beat the shape the scorer extracts from the predicted psi?

Targets (boundary_labels.npz): the scorer's 7 consistency scalars and the LCFS radius at 32 angles about the axis, on
every 3rd TRAIN frame (TRAIN fits). Inputs and trunk shape = the entry's first member config (trunk1_t19-ew2: width
2048, depth 2, dropout 0.1, lr 2.53e-3, batch 4096, 30 epochs, first/last 10 % of each shot weighted x3), MSE on
standardised targets, NaN masked. Output: predictions on the 180-shot VALID sample rows (no selection made on them here).

    flock runs/strat/gpu.lock env QW_GPU_GB=12 runs/capped.sh s37p 24 .venv/bin/python paper/strat/fusion/head_boundary.py <seed>
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("STRAT_CUDA", "0")       # this step trains on the GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import fx  # noqa: E402,I001

CK = fx.TRUNKS / "trunk1_t19-ew2_seed2.pt"
CLIP = float(os.environ.get("STRAT_CLIP", "0"))
FAST = os.environ.get("STRAT_FAST", "1") == "1"                               # Oct 3: bf16 autocast + fused AdamW + compile (3.7x)


class PLREmb(nn.Module):
    """§45f numerical-feature embedding (Gorishniy et al. 2022, PLR): per feature, sin / cos(2 pi c x) with K trainable frequencies
    c ~ N(0, sigma), a per-feature linear map 2K -> d and ReLU; output flattened to fin x d."""
    def __init__(self, fin, k, d, sigma):
        super().__init__()
        self.c = nn.Parameter(sigma * torch.randn(fin, k)); self.w = nn.Parameter(torch.randn(fin, 2 * k, d) / (2 * k) ** 0.5)
        self.b = nn.Parameter(torch.zeros(fin, d))

    def forward(self, x):
        v = 2 * torch.pi * x[..., None] * self.c
        return torch.relu(torch.einsum("bfk,fkd->bfd", torch.cat([torch.sin(v), torch.cos(v)], -1), self.w) + self.b).flatten(1)


class Net(nn.Module):
    def __init__(self, fin, out, width, depth, drop, emb=None):
        super().__init__()
        self.emb = PLREmb(fin, int(emb[0]), int(emb[1]), float(emb[2])) if emb else None    # emb = (K, d, sigma) or None (all heads < §45f)
        self.inp = nn.Sequential(nn.Linear(fin * int(emb[1]) if emb else fin, width), nn.GELU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(drop),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, out); self.skip = nn.Linear(fin, out)

    def forward(self, x):
        h = self.inp(self.emb(x) if self.emb is not None else x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(self.norm(h)) + self.skip(x)


def cv_shots(d, tr, Ytr, k: int, K: int) -> np.ndarray:
    """§41q stratified K-fold over TRAIN shots: strata = Ip sign (ip_sign.json) x quartile of the shot-mean li label; shots dealt
    round-robin (random order, fixed seed) within each stratum. Returns the shot ids of fold k."""
    import json
    sg = json.loads((fx.OUT / "ip_sign.json").read_text()); sh = d.shot[tr]; us, inv = np.unique(sh, return_inverse=True)
    li = Ytr[:, 6]; ok = np.isfinite(li)
    m = np.bincount(inv[ok], li[ok], minlength=len(us)) / np.maximum(np.bincount(inv[ok], minlength=len(us)), 1)
    q = np.searchsorted(np.quantile(m, [0.25, 0.5, 0.75]), m); strata = q + 4 * np.array([sg[str(int(u))] < 0 for u in us])
    out, rng = [], np.random.default_rng(0)
    for st_ in np.unique(strata):
        mem = rng.permutation(us[strata == st_]); out.append(mem[k::K])
    return np.concatenate(out)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    mode = sys.argv[2] if len(sys.argv) > 2 else "shape"                       # "qb": §37r q95 / betaN heads, all TRAIN rows
    torch.manual_seed(seed); np.random.seed(seed)
    import sota
    sota.DEV = dev = torch.device("cuda")
    d = sota.Data()
    if os.environ.get("STRAT_INPUTS", "") == "magfix":                         # §38g: corrected magnetics block
        import inputs_fix
        inputs_fix.use(d)
    if os.environ.get("STRAT_CANON", "") in ("1", "2"):                       # §41q: Ip-sign canonical currents (2: + sign input)
        import canon
        canon.use(d, indicator=os.environ["STRAT_CANON"] == "2")
    if os.environ.get("STRAT_HFULL", "") == "1":                              # §41q: full-rate current history (24 inputs)
        import hfull_inputs
        hfull_inputs.use(d)
    if os.environ.get("STRAT_THZ", "") == "1":                                # §41q: + fixed-coordinate Thomson (§41f thz block)
        import hfull_inputs
        hfull_inputs.use(d, kind="thz")
    for kind in ("ctx", "thzs", "liphys", "qshape"):                                               # §41q: shot context / smoothed thz
        if os.environ.get(f"STRAT_{kind.upper()}", "") == "1":
            import hfull_inputs
            hfull_inputs.use(d, kind=kind)
    if os.environ.get("STRAT_FUT", "") == "1":                               # §41t: + future context (current quench ahead)
        import hfull_inputs
        hfull_inputs.use(d, kind="fut")
    if os.environ.get("STRAT_THP", ""):                                      # §41r: + Thomson on psi_N of a psi trunk set ("psiF")
        import hfull_inputs
        hfull_inputs.use(d, kind="thp_" + os.environ["STRAT_THP"])
    if os.environ.get("STRAT_QPHYS", "") in ("1", "2"):                      # §42a: inputs-only physics scalings (q / beta ratios)
        import qphys_inputs                                                    # "2" (§42e): + coil currents / |Ip| (Shafranov)
        qphys_inputs.use(d, ratios=os.environ["STRAT_QPHYS"] == "2")
    if os.environ.get("STRAT_VAC", "") == "1":                                 # §45j: vacuum-field inputs (coil Green's functions)
        import vac_inputs
        vac_inputs.use(d)
    if os.environ.get("STRAT_SUBF", "") == "1":                                # §45m: sub-frame window statistics (subf_feats.py)
        import hfull_inputs
        hfull_inputs.use(d, kind="subf")
    if "QW_GPU_GB" in os.environ:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["QW_GPU_GB"]) / 120.0))
    ck = torch.load(CK, map_location="cpu", weights_only=False); c = ck["cfg"]; fm = ck["fm"]; edge = bool(c.get("edge", False))
    edge = edge or os.environ.get("STRAT_EDGE", "") == "1"                   # §38y: shot-phase inputs (time to end, elapsed)
    if os.environ.get("STRAT_NO_THOMSON", "") == "1":                          # §41q: drop the index-resampled Thomson block (thz replaces it)
        fm = np.asarray(fm, bool).copy(); fm[[i for i, n_ in enumerate(d.fn) if n_.startswith("thomson")]] = False
    tag = os.environ.get("STRAT_LABEL_SUFFIX", "")                            # §37s: "_v2" = 64 rays, all TRAIN frames
    lab = np.load(fx.OUT / f"boundary_labels{tag}.npz")
    tr, Ytr, va = lab["rows_train"], lab["y_train"].astype(np.float64), lab["rows_valid"]
    fold = os.environ.get("STRAT_FOLD", "")                                   # §38u: "k/K" -> hold out 1/K of TRAIN shots
    cv = os.environ.get("STRAT_CV", "")                                       # §41q: stratified shot folds "k/K" (proxy CV)
    if cv:
        ck_, cK_ = map(int, cv.split("/")); held = cv_shots(d, tr, Ytr, ck_, cK_)
        hm = np.isin(d.shot[tr], held); oof_rows = tr[hm]; tr, Ytr = tr[~hm], Ytr[~hm]; fold = cv
    elif fold:
        fk, fK = map(int, fold.split("/")); us = np.random.default_rng(0).permutation(np.unique(d.shot[tr]))
        hm = np.isin(d.shot[tr], us[fk::fK]); oof_rows = tr[hm]; tr, Ytr = tr[~hm], Ytr[~hm]
    sub = os.environ.get("STRAT_SUB", "")                                     # §41q fast proxy: train on TRAIN-shot subset k of K
    if sub:
        sk, sK = map(int, sub.split("/")); us = np.random.default_rng(0).permutation(np.unique(d.shot[tr]))
        keep = np.isin(d.shot[tr], us[sk::sK]); tr, Ytr = tr[keep], Ytr[keep]
    allf = os.environ.get("STRAT_ALL", "") == "1"                              # §41q final build: + every VALID / TEST frame
    if allf and mode == "shape":
        vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); assert vt["y"].shape[1] == Ytr.shape[1]
        tr, Ytr = np.concatenate([tr, vt["rows"]]), np.concatenate([Ytr, vt["y"].astype(np.float64)])
    if mode == "qb":
        tr = np.arange(len(d.split)) if allf else np.nonzero(d.split == 0)[0]
        if sub:
            tr = tr[np.isin(d.shot[tr], us[sk::sK])]
        Ytr = d.S[tr][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64)
    if mode == "xp":                                                          # §42t: active X-point heads (xp_labels.py: R_l, Z_l, R_u, Z_u)
        xl = np.load(fx.OUT / (os.environ.get("STRAT_XP_LABELS", "xp_labels") + ".npz")); yx = xl["y"]; tr = np.arange(len(d.split)) if allf else np.nonzero(d.split == 0)[0]
        tr = tr[np.isfinite(yx[tr]).any(1)]
        if sub:
            tr = tr[np.isin(d.shot[tr], us[sk::sK])]
        Ytr = yx[tr].astype(np.float64)
    qlog = mode == "qb" and os.environ.get("STRAT_QB_LOG", "") == "1"        # §41q: log q95 / log betaN targets (q ~ 1/Ip)
    qlw = float(os.environ.get("STRAT_QB_LOGW", "0"))                         # weight y^qlw per output (2 = q-space MSE)
    if qlog:
        Yraw = np.where(Ytr > 0, Ytr, np.nan); Ytr = np.log(Yraw)
    n_main = Ytr.shape[1]
    aux = float(os.environ.get("STRAT_AUX_PSI", "0"))                         # §41q: + the 64 psi PCA coefficients as aux targets
    if aux > 0:
        Ytr = np.concatenate([Ytr, np.asarray(d.C[tr][:, :64], np.float64)], 1)
    axp = float(os.environ.get("STRAT_AUX_XP", "0"))                          # §42v: + the 4 X-point labels (xp_labels.npz) as aux targets
    if axp > 0:
        Ytr = np.concatenate([Ytr, np.load(fx.OUT / (os.environ.get("STRAT_XP_LABELS", "xp_labels") + ".npz"))["y"][tr].astype(np.float64)], 1)
    X = d.inputs(tr, fm, edge); mu, sd = sota.moments64(X)
    ymu, ysd = np.nanmean(Ytr, 0), np.nanstd(Ytr, 0) + 1e-9
    T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=dev)
    Xg = T(np.clip((X - mu) / sd, -10, 10)); del X
    Yn = (Ytr - ymu) / ysd; Mg = T(np.isfinite(Yn)); Yg = T(np.nan_to_num(Yn))
    if qlog and qlw > 0:                                                      # fold the y^qlw weights into the mask
        V = np.nan_to_num(Yraw ** qlw); V = V / np.nanmean(np.where(np.isfinite(Yn), V, np.nan), 0); Mg = Mg * T(V)
    pos = d.edge_feats()[tr, 1]; Wg = T(1 + c.get("edge_w", 0.0) * ((pos < 0.1) | (pos > 0.9))
                                        + float(os.environ.get("STRAT_END_W", "0")) * (pos > 0.9)); Wg = Wg / Wg.mean()   # §38y
    c = {**c, "width": int(os.environ.get("STRAT_WIDTH", c["width"])), "depth": int(os.environ.get("STRAT_DEPTH", c["depth"])),
         "drop": float(os.environ.get("STRAT_DROP", c["drop"])),                  # §41q: dropout override
         "emb": [float(v) for v in os.environ["STRAT_EMB"].split(",")] if os.environ.get("STRAT_EMB") else None}   # §45f: "K,d,sigma"
    wd = float(os.environ.get("STRAT_WD", "1e-4")); xnoise = float(os.environ.get("STRAT_XNOISE", "0"))   # §41q regularisation
    mix = float(os.environ.get("STRAT_MIXUP", "0")); mixd = torch.distributions.Beta(mix, mix) if mix > 0 else None   # §42x: Beta(a, a)
    net = Net(Xg.shape[1], Yg.shape[1], c["width"], c["depth"], c["drop"], c.get("emb")).to(dev)
    torch.backends.cuda.matmul.allow_tf32 = FAST; fwd = torch.compile(net) if FAST else net
    sw = float(os.environ.get("STRAT_SCAL_W", "1"))                            # §37aa: weight of the 7 scalar outputs
    OW = torch.ones(Yg.shape[1], device=dev); OW[:7] = sw
    if aux > 0:
        OW[n_main:] = aux
    if axp > 0:
        OW[-4:] = axp
    epochs = int(os.environ.get("STRAT_EPOCHS", c["epochs"]))                  # §37z: 60 (default = the trunk's 30)
    steps = epochs * (len(Xg) // c["batch"])
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=wd, fused=FAST)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps, pct_start=0.05)
    HUB = float(os.environ.get("STRAT_HUBER", "0"))                            # §45h: robust loss (standardised units; 0 = MSE)
    ema_d = float(os.environ.get("STRAT_EMA", "0"))                            # §45g: exponential moving average of the weights
    ema = (torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_d)) if ema_d > 0 else None)
    net.train()
    for step in range(steps):
        i = torch.randint(0, len(Xg), (c["batch"],), device=dev)
        xb = Xg[i] if xnoise <= 0 else Xg[i] + xnoise * torch.randn_like(Xg[i])     # §41q: Gaussian input noise (standardised)
        yb, mb, wb = Yg[i], Mg[i], Wg[i]
        if mix > 0:                                                             # §42x: mixup (convex pairs; a target counts only
            j = torch.randperm(len(i), device=dev); lam = mixd.sample((len(i), 1)).to(dev)   # where both members are labelled)
            xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            mb = mb * mb[j]; wb = lam[:, 0] * wb + (1 - lam[:, 0]) * wb[j]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=FAST):
            out = fwd(xb)
        r = out.float() - yb
        e = (r.square() if HUB <= 0 else 2 * torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=HUB)) * mb * OW   # §45h
        loss = (e.sum(1) * wb).sum() / mb.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        if CLIP > 0:                                                            # §37u: gradient-norm clipping (0 = off, as §37p-t)
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step(); sch.step()
        if ema is not None:
            ema.update_parameters(net)
        if step % 2000 == 0:
            print(f"[s37p-head] step {step}/{steps} loss {loss.item():.4f}", flush=True)
    if ema is not None:                                                        # predictions and the checkpoint use the averaged weights
        net.load_state_dict(ema.module.state_dict())
    net.eval()
    with torch.no_grad():
        xv = T(np.clip((d.inputs(va, fm, edge) - mu) / sd, -10, 10))
        pv = torch.cat([net(xv[i:i + 32768]) for i in range(0, len(xv), 32768)]).cpu().numpy() * ysd + ymu
    pv = np.exp(pv[:, :n_main]) if qlog else pv[:, :n_main]
    stem = {"shape": f"head_boundary{tag}", "qb": "head_qb", "xp": "head_xp"}[mode] + os.environ.get("STRAT_OUT_TAG", "")
    np.savez(fx.OUT / f"{stem}_s{seed}.npz", rows_valid=va, pred_valid=pv.astype(np.float32))
    if fold:                                                                  # out-of-fold predictions on the held-out shots
        with torch.no_grad():
            xo = T(np.clip((d.inputs(oof_rows, fm, edge) - mu) / sd, -10, 10))
            po = torch.cat([net(xo[i:i + 32768]) for i in range(0, len(xo), 32768)]).cpu().numpy() * ysd + ymu
        np.savez(fx.OUT / f"{stem}_s{seed}_oof.npz", rows_oof=oof_rows, pred_oof=po[:, :n_main].astype(np.float32))
    torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd, "fm": fm, "edge": edge, "cfg": c,
                "inputs": os.environ.get("STRAT_INPUTS", ""), "magfix_name": os.environ.get("QW_MAGFIX", "X_magfix"),
                "canon": {"1": 1, "2": 2}.get(os.environ.get("STRAT_CANON", ""), 0), "hfull": os.environ.get("STRAT_HFULL", "") == "1",
                "thz_extra": os.environ.get("STRAT_THZ", "") == "1", "all_data": allf, "qb_log": qlog, "n_main": n_main,
                "ctx": os.environ.get("STRAT_CTX", "") == "1", "thzs": os.environ.get("STRAT_THZS", "") == "1", "liphys": os.environ.get("STRAT_LIPHYS", "") == "1", "qshape": os.environ.get("STRAT_QSHAPE", "") == "1",
                "thp": os.environ.get("STRAT_THP", ""), "fut": os.environ.get("STRAT_FUT", "") == "1",
                "qphys": {"1": 1, "2": 2}.get(os.environ.get("STRAT_QPHYS", ""), 0), "aux_xp": axp,
                "vac": os.environ.get("STRAT_VAC", "") == "1", "subf": os.environ.get("STRAT_SUBF", "") == "1"},
               fx.OUT / f"{stem}_s{seed}.pt")
    print(f"{'S37P_HEAD_DONE' if mode == 'shape' else 'S37R_HEAD_DONE'} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
