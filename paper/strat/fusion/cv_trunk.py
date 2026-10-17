"""Doc 35 §38a: quick trunk screening by stratified shot subsampling + K-fold cross-validation (TRAIN shots only).

User (Oct 2): no massive training runs without evidence; iterate on stratified samples with cross-validation.
N_SHOTS TRAIN shots sampled within 3 x 3 strata (tercile of mean |Ip| x tercile of mean betaN, shot level, seed 0) and
split into K shot-level folds stratified the same way. Per fold: train on K-1 folds with the training loop of
sota.train_eval (same loss, weighting, schedule), score the held-out fold:
- r2_psi: pooled flux R^2, exact in coefficient space (orthonormal PCA; truth = the stored 256-coefficient psi).
- r2 q95 and betaN.
VALID/TEST are never touched.

    .venv/bin/python paper/strat/fusion/cv_trunk.py <arm> '<cfg overrides json>'   ->  runs/strat/fusion/cv_trunk/<arm>.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import fx  # noqa: I001  (paths)

N_SHOTS, K = int(os.environ.get("CV_SHOTS", "900")), int(os.environ.get("CV_FOLDS", "3"))
BASE = dict(k=64, width=2048, depth=2, drop=0.1, lr=0.0025322139193052723, batch=4096, epochs=30, w_scal=1.5123263659215784,
            gamma=0.3152355297039971, lags="sym", thomson=True, quantum="none", qcfg=None, k_lo=32, q_layers=4, seed=2, edge_w=2.0)


def folds(d, sota):
    tr_shots = d.shots_of(0); rng = np.random.default_rng(0)
    ipc = d.fn.index("mag_plasma_current"); bi = sota.LABELS.index("efit_beta_n")
    feats = []
    for s in tr_shots:
        rows = np.nonzero(d.shot == s)[0]
        feats.append((np.nanmean(np.abs(np.asarray(d.X[rows[::5], ipc], np.float64))), np.nanmean(d.S[rows, bi])))
    F = np.nan_to_num(np.array(feats))
    t = lambda v: np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
    strat = 3 * t(F[:, 0]) + t(F[:, 1])
    pick, fold = [], []
    for g in range(9):
        idx = rng.permutation(np.nonzero(strat == g)[0])[: int(round(N_SHOTS * (strat == g).mean()))]
        pick += [tr_shots[i] for i in idx]; fold += [j % K for j in range(len(idx))]
    return np.array(pick), np.array(fold)


def run_fold(d, sota, c, train_rows, test_rows, dev):
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    fm = d.feature_mask(c["lags"], c["thomson"]); k = c["k"]; edge = bool(c.get("edge", False))
    X = d.inputs(train_rows, fm, edge, bool(c.get("hist")), bool(c.get("thz"))); mu, sd = sota.moments64(X)
    hidx = [sota.LABELS.index(h) for h in sota.HEADS]
    Str = d.S[train_rows][:, hidx]; smu, ssd = np.nanmean(Str, 0), np.nanstd(Str, 0) + 1e-6
    T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=dev)
    Xg = T(np.clip((X - mu) / sd, -10, 10)); del X
    Cg = T(d.C[train_rows][:, :k]); Sg = T((Str - smu) / ssd)
    sv = d.meta["sv"][:k].to(dev); w = (sv / sv[0]) ** (-c["gamma"]); w = w / w.mean()
    pos = d.edge_feats()[train_rows, 1]; Wg = T(1 + c["edge_w"] * ((pos < 0.1) | (pos > 0.9))); Wg = Wg / Wg.mean()
    model = sota.Hybrid(Xg.shape[1], k, c["width"], c["depth"], c["drop"], c["quantum"], c["qcfg"], c["k_lo"], c["q_layers"],
                        c.get("k_hi")).to(dev)
    steps = c["epochs"] * max(1, len(Cg) // c["batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps, pct_start=0.05)
    denom = Cg.pow(2).sum(1).mean(); clip = float(c.get("clip", 0)); model.train()
    for _ in range(steps):
        i = torch.randint(0, len(Cg), (c["batch"],), device=dev)
        pc, ps = model(Xg[i])
        lp = ((((pc - Cg[i]) ** 2) * w).sum(1) * Wg[i]).mean() / denom
        tgt = Sg[i]; m = torch.isfinite(tgt)
        loss = lp + c["w_scal"] * ((ps - torch.nan_to_num(tgt)) ** 2 * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); sch.step()
    model.eval(); del Xg
    with torch.no_grad():
        xt = T(np.clip((d.inputs(test_rows, fm, edge, bool(c.get("hist")), bool(c.get("thz"))) - mu) / sd, -10, 10))
        pc, ps = zip(*[model(xt[j:j + 32768]) for j in range(0, len(xt), 32768)])
        pc = torch.cat(pc); ps = torch.cat(ps).cpu().numpy() * ssd + smu
        C = T(d.C[test_rows]); pad = torch.zeros_like(C); pad[:, :k] = pc
        sse = float((pad - C).square().sum())
        comps = d.meta["comps"].to(dev).float(); mean = d.meta["mean"].to(dev).float()
        s1 = s2 = 0.0; n = 0
        for j in range(0, len(C), 8192):
            p = C[j:j + 8192] @ comps + mean; s1 += float(p.sum()); s2 += float(p.square().sum()); n += p.numel()
        sst = s2 - s1 * s1 / n
    out = {"r2_psi": 1 - sse / sst}
    for h in ("efit_q95", "efit_beta_n"):
        y = d.S[test_rows, sota.LABELS.index(h)]; p = ps[:, sota.HEADS.index(h)]; mm = np.isfinite(y)
        out[f"r2_{h}"] = float(1 - ((y[mm] - p[mm]) ** 2).sum() / ((y[mm] - y[mm].mean()) ** 2).sum())
    return out


def main() -> None:
    arm = sys.argv[1]; over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    c = {**BASE, **over}
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("STRAT_CUDA", "0")
    import sota
    sota.DEV = dev = torch.device("cuda"); d = sota.Data()
    shots, fold = folds(d, sota)
    res = {"arm": arm, "cfg": c, "n_shots": len(shots), "folds": []}
    for f in range(K):
        tr = np.nonzero(np.isin(d.shot, shots[fold != f]))[0]; te = np.nonzero(np.isin(d.shot, shots[fold == f]))[0]
        r = run_fold(d, sota, c, tr, te, dev); r["n_train_frames"] = int(len(tr)); res["folds"].append(r)
        print(f"[cv] {arm} fold {f}: " + " ".join(f"{k_} {v:.5f}" for k_, v in r.items() if k_.startswith("r2")), flush=True)
    for key in ("r2_psi", "r2_efit_q95", "r2_efit_beta_n"):
        v = [r[key] for r in res["folds"]]; res[key] = [float(np.mean(v)), float(np.std(v))]
    out = fx.OUT / "cv_trunk"; out.mkdir(exist_ok=True); (out / f"{arm}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"CV_DONE {arm} r2_psi {res['r2_psi'][0]:.5f}+-{res['r2_psi'][1]:.5f} q95 {res['r2_efit_q95'][0]:.4f} betaN {res['r2_efit_beta_n'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
