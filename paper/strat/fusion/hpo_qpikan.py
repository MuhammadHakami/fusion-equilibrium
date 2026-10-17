"""Doc 35 §38ac: Optuna search (classical + quantum NAS and HPO) for the QPI-KAN shape head (qpikan.py).

Data and loss = the J1 heads' (head_boundary.py, v8 labels: 7 scalars + 64 rays; magfix inputs; scalar weight 9.14; edge
weights; clip 1.0), plus the physics-informed volume consistency (volume head vs the ray polygon, TRAIN-label calibration).
Selection never sees VALID: 10% of TRAIN shots (rng 0) are the search's validation set; the gate on the VALID sample is
run once on the finally retrained config. Search space:
  classical NAS  block {kan, mlp}, width {512, 1024, 2048}, depth {1, 2, 3}, Chebyshev order {3, 4, 6}, dropout
  quantum NAS    register layers {2, 4, 6}, controller width {32, 64, 128}, inter-patch link {independent, cascade}
                 (30-atom input-dependent Rydberg register, 3 x 10, always on: competitions use the hybrid only)
  HPO            learning rate, physics weight (0 or log-uniform 1e-3 .. 1)
Trials train on a TRAIN subsample for a few epochs (median pruning per epoch). --final N retrains trial N on all TRAIN
shots and writes head_boundary_v8_QK<N>_s<seed>.{npz,pt} (VALID predictions, as head_boundary.py).

    .venv/bin/python paper/strat/fusion/hpo_qpikan.py --trials 40 --frac 0.25 --epochs 6
    .venv/bin/python paper/strat/fusion/hpo_qpikan.py --final <trial> --epochs 60 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("STRAT_CUDA", "0")
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import qpikan as Q  # noqa: E402

CK = fx.TRUNKS / "trunk1_t19-ew2_seed2.pt"
DB = f"sqlite:///{fx.OUT / 'hpo_qpikan.db'}"


def data():
    import inputs_fix
    import sota
    sota.DEV = torch.device("cuda")
    d = sota.Data(); inputs_fix.use(d)
    if os.environ.get("STRAT_CANON", "") in ("1", "2"):                       # §45s: Ip-sign canonical currents (2: + sign input), as head_boundary
        import canon
        canon.use(d, indicator=os.environ["STRAT_CANON"] == "2")
    import hfull_inputs                                                      # wrappers in gpu_build.heads_gpu order
    if os.environ.get("STRAT_HFULL", "") == "1":                              # §41r: + full-rate current history (as J1AH)
        hfull_inputs.use(d)
    if os.environ.get("STRAT_THZ", "") == "1":                                # §41q: + fixed-coordinate Thomson (as J1T)
        hfull_inputs.use(d, kind="thz")
    if os.environ.get("STRAT_LIPHYS", "") == "1":                             # §43h: + li current-profile proxies (as J1RL)
        hfull_inputs.use(d, kind="liphys")
    if os.environ.get("STRAT_FUT", "") == "1":                                # §41t: + future context
        hfull_inputs.use(d, kind="fut")
    if os.environ.get("STRAT_THP", ""):                                       # §41r: + Thomson on predicted psi_N
        hfull_inputs.use(d, kind="thp_" + os.environ["STRAT_THP"])
    if os.environ.get("STRAT_QPHYS", "") in ("1", "2"):                       # §43h: + physics-ratio inputs (as J1RL; "2" = + coils / |Ip|)
        import qphys_inputs
        qphys_inputs.use(d, ratios=os.environ["STRAT_QPHYS"] == "2")
    ck = torch.load(CK, map_location="cpu", weights_only=False); c = ck["cfg"]; fm = ck["fm"]; edge = bool(c.get("edge", False))
    lab = np.load(fx.OUT / "boundary_labels_v8.npz")
    return d, sota, c, fm, edge, lab


class Setup:
    """Standardised tensors for a set of TRAIN rows (fit rows) and an evaluation row set."""

    def __init__(self, d, sota, c, fm, edge, rows, Y, eval_rows, eval_Y):
        dev = torch.device("cuda"); T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=dev)  # noqa: E731
        X = d.inputs(rows, fm, edge); self.mu, self.sd = sota.moments64(X)
        self.ymu, self.ysd = np.nanmean(Y, 0), np.nanstd(Y, 0) + 1e-9
        self.X = T(np.clip((X - self.mu) / self.sd, -10, 10)); del X
        Yn = (Y - self.ymu) / self.ysd; self.M = T(np.isfinite(Yn)); self.Y = T(np.nan_to_num(Yn))
        pos = d.edge_feats()[rows, 1]; W = 1 + c.get("edge_w", 0.0) * ((pos < 0.1) | (pos > 0.9)); self.W = T(W / W.mean())
        self.Xe = T(np.clip((d.inputs(eval_rows, fm, edge) - self.mu) / self.sd, -10, 10)) if eval_rows is not None else None
        self.Ye = eval_Y
        ok = np.isfinite(Y).all(1)                                              # volume calibration: label rays -> label volume
        k, v = Q.ray_shape(torch.tensor(Y[ok][:200000, :2]), torch.tensor(Y[ok][:200000, 7:]))
        A = np.c_[v.numpy(), np.ones(len(v))]; self.vcal = np.linalg.lstsq(A, Y[ok][:200000, 5], rcond=None)[0]
        self.vsd = float(np.nanstd(Y[:, 5]))


def build(p, fin, out):
    return Q.QPIKAN(fin, out, width=p["width"], depth=p["depth"], order=p.get("order", 4), block=p["block"], drop=p["drop"],
                    q_layers=p["q_layers"], ansatz="rydberg", link=p["link"], q_in=p["q_in"], register=True,
                    alpha_init=p.get("alpha_init", 0.0)).cuda()


def evaluate(net, S, batch=16384):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(S.Xe), batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(net(S.Xe[i:i + batch]).float())
    P = torch.cat(out).cpu().numpy() * S.ysd + S.ymu; net.train(); r2 = []
    for k in range(7):
        m = np.isfinite(S.Ye[:, k]); r2.append(1 - np.mean((S.Ye[m, k] - P[m, k]) ** 2) / np.var(S.Ye[m, k]))
    return float(np.mean(r2)), r2, P


def train(p, S, epochs, sw=9.142857, batch=4096, clip=1.0, trial=None, log=""):
    net = build(p, S.X.shape[1], S.Y.shape[1]); OW = torch.ones(S.Y.shape[1], device="cuda"); OW[:7] = sw
    steps = max(1, epochs * (len(S.X) // batch)); per_epoch = max(1, len(S.X) // batch)
    am = float(p.get("alpha_mult", 1.0)); rest = [q for n, q in net.named_parameters() if n != "alpha"]   # stage 2: alpha lr x am
    opt = torch.optim.AdamW([{"params": rest}, {"params": [net.alpha], "weight_decay": 0.0}], lr=p["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[p["lr"], p["lr"] * am], total_steps=steps, pct_start=0.05)
    ysd, ymu = torch.tensor(S.ysd, device="cuda", dtype=torch.float32), torch.tensor(S.ymu, device="cuda", dtype=torch.float32)
    a, b = (float(v) for v in S.vcal); wpi = p["w_pi"]; t0 = time.time(); best = -np.inf
    fwd = torch.compile(net) if os.environ.get("QW_QCOMPILE", "") == "1" else net   # §43h: inductor on top of QW_QFAST (1.1x)
    for step in range(steps):
        i = torch.randint(0, len(S.X), (batch,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(S.X[i])
        out = out.float(); e = (out - S.Y[i]).square() * S.M[i] * OW
        loss = (e.sum(1) * S.W[i]).sum() / S.M[i].sum().clamp(min=1)
        if wpi > 0:                                                             # physics-informed volume consistency
            y = out * ysd + ymu; k_, v_ = Q.ray_shape(y[:, :2], y[:, 7:])
            loss = loss + wpi * (((y[:, 5] - (a * v_ + b)) / S.vsd) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), clip); opt.step(); sch.step()
        if trial is None and (step + 1) % max(1, steps // 10) == 0:            # final mode: progress every 10% of the steps
            print(f"[qk]{log} step {step + 1}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % per_epoch == 0 and S.Xe is not None and trial is not None:
            r2, _, _ = evaluate(net, S); ep = (step + 1) // per_epoch; best = max(best, r2)
            trial.report(r2, ep); print(f"[qk]{log} epoch {ep}/{epochs} val R2 {r2:.5f} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return net


def space(trial):
    p = dict(block=trial.suggest_categorical("block", ["kan", "mlp"]), width=trial.suggest_categorical("width", [512, 1024, 2048]),
             depth=trial.suggest_int("depth", 1, 3), drop=trial.suggest_categorical("drop", [0.0, 0.05, 0.1]),
             q_layers=trial.suggest_categorical("q_layers", [2, 4, 6]), q_in=trial.suggest_categorical("q_in", [32, 64, 128]),
             link=trial.suggest_categorical("link", ["independent", "cascade"]), lr=trial.suggest_float("lr", 3e-4, 4e-3, log=True))
    if p["block"] == "kan":
        p["order"] = trial.suggest_categorical("order", [3, 4, 6])
    p["w_pi"] = trial.suggest_float("w_pi", 1e-3, 1.0, log=True) if trial.suggest_categorical("pi", [True, False]) else 0.0
    return p


def space_v2(trial):
    """§43j search space: around the deployed register-head recipe (mlp 4096 x 4, register engaged at alpha 0.1), wider / deeper
    than v1, with the physics-ratio inputs set by env (STRAT_THZ / HFULL / LIPHYS / QPHYS as J1RL)."""
    p = dict(block=trial.suggest_categorical("block", ["kan", "mlp"]), width=trial.suggest_categorical("width", [1024, 2048, 4096]),
             depth=trial.suggest_int("depth", 2, 4), drop=trial.suggest_categorical("drop", [0.0, 0.05, 0.1, 0.2]),
             q_layers=trial.suggest_categorical("q_layers", [2, 4]), q_in=trial.suggest_categorical("q_in", [64, 128]),
             link=trial.suggest_categorical("link", ["independent", "cascade"]), lr=trial.suggest_float("lr", 5e-4, 4e-3, log=True),
             alpha_mult=trial.suggest_categorical("alpha_mult", [1.0, 10.0]), alpha_init=0.1)
    if p["block"] == "kan":
        p["order"] = trial.suggest_categorical("order", [3, 4, 6])
    p["w_pi"] = trial.suggest_float("w_pi", 1e-3, 1.0, log=True) if trial.suggest_categorical("pi", [True, False]) else 0.0
    return p


BASE_V2 = {"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade",
           "lr": 0.0025322139193052723, "alpha_mult": 1.0, "pi": True, "w_pi": 0.1}           # = the deployed J1QPIpi recipe


def space_q(trial, base):
    """Stage 2 (quantum NAS): the classical part fixed to `base`; search how the register is engaged."""
    p = {**base, "q_layers": trial.suggest_categorical("q_layers", [2, 4, 6]), "q_in": trial.suggest_categorical("q_in", [32, 64, 128]),
         "link": trial.suggest_categorical("link", ["independent", "cascade"]),
         "alpha_mult": trial.suggest_categorical("alpha_mult", [1.0, 10.0, 30.0]), "alpha_init": trial.suggest_categorical("alpha_init", [0.0, 0.1])}
    return p


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=40); ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--final", type=int, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--study", default="qpikan_v1"); ap.add_argument("--stage2", type=int, default=None,
                    help="stage-2 quantum NAS on this stage-1 trial's classical config (study qpikan_v1)")
    ap.add_argument("--fixed", default=None, help="final mode on a JSON param dict instead of a study trial (§38ad fallback)")
    ap.add_argument("--tag", default=None, help="output stem suffix for --fixed (head_boundary_v8_<tag>_s<seed>)")
    ap.add_argument("--space", default="v1", help="v1 (§38ac) | v2 (§43j: around the deployed recipe; the recipe is enqueued as trial 0)")
    a = ap.parse_args()
    if "QW_GPU_GB" in os.environ:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["QW_GPU_GB"]) / 120.0))
    d, sota, c, fm, edge, lab = data(); tr, Ytr = lab["rows_train"], lab["y_train"].astype(np.float64)
    shots = np.unique(d.shot[tr]); hold = np.random.default_rng(0).permutation(shots)[: len(shots) // 10]; hm = np.isin(d.shot[tr], hold)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.final is not None or a.fixed:                                           # retrain on all TRAIN shots -> VALID npz
        if a.fixed:
            p = json.loads(a.fixed)
        else:
            st = optuna.load_study(study_name=a.study, storage=DB); p = {**st.trials[a.final].params}
            if a.study != "qpikan_v1":                                       # stage-2 trial: merge its classical base
                p = {**json.loads(st.user_attrs["base"]), **p}
        p.setdefault("order", 4); p["w_pi"] = p.get("w_pi", 0.0) if p.pop("pi", False) else p.get("w_pi", 0.0)
        if os.environ.get("STRAT_ALL", "") == "1":                             # §41q final build: + every VALID / TEST frame
            vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz")
            tr, Ytr = np.concatenate([tr, vt["rows"]]), np.concatenate([Ytr, vt["y"].astype(np.float64)])
        va = lab["rows_valid"]; S = Setup(d, sota, c, fm, edge, tr, Ytr, None, None)
        net = train(p, S, a.epochs, log=f" final{a.final}"); net.eval()
        with torch.no_grad():
            xv = torch.tensor(np.clip((d.inputs(va, fm, edge) - S.mu) / S.sd, -10, 10), dtype=torch.float32, device="cuda")
            pv = torch.cat([net(xv[i:i + 16384]).float() for i in range(0, len(xv), 16384)]).cpu().numpy() * S.ysd + S.ymu
        stem = f"head_boundary_v8_{a.tag}" if a.fixed else f"head_boundary_v8_QK{a.final}"
        np.savez(fx.OUT / f"{stem}_s{a.seed}.npz", rows_valid=va, pred_valid=pv.astype(np.float32))
        torch.save({"net": net.state_dict(), "mu": S.mu, "sd": S.sd, "ymu": S.ymu, "ysd": S.ysd, "fm": fm, "edge": edge, "arch": p,
                    "cfg": {**c, "width": p["width"], "depth": p["depth"], "drop": p["drop"]}, "inputs": "magfix", "canon": {"1": 1, "2": 2}.get(os.environ.get("STRAT_CANON", ""), 0),
                    "magfix_name": os.environ.get("QW_MAGFIX", "X_magfix"), "thz_extra": os.environ.get("STRAT_THZ", "") == "1",
                    "all_data": os.environ.get("STRAT_ALL", "") == "1", "hfull": os.environ.get("STRAT_HFULL", "") == "1",
                    "fut": os.environ.get("STRAT_FUT", "") == "1", "thp": os.environ.get("STRAT_THP", ""),
                    "liphys": os.environ.get("STRAT_LIPHYS", "") == "1", "qphys": {"1": 1, "2": 2}.get(os.environ.get("STRAT_QPHYS", ""), 0),
                    "qfast": os.environ.get("QW_QFAST", "") == "1"}, fx.OUT / f"{stem}_s{a.seed}.pt")
        print(f"QK_FINAL_DONE {'fixed ' + a.tag if a.fixed else 'trial=' + str(a.final)} seed={a.seed}", flush=True); return
    fit = np.nonzero(~hm)[0]; fit = np.random.default_rng(1).choice(fit, int(a.frac * len(fit)), replace=False)
    S = Setup(d, sota, c, fm, edge, tr[fit], Ytr[fit], tr[hm], Ytr[hm])
    print(f"[qk] fit rows {len(fit)}, held-out TRAIN rows {int(hm.sum())} ({len(hold)} shots)", flush=True)
    st = optuna.create_study(study_name=a.study, storage=DB, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2 if a.space == "v1" else 3))
    if a.space == "v2" and not st.trials:
        st.enqueue_trial(BASE_V2)                                                  # baseline under the same protocol
    base = None
    if a.stage2 is not None:
        base = {**optuna.load_study(study_name="qpikan_v1", storage=DB).trials[a.stage2].params}
        base.setdefault("order", 4); base["w_pi"] = base.get("w_pi", 0.0) if base.pop("pi", False) else 0.0
        st.set_user_attr("base", json.dumps(base))
    def objective(trial):
        p = (space_v2(trial) if a.space == "v2" else space(trial)) if base is None else space_q(trial, base)
        print(f"[qk] trial {trial.number} {json.dumps(p)}", flush=True)
        net = train(p, S, a.epochs, trial=trial, log=f" t{trial.number}")
        r2, each, _ = evaluate(net, S); trial.set_user_attr("r2_each", each); trial.set_user_attr("alpha", float(net.alpha.item()))
        print(f"QK_TRIAL {trial.number} val R2 {r2:.5f} | " + " ".join(f"{v:.4f}" for v in each) + f" | alpha {net.alpha.item():+.4f}", flush=True)
        del net; torch.cuda.empty_cache(); return r2
    st.optimize(objective, n_trials=a.trials, catch=(torch.cuda.OutOfMemoryError,))
    b = st.best_trial; print(f"QK_BEST trial {b.number} val R2 {b.value:.5f} {json.dumps(b.params)}", flush=True)


if __name__ == "__main__":
    main()
