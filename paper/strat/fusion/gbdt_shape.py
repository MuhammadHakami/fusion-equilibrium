"""Doc 35 §45a: gradient-boosted-tree heads for the 7 consistency shape scalars (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li).

Why: the consistency term carries 0.00068 of our 0.0011 S gap to #1 and its limit is head accuracy (§44). A public competitor at rank 10
(Anulum, S 0.996359, consistency 0.98770 vs our 0.98630; docs/METHODS_REPORT.md) blends four gradient-boosted tree heads into its network
outputs at fixed weights - li 0.26, R_axis 0.14, kappa 0.09, lower X-point R 0.16. Our only tree test was q95 / betaN (§41v: trees much
weaker than the neural heads, blend +0.00005). Here CatBoost (GPU, depth 8, RMSE) on the deployed J1RL head input set (same pipeline as
gpu_build.heads_gpu), every 3rd TRAIN frame, one model per scalar; predictions on every VT frame (VALID + TEST, never used to fit).
Read-out per scalar: tree R2 alone, the deployed TRAIN-only head ensemble (F8 proportions) R2, and their blend with one weight w chosen
by 5-fold shot cross-fitting over VT (out-of-fold R2), shot-level pooled sums as the scorer.
PRE-DECLARED rule (2026-10-16, before any result; as §41s): adopt the tree blend for a scalar iff its out-of-fold gain >= +0.0003 R2;
adopted scalars -> all-data trees + an F-build (expected S effect = 0.20 x transfer 0.88 x mean-7 gain).

    .venv/bin/python paper/strat/fusion/gbdt_shape.py dump
    .venv-tdc/bin/python paper/strat/fusion/gbdt_shape.py fit [--all]
    .venv/bin/python paper/strat/fusion/gbdt_shape.py eval
    .venv/bin/python paper/strat/fusion/gbdt_shape.py dumppub           # J1RL inputs of every public test frame -> gbdt_shape_Xpub.npy
    .venv-tdc/bin/python paper/strat/fusion/gbdt_shape.py litree        # all-data li tree -> litree.npz {public, data} for QW_LITREE
    QW_FOLD=<fold> .venv/bin/python paper/strat/fusion/gbdt_shape.py dumppub         # Final phase: gbdt_shape_X<tag>.npy for that fold
    QW_FOLD=<fold> .venv-tdc/bin/python paper/strat/fusion/gbdt_shape.py litreepred  # saved all-data tree (no refit) -> litree_<tag>.npz
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "fusion"
import os as _os
V2 = _os.environ.get("GBDT_V2", "") == "1"                                     # §45c: all TRAIN frames + depth 10 / lr 0.05 / 6000 it
DATA = OUT / ("gbdt_shape_data_full.npz" if V2 else "gbdt_shape_data.npz"); SUF = "_v2" if V2 else ""
FTAG = "public" if _os.environ.get("QW_FOLD", "diii_d_public_test") == "diii_d_public_test" else "private"   # fold.py convention
XPUB = OUT / ("gbdt_shape_Xpub.npy" if FTAG == "public" else f"gbdt_shape_X{FTAG}.npy")       # Development file names unchanged
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
STEM = "head_boundary_v8_J1RL_s1"                       # input pipeline of the deployed J1RL shape heads
if not (OUT / f"{STEM}.pt").exists():                     # from-scratch rebuild (only the all-data J1RLA heads): identical input
    STEM = "head_boundary_v8_J1RLA_s1"                    # pipeline (same fm / edge / feature-block flags, checked 2026-10-16)
if not (OUT / f"{STEM}.pt").exists():                     # F15K / F16K rebuild (J1RLHA only): identical pipeline - same flags and fm, and
    STEM = "head_boundary_v8_J1RLHA_s1"                   # head_inputs equal to J1RL_s1's on 20,000 rows (max |diff| 0.0, 2026-10-17)
STEM = _os.environ.get("QW_TREE_STEM", STEM)             # explicit anchor (repro/fusion/train_all.sh sets J1RLHA_s1)
DEPLOYED = ("head_boundary_v8_J1RL_s1*3+head_boundary_v8_J1QPIpi_s1+head_boundary_v8_J1QPIpi_s2+head_boundary_v8_J1QH_s1+"
            "head_boundary_v8_J1QH_s2+head_boundary_v8_J1QH_s3")   # TRAIN-only heads in F8's proportions (s44_transfer_vt)


def head_inputs(d, rows, stem):
    """The raw input matrix a head checkpoint reads (the dd chain of gpu_build.heads_gpu, without its normalisation)."""
    import torch
    import gpu_build as GB
    ck = torch.load(GB.fx.OUT / f"{stem}.pt", map_location="cpu", weights_only=False)
    dd = GB.fixed(d, GB.mf_name(stem, ck)) if ck.get("inputs") == "magfix" else d
    dd = GB.canonical(dd, ck.get("canon") == 2) if ck.get("canon") else dd
    for key, tag in (("hfull", None), ("thz_extra", "thz"), ("ctx", "ctx"), ("thzs", "thzs"), ("liphys", "liphys"), ("qshape", "qshape"),
                     ("fut", "fut")):
        if ck.get(key):
            dd = GB.with_hfull(dd) if tag is None else GB.with_hfull(dd, tag)
    if ck.get("thp"):
        dd = GB.with_hfull(dd, "thp_" + ck["thp"])
    if ck.get("qphys"):
        dd = GB.with_hfull(dd, "qphys2" if ck.get("qphys") == 2 else "qphys")
    return dd.inputs(rows, ck["fm"], ck["edge"]).astype(np.float32)


def dump() -> None:
    import fx
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    d = sota.Data(); tr = np.load(OUT / "boundary_labels_v8.npz"); vt = np.load(OUT / "boundary_labels_v8_vt.npz")
    step = 1 if V2 else 3; rtr = tr["rows_train"][::step]; ytr = tr["y_train"][::step, :7]
    keep = d.split[vt["rows"]] != 0; rvt = vt["rows"][keep]; yvt = vt["y"][keep, :7]
    Xtr = head_inputs(d, rtr, STEM); Xvt = head_inputs(d, rvt, STEM)
    np.savez(DATA, Xtr=Xtr, ytr=ytr.astype(np.float32), rtr=rtr, Xvt=Xvt, yvt=yvt.astype(np.float32), rvt=rvt, shot_vt=d.shot[rvt])
    print(f"GBDT_SHAPE_DUMP_DONE train {len(rtr)} vt {len(rvt)} features {Xtr.shape[1]}", flush=True)


def fit(all_rows: bool) -> None:
    from catboost import CatBoostRegressor
    z = np.load(DATA); X, Y = z["Xtr"], z["ytr"]
    if all_rows:
        X, Y = np.concatenate([X, z["Xvt"][::3]]), np.concatenate([Y, z["yvt"][::3]])
    out = {}
    for j, nm in enumerate(CONS):
        ok = np.isfinite(Y[:, j]); t = time.time()
        m = CatBoostRegressor(iterations=6000 if V2 else 3000, depth=10 if V2 else 8, learning_rate=0.05 if V2 else 0.08, loss_function="RMSE",
                              task_type="GPU", devices="0", gpu_ram_part=0.2, random_seed=0, verbose=1000, allow_writing_files=False)
        m.fit(X[ok], Y[ok, j]); out[nm] = m.predict(z["Xvt"]).astype(np.float32)
        m.save_model(str(OUT / f"gbdt_shape_{nm}{SUF}{'_all' if all_rows else ''}.cbm"))
        print(f"[gbdt] {nm}: {time.time() - t:.0f}s", flush=True)
    np.savez(OUT / f"gbdt_shape_pred{SUF}{'_all' if all_rows else ''}.npz", **out); print("GBDT_SHAPE_FIT_DONE", flush=True)


def evaluate() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                      # before fx (which hides the GPU) and gpucap's probe
    import torch
    import fx
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB
    z = np.load(DATA); rows, Y, sh = z["rvt"], z["yvt"].astype(np.float64), z["shot_vt"]; g = np.load(OUT / f"gbdt_shape_pred{SUF}.npz")
    d = sota.Data(); dep = GB.heads_gpu(d, rows, DEPLOYED)[:, :7]
    us = np.random.default_rng(0).permutation(np.unique(sh)); W = np.r_[0:0.55:0.05]; res = {}
    for j, nm in enumerate(CONS):
        y = Y[:, j]; t = g[nm].astype(np.float64); p0 = dep[:, j]; ok = np.isfinite(y) & np.isfinite(p0) & np.isfinite(t)
        r2 = lambda p, m: 1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)          # noqa: E731
        oof = np.full(len(y), np.nan); ws = []
        for f in range(5):
            te = np.isin(sh, us[f::5]); trm = ok & ~te
            wb = max(W, key=lambda w: r2((1 - w) * p0 + w * t, trm)); oof[te] = ((1 - wb) * p0 + wb * t)[te]; ws.append(float(wb))
        res[nm] = {"tree": r2(t, ok), "deployed": r2(p0, ok), "oof_blend": r2(oof, ok), "w_folds": ws,
                   "curve": {round(float(w), 2): r2((1 - w) * p0 + w * t, ok) for w in W},
                   "corr_err": float(np.corrcoef((t - y)[ok], (p0 - y)[ok])[0, 1])}
        gain = res[nm]["oof_blend"] - res[nm]["deployed"]; res[nm]["gain"] = gain; res[nm]["adopt"] = bool(gain >= 3e-4)
        print(f"[gbdt] {nm:>7}: tree {res[nm]['tree']:.5f} deployed {res[nm]['deployed']:.5f} OOF blend {res[nm]['oof_blend']:.5f} "
              f"({gain:+.5f}) w {ws} err corr {res[nm]['corr_err']:.3f}" + ("  ADOPT" if res[nm]["adopt"] else ""), flush=True)
    mg = float(np.mean([res[n]["gain"] for n in CONS])); res["mean7_gain"] = mg; res["S_effect"] = 0.20 * 0.88 * mg
    print(f"[gbdt] mean-7 OOF gain {mg:+.6f} -> expected S {0.20 * 0.88 * mg:+.6f}", flush=True)
    (OUT / f"gbdt_shape_eval{SUF}.json").write_text(json.dumps(res, indent=1)); print("GBDT_SHAPE_EVAL_DONE", flush=True)


def dumppub() -> None:
    import fx
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    import submit_public as SP
    d = sota.Data(); pub = SP.PublicData(d.meta); X = head_inputs(pub, np.arange(len(pub.X)), STEM)
    np.save(XPUB, X); np.save(OUT / "gbdt_shape_ndata.npy", np.array([len(d.shot)]))
    print(f"GBDT_SHAPE_DUMPPUB_DONE public frames {len(X)} features {X.shape[1]}", flush=True)


def litree() -> None:
    """§45a adopted scalar li: all-data tree (TRAIN every 3rd + VT every 3rd) for the public build; VT keys from the TRAIN-only tree."""
    from catboost import CatBoostRegressor
    z = np.load(DATA); j = CONS.index("li"); X = np.concatenate([z["Xtr"], z["Xvt"][::3]]); Y = np.concatenate([z["ytr"], z["yvt"][::3]])[:, j]
    ok = np.isfinite(Y); m = CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.08, loss_function="RMSE", task_type="GPU", devices="0",
                                               gpu_ram_part=0.15, random_seed=0, verbose=1000, allow_writing_files=False)
    m.fit(X[ok], Y[ok]); m.save_model(str(OUT / "gbdt_shape_li_all.cbm"))
    if not XPUB.exists():                                                      # from-scratch rebuild: litreepred writes the fold file
        print("GBDT_LITREE_DONE model saved (no test-fold inputs yet)", flush=True); return
    pub = m.predict(np.load(XPUB)).astype(np.float64); n = int(np.load(OUT / "gbdt_shape_ndata.npy")[0]) if (OUT / "gbdt_shape_ndata.npy").exists() else 0
    data = np.full(n, np.nan)                                                  # VT keys (TRAIN-only tree, evaluation only)
    if n and (OUT / "gbdt_shape_pred.npz").exists():
        data[z["rvt"]] = np.load(OUT / "gbdt_shape_pred.npz")["li"]
    np.savez(OUT / "litree.npz", public=pub, data=data); print(f"GBDT_LITREE_DONE public {len(pub)} finite {np.isfinite(pub).all()}", flush=True)


def litreepred() -> None:
    """Final phase: the SAVED all-data li tree (gbdt_shape_li_all.cbm, no refit) on a test fold's frames -> litree_<tag>.npz (the data key
    is copied from litree.npz, so the QW_LITREE hook reads it unchanged)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(); m.load_model(str(OUT / "gbdt_shape_li_all.cbm")); pub = m.predict(np.load(XPUB)).astype(np.float64)
    dst = OUT / ("litree.npz" if FTAG == "public" else f"litree_{FTAG}.npz")
    if FTAG == "public" and dst.exists():
        ref = np.load(dst)["public"]; print(f"[litree] public re-prediction vs litree.npz: max |diff| {np.abs(pub - ref).max():.2e}", flush=True); return
    data = np.load(OUT / "litree.npz")["data"] if (OUT / "litree.npz").exists() else np.full(0, np.nan)   # the hook reads "public" on test folds
    np.savez(dst, public=pub, data=data); print(f"GBDT_LITREE_PRED_DONE {FTAG} frames {len(pub)} finite {np.isfinite(pub).all()}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    {"dump": dump, "fit": lambda: fit("--all" in sys.argv), "eval": evaluate, "dumppub": dumppub, "litree": litree, "litreepred": litreepred}[sys.argv[1]]()
