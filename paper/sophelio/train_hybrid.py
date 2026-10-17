"""Sophelio Fusion Equilibrium Challenge (DIII-D, Ch1) — hybrid arms on the starter kit's OWN
UNetLite flux-map baseline and OWN pipeline (baseline/sophelio-starter: local parquet loader,
magnetics features interpolated to efit_times, shot-level 70/10/20 split with RandomState(42),
StandardScaler, Adam 1e-3 + ReduceLROnPlateau, MSE; RidgeCV for the q95/betaN scalars), scored
with the vendored official scorer (local_score.py functions: composite S, R2_psi, R2_{q95,betaN},
D_LCFS, Consistency) on the held-out test shots.

Register insertion: UNetLite.fc1 (n_features -> 512 latent) is wrapped with the zero-init gated
Bottleneck on the 512-d latent: vanilla | mlp30 | mr30ryd (parameter-matched cores).
Usage: uv run paper/sophelio/train_hybrid.py --variant mr30ryd --seed 1337 [--n-shots 200 --epochs 50]
Fixed across arms: shot sample + split (seed 42), features, epochs, batch, lr. Training seed varies.
"""
import argparse, json, os, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
STARTER = ROOT / "baseline" / "sophelio-starter"
os.chdir(STARTER)
for p in (STARTER, STARTER / "fusion_scoring", ROOT / "paper"):
    sys.path.insert(0, str(p))
import numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from experiments import (EFIT_SCALAR_LABELS, ExperimentConfig, ScalarTargetModels, build_feature_matrix,  # noqa: E402
                         compute_metrics, interpolate_magnetics_to_efit, interpolate_thomson_to_efit, load_shots)
from experiments_torch import UNetLite, predict_pytorch, train_pytorch_model  # noqa: E402
import local_score as LS  # noqa: E402  (vendored official scorer glue: build_reference, score_shot, ...)
from metrics import Accum, finalize_machine  # noqa: E402
from common import CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from model import Bottleneck, MLPCore, _count, build_config_core, parse_config  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

DATA_DIR = ROOT / "data" / "sophelio" / "fusion-equilibrium-challenge"
DATA_SEED = 42  # shot sample + split, identical across arms and seeds (starter default)
Q95, BETAN = "efit_q95", "efit_beta_n"
INCLUDE_THOMSON = False  # set from --thomson in main()
N_WORKERS = int(os.environ.get("SOPHELIO_SCORE_WORKERS", "4"))  # the official scorer is CPU contour code; parallel over shots


def _ref_one(args):
    psi, key, cache_dir, R, Z, mask_coarse, mask_f = args
    cache = pathlib.Path(cache_dir) / f"ref_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["contours"]), z["cons"], z["cmask"], float(z["rgeo"])
    ref = LS.build_reference(psi, R, Z, mask_coarse, mask_f)
    np.savez(cache, contours=np.array(ref[0], dtype=object), cons=ref[1], cmask=ref[2], rgeo=ref[3])
    return ref


def _score_one(args):
    gt, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means = args
    return LS.score_shot(gt, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means)


def wrap(model: UNetLite, variant: str, q_layers: int = 4) -> UNetLite:
    if variant == "vanilla":
        return model
    ch = model.fc1.out_features  # 512
    if variant == "mlp30":
        core, out = MLPCore(ch, 30, _count(MultiRegister(ch, 10, 3, q_layers, "hea"))), 30
    elif variant == "mr30ryd":
        core, out = MultiRegister(ch, 10, 3, q_layers, "rydberg"), 30
    elif parse_config(variant) is not None:                       # wave-G grammar (q30-link=shared, ...)
        core, out = build_config_core(parse_config(variant), ch, q_layers)
    else:
        raise ValueError(variant)
    model.fc1 = nn.Sequential(model.fc1, Bottleneck(ch, core, out))
    return model


def split_by_shot(shot_ids, config):
    """Exactly FusionDataPipeline.prepare's shot-level split."""
    uniq = np.unique(shot_ids)
    n_test = max(1, int(len(uniq) * config.test_size)); n_val = max(1, int(len(uniq) * config.val_size))
    n_train = len(uniq) - n_test - n_val
    perm = np.random.RandomState(config.random_state).permutation(uniq)
    tr, va, te = set(perm[:n_train]), set(perm[n_train:n_train + n_val]), set(perm[n_train + n_val:])
    return (np.isin(shot_ids, list(tr)), np.isin(shot_ids, list(va)), np.isin(shot_ids, list(te)), sorted(te), sorted(va))


def shot_predictions(shot, model, scaler, scal_models, device):
    """Per-frame predictions aligned to efit_times: NaN where features are non-finite (scorer -> mean)."""
    psi_gt = shot["efit_psirz"]; T = psi_gt.shape[0]
    feats = interpolate_magnetics_to_efit(shot)
    if INCLUDE_THOMSON:
        th = interpolate_thomson_to_efit(shot)
        feats = np.hstack([feats, th]) if th is not None else feats
    feats = feats[:T]
    valid = np.isfinite(feats).all(axis=1)
    psirz = np.full((T, 65, 65), np.nan, dtype=np.float64); q95 = np.full(T, np.nan); betaN = np.full(T, np.nan)
    if valid.any():
        Xs = scaler.transform(feats[valid]).astype(np.float32)
        psirz[valid] = predict_pytorch(model, Xs, device)
        S = scal_models.predict(Xs)
        q95[valid] = S[:, EFIT_SCALAR_LABELS.index(Q95)]; betaN[valid] = S[:, EFIT_SCALAR_LABELS.index(BETAN)]
    gt = {"psi": psi_gt.astype(np.float32), "q95": np.asarray(shot["scalars"][Q95], dtype=np.float64)[:T],
          "betaN": np.asarray(shot["scalars"][BETAN], dtype=np.float64)[:T]}
    return gt, {"psirz": psirz, "q95": q95, "betaN": betaN}


def official_score(gts, preds, ref_cache_dir: pathlib.Path):
    """Replicates local_score.main's aggregation with the vendored scorer functions."""
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]; mask_coarse = mask["mask_coarse"].astype(bool); mask_f = mask_coarse.astype(np.float64)
    refs = []; psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    ref_cache_dir.mkdir(parents=True, exist_ok=True)
    # references (GT contours + derived scalars) in parallel across shots, cached per shot
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # fresh interpreters: no copy of the parent's arrays
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        refs = list(ex.map(_ref_one, [(gt["psi"], key, str(ref_cache_dir), R, Z, mask_coarse, mask_f) for gt, key in gts]))
    for (gt, key), ref in zip(gts, refs):
        g = gt["psi"].astype(np.float64); fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(gt[name], dtype=np.float64); v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]; cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
    ref_stats = dict(psi_sum=psi_sum, psi_sumsq=psi_sumsq, psi_n=psi_n, scal_sum=scal_sum, scal_sumsq=scal_sumsq, scal_n=scal_n,
                     cons_sum=cons_sum, cons_sumsq=cons_sumsq, cons_n=cons_n)
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi, np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))
    totals = {s: 0.0 for s in PSI_SIGNS}
    for (gt, _), p in zip(gts, preds):
        rr, _, _ = LS.psi_residuals(gt["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS: totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    acc = Accum(); acc.psi_sign = psi_sign
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        parts = list(ex.map(_score_one, [(gt, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means) for (gt, _), ref, p in zip(gts, refs, preds)]))
    for part in parts:
        acc.add(part)
    res = finalize_machine(acc, ref_stats)
    out = {k: float(res[k]) for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency") if k in res} | {"psi_sign": int(psi_sign)}
    if "r2_cons_each" in res:  # per-functional Consistency R2 (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li)
        out["r2_cons_each"] = {k: (None if v is None else float(v)) for k, v in res["r2_cons_each"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vanilla"); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-shots", type=int, default=200); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "sophelio"))
    ap.add_argument("--score-split", default="test", choices=["test", "val"], help="which held-out shots the official scorer sees (val = HPO objective)")
    ap.add_argument("--tag", default="", help="suffix for the output JSON name (HPO trials)")
    ap.add_argument("--thomson", action="store_true", help="round-2 trunk: add the starter's Thomson-scattering features (default off = starter default)")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global INCLUDE_THOMSON; INCLUDE_THOMSON = a.thomson
    config = ExperimentConfig(data_source="local", local_data_dir=DATA_DIR, n_shots=a.n_shots, random_state=DATA_SEED,
                              epochs=a.epochs, batch_size=a.batch, learning_rate=a.lr, device="cuda", output_dir=out, include_thomson=a.thomson)
    t0 = time.time()
    shots = load_shots(config)
    X, Y, S, shot_ids = build_feature_matrix(shots, config.include_thomson)
    tr, va, te, test_shot_idx, val_shot_idx = split_by_shot(shot_ids, config)
    if a.score_split == "val":
        test_shot_idx = val_shot_idx          # HPO: the official scorer runs on the VALIDATION shots; test untouched
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = (scaler.transform(X[m]).astype(np.float32) for m in (tr, va, te))
    Ytr, Yva, Yte, Str = Y[tr], Y[va], Y[te], S[tr]
    n_loaded, n_feat = len(shots), int(Xtr.shape[1])
    keep = set(int(k) for k in test_shot_idx)
    shots = [sh for sh in shots if sh['shot_index'] in keep]  # only the test shots are needed later (per-shot scoring)
    del X, Y, S; import gc; gc.collect()                        # ~4 GB less resident memory on the shared GB10
    print(f"[sophelio] {a.variant} s{a.seed}: shots {n_loaded} rows train/val/test {tr.sum()}/{va.sum()}/{te.sum()} "
          f"features {n_feat} test shots {len(test_shot_idx)} load {time.time()-t0:.0f}s", flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = wrap(UNetLite(n_feat), a.variant); n_params = _count(model)
    t1 = time.time()
    res = train_pytorch_model(model, Xtr, Ytr, Xva, Yva, config)
    train_s = time.time() - t1; model, device = res["model"], res["device"]
    Y_pred = predict_pytorch(model, Xte, device); m = compute_metrics(Yte, Y_pred)
    scal_models = ScalarTargetModels(list(EFIT_SCALAR_LABELS)).fit(Xtr, Str)
    by_idx = {s["shot_index"]: s for s in shots}
    gts, preds = [], []
    t2 = time.time()
    for k in test_shot_idx:
        gt, pr = shot_predictions(by_idx[int(k)], model, scaler, scal_models, device)
        gts.append((gt, int(k))); preds.append(pr)
    official = official_score(gts, preds, out / "ref_cache")
    result = dict(variant=a.variant, seed=a.seed, thomson=a.thomson, n_shots=n_loaded, n_test_shots=len(test_shot_idx), n_features=n_feat,
                  epochs=a.epochs, batch=a.batch, lr=a.lr, num_params=n_params, train_seconds=train_s, score_seconds=time.time() - t2,
                  test_R2=float(m["R2"]), test_MSE=float(m["MSE"]), test_SSIM=(float(m["SSIM"]) if m.get("SSIM") is not None else None),
                  official=official, final_val_loss=float(res["val_losses"][-1]))
    result["score_split"] = a.score_split
    (out / f"{a.variant}_s{a.seed}{a.tag}.json").write_text(json.dumps(result, indent=1))
    torch.save({"model_state_dict": model.state_dict()}, out / f"{a.variant}_s{a.seed}.pt")
    for k, v in result.items(): print(f"{k}:{' ' * max(1, 16 - len(k))}{v}")


if __name__ == "__main__":
    main()
