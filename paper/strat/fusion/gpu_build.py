"""GPU Fusion pipeline for the once-only local TEST score and the public leaderboard file, from one code path:
trunk ensemble (entry or psi specialists) -> shape heads -> GPU post-processing (gpu_post, validated = CPU warps under
the official scorer) -> official scorer (TEST) or the submission npz (public, float32).

    .venv/bin/python paper/strat/fusion/gpu_build.py --split test --tag <name> [--psi entry|psiA] [--heads STEMS] [--kw JSON]
    .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag <name> ...   -> runs/sophelio_sota/submission_<tag>/
--heads: '+'-joined head stems (default the 6-seed v8 heads); --kw: per-step warp overrides (gpu_pipe.post_all).
TEST writes runs/strat/fusion/gpu_test_<tag>.json and refuses to overwrite it (TEST once).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402,I001
import fold as FD  # noqa: E402,I001
import axis_shift as AX  # noqa: E402
import gpu_pipe as GP  # noqa: E402
from gpu_post import GPUPost  # noqa: E402
from head_boundary import Net  # noqa: E402

SOTA = fx.ROOT / "runs" / "sophelio_sota"
PSI = {"entry": None, "psiA": ["s38c_PSI-A.pt", "s38c_PSI-A-s1.pt", "s38c_PSI-A-s3.pt"],
       "psiF": ["s38i_PSI-F.pt", "s38i_PSI-F-s1.pt", "s38i_PSI-F-s3.pt"],
       "psiT": ["s41f_PSI-T.pt", "s41f_PSI-T-s1.pt", "s41f_PSI-T-s3.pt"],
       "psiTA": ["s41f_PSI-TA.pt", "s41f_PSI-TA-s1.pt", "s41f_PSI-TA-s3.pt"],
       "psiF6": ["s38i_PSI-F.pt", "s38i_PSI-F-s1.pt", "s38i_PSI-F-s3.pt", "s38i_PSI-F-s4.pt", "s38i_PSI-F-s5.pt", "s38i_PSI-F-s6.pt"]}                # §38i: corrected inputs; §41f: + Thomson on fixed Z / R
QF = ["s38g_QF.pt", "s38g_QF-s1.pt", "s38g_QF-s3.pt"]                                       # §38g: corrected inputs
PSI["psiFI"] = ["IPF_s2.pt", "IPF_s1.pt", "IPF_s3.pt"]                                         # §41q: PSI-F on the organizers' Ip fix
PSI["psiFA"] = ["PSIFA_s2.pt", "PSIFA_s1.pt", "PSIFA_s3.pt"]                                   # §41q final: PSI-F on all 7,041 shots
PSI["psiFC"] = ["PCN_s2.pt", "PCN_s1.pt"]                                                       # §41q: canonical PSI-F (mephiles, TRAIN only)
PSI["psiOOF"] = [f"PSIO_k{k}.pt" for k in range(5)]                                    # §41r: PSI-F leaving out TRAIN fold k (out-of-fold psi)
PSI["psiPQ2A"] = ["PQ2A_s2.pt", "PQ2A_s1.pt", "PQ2A_s3.pt"]                                   # §42h: psiFA + qphys2
PSI["PQmix"] = ["PQ2.pt", "LS_ctrl.pt"]                                                  # §42m: recipe mix (VT analogue of PQ2A + psiFA)
PSI["psiMixA"] = ["PQ2A_s2.pt", "PQ2A_s1.pt", "PQ2A_s3.pt", "PSIFA_s2.pt", "PSIFA_s1.pt", "PSIFA_s3.pt"]   # §42m production mix
for _t in ("LS_ctrl", "LS_w3", "LS_w30", "K128", "K256", "PQ2", "PQL10", "PQ2s2", "PQ2s3", "PQ2H"):                               # §42b level-set / §42c k trunks (single seed)
    PSI[_t] = [f"{_t}.pt"]
IPFIX_PREFIX = ("IPF_", "QFI_")              # sota trunks trained with QW_MAGFIX=X_ipfix (heads record it as ck["magfix_name"])
V8 = "+".join(f"head_boundary_v8_s{s}" for s in range(1, 7))


def fixed(d, name="X_magfix"):
    """Shallow copy of the data object reading a corrected magnetics block (§38g X_magfix, §41q X_ipfix); cached on d per name."""
    if not hasattr(d, "_fixed"):
        d._fixed = {}
    if name not in d._fixed:
        import copy
        import inputs_fix
        d._fixed[name] = inputs_fix.use(copy.copy(d), public=not hasattr(d, "split"), name=name)
    return d._fixed[name]


_CANON = {}


def canonical(dd, indicator=False):
    """Ip-sign canonical view of a data object (§41q canon.py; indicator: + the sign column); cached per object."""
    key = ("canon", indicator, id(dd))
    if key not in _CANON:
        import copy
        import canon
        _CANON[key] = (dd, canon.use(copy.copy(dd), public=not hasattr(dd, "split"), indicator=indicator))
    return _CANON[key][1]


def with_hfull(dd, kind="hfull"):
    """Data view with extra inputs appended (hfull_inputs.py: §41q history or the thz block); cached per object."""
    key = (kind, id(dd))
    if key not in _CANON:
        import copy
        import hfull_inputs
        import qphys_inputs
        if kind == "vac":                                                     # §45j: vacuum-field inputs
            import vac_inputs
            _CANON[key] = (dd, vac_inputs.use(copy.copy(dd)))
        else:
            _CANON[key] = (dd, qphys_inputs.use(copy.copy(dd), ratios=kind == "qphys2") if kind.startswith("qphys")   # §42a / §42e
                           else hfull_inputs.use(copy.copy(dd), public=not hasattr(dd, "split"), kind=kind))
    return _CANON[key][1]


def mf_name(stem, ck):
    """Corrected-input set a checkpoint was trained on."""
    return ck.get("magfix_name") or ("X_ipfix" if os.path.basename(str(stem)).startswith(IPFIX_PREFIX) else "X_magfix")


def heads_gpu(d, rows, stems):
    """Mean of the head checkpoints (stem.pt) on rows, GPU; each head reads the inputs it was trained on."""
    if stems.startswith("route:"):                                           # §45o: "route:<normal stems>|<reversed-Ip stems>" per shot
        import canon
        normal, _, rev = stems[len("route:"):].partition("|")
        s = np.asarray(canon.shot_sign(d, public=not hasattr(d, "split")))[rows]
        pn = heads_gpu(d, rows, normal); pr = heads_gpu(d, rows, rev)
        return np.where((s < 0)[:, None], pr, pn)
    if stems.startswith("oracle:") or stems.startswith("xporacle:"):            # §43e diagnostic: true labels as head targets (VT rows only)
        kind, _, fb = stems.partition(":"); p = heads_gpu(d, rows, fb)
        if kind == "oracle":
            z = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); pos = {r: i for i, r in enumerate(z["rows"])}; y = z["y"].astype(np.float64)
            t = np.array([y[pos[r]] if r in pos else np.full(y.shape[1], np.nan) for r in rows])[:, :p.shape[1]]
        else:
            t = np.load(fx.OUT / "xp_labels.npz")["y"][rows].astype(np.float64)[:, :p.shape[1]]
        return np.where(np.isfinite(t), t, p)
    out, wts = [], []
    for st in stems.split("+"):
        st, _, w = st.partition("*"); wts.append(float(w or 1))               # §42: "stem*3" = weight 3 (no repeated inference)
        ck = torch.load(fx.OUT / f"{st}.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
        dd = fixed(d, mf_name(st, ck)) if ck.get("inputs") == "magfix" else d
        dd = canonical(dd, ck.get("canon") == 2) if ck.get("canon") else dd
        dd = with_hfull(dd) if ck.get("hfull") else dd
        dd = with_hfull(dd, "thz") if ck.get("thz_extra") else dd
        dd = with_hfull(dd, "ctx") if ck.get("ctx") else dd
        dd = with_hfull(dd, "thzs") if ck.get("thzs") else dd
        dd = with_hfull(dd, "liphys") if ck.get("liphys") else dd                    # §42p
        dd = with_hfull(dd, "qshape") if ck.get("qshape") else dd                    # §42s
        dd = with_hfull(dd, "fut") if ck.get("fut") else dd                          # §41t
        dd = with_hfull(dd, "thp_" + ck["thp"]) if ck.get("thp") else dd            # §41r
        dd = with_hfull(dd, "qphys2" if ck.get("qphys") == 2 else "qphys") if ck.get("qphys") else dd   # §42a / §42e
        dd = with_hfull(dd, "vac") if ck.get("vac") else dd                         # §45j
        dd = with_hfull(dd, "subf") if ck.get("subf") else dd                       # §45m
        if "arch" in ck:                                     # §38ac: quantum-classical PI-KAN head (hpo_qpikan.py --final)
            import hpo_qpikan as HQ
            net = HQ.build(ck["arch"], len(ck["mu"]), len(ck["ymu"]))
        else:
            net = Net(len(ck["mu"]), len(ck["ymu"]), c["width"], c["depth"], c["drop"]).cuda()
        net.load_state_dict(ck["net"]); net.eval()
        x = torch.tensor(np.clip((dd.inputs(rows, ck["fm"], ck["edge"]) - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32, device="cuda")
        tk, _, ts = os.environ.get("QW_TTA", "").partition(":")                 # §43d: test-time input-noise averaging (off by default)
        with torch.no_grad():
            p = torch.cat([net(x[i:i + 32768]) for i in range(0, len(x), 32768)])
            if tk:
                g = torch.Generator(device="cuda").manual_seed(0)
                for _ in range(int(tk)):
                    p += torch.cat([net(x[i:i + 32768] + float(ts) * torch.randn(x[i:i + 32768].shape, device="cuda", generator=g))
                                    for i in range(0, len(x), 32768)])
                p /= int(tk) + 1
            p = p.cpu().numpy()
        nm_ = ck.get("n_main", p.shape[1]); p = p[:, :nm_] * ck["ysd"][:nm_] + ck["ymu"][:nm_]   # §41q: drop aux outputs
        out.append(np.exp(p) if ck.get("qb_log") else p)                    # §41q: log-target q / b heads
    res = np.average(np.stack(out), axis=0, weights=np.asarray(wts)).astype(np.float64)
    lt = os.environ.get("QW_LITREE", "")                                     # §45a: li of the shape heads blended with the GBDT li tree
    if lt and "head_boundary" in stems and res.shape[1] >= 7:                # ("path.npz:w"; keys public / data = row-aligned predictions)
        path, _, w = lt.partition(":"); t = np.load(path)["data" if hasattr(d, "split") else "public"][rows]; ok = np.isfinite(t)
        res[ok, 6] = (1 - float(w)) * res[ok, 6] + float(w) * t[ok]
    return res


def predict(d, shots, a, sota, ES, arms=None, sink=None, psi_sets=None):
    """Per shot: post-processed psi (T, 65, 65) float64, q95, betaN. arms = [(name, kw JSON)] -> {name: per-shot dict}
    from one shared trunk / head inference (§41y VT-wide arms differ only in the post-processing). sink(arm, shots, psi_gpu, q, b):
    streaming consumer per row group - nothing is kept (the 15:45 OOM held every arm's psi for 193k frames)."""
    sel = json.loads((SOTA / "ens_select_mlp.json").read_text())
    ent = [ES.member(str(fx.TRUNKS / m), d, shots) for m in sel["members"]]

    def members(paths):                                     # each trunk reads the inputs it was trained on
        out = []
        for p in paths:
            if (SOTA / p).exists():
                cf = torch.load(SOTA / p, map_location="cpu", weights_only=False)["cfg"]
                dd = fixed(d, mf_name(p, {})) if cf.get("inputs") == "magfix" else d
                dd = canonical(dd) if cf.get("canon") else dd
                dd = with_hfull(dd, "qphys2" if int(cf.get("qphys", 0)) == 2 else "qphys") if cf.get("qphys") else dd   # §42h
                dd = with_hfull(dd, "hfull") if cf.get("hfull") else dd                                                   # §42o
                out.append(ES.member(str(SOTA / p), dd, shots))
        assert out, paths
        return out
    base_ = lambda n_: n_.split("~")[0]                                                       # §42k: "<set>~s" = shrunk coefficients
    flags_ = lambda n_: (n_.split("~")[1] if n_ and "~" in n_ else "")                        # §42m: "<set>~g" = axis-guarded
    psims = ({n_: (ent if PSI[base_(n_)] is None else members(PSI[base_(n_)])) for n_ in psi_sets} if psi_sets   # §42c: several psi sets,
             else {None: ent if PSI[a.psi] is None else members(PSI[a.psi])})                 # one shared head / entry inference (sink only)
    guard = None
    if a.guard:                                     # §42m: '<fallback set>:<thr m>' - frames whose raw O-point is > thr from the head
        gs_, gt_ = a.guard.split(":")               # axis take the fallback trunk's coefficients (label-free; VT shot 2996: Z -0.90 m)
        guard = (ent if PSI[gs_] is None else members(PSI[gs_]), float(gt_)); guard_n = [0, 0]
    shrink = None
    if a.shrink:                                                                              # §42k: c' = mean_true + slope (c - mean_pred)
        z_ = np.load(a.shrink); k0 = 16; sl = np.clip(z_["slope"], 0, 1); sl[:k0] = 1.0
        shrink = (sl, z_["mean_true"], z_["mean_pred"])
    qfm = None if a.qb in ("entry", "qh50") else members(QF)
    comps, mean = d.meta["comps"][:ES.KMAX].numpy().astype(np.float64), d.meta["mean"].numpy().astype(np.float64)
    H = sota.HEADS; iq, ib, ir_, iz_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis"))
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
    single = arms is None; arms = [(None, a.kw)] if single else arms
    cfg = []
    for nm_, kj in arms:
        kw = json.loads(kj) if kj else {}; cfg.append((nm_, kw, tuple(kw.pop("_steps", ("shift", "bw", "kw", "li")))))
    out, group = {nm_: {} for nm_, _, _ in cfg}, []
    def flush():
        rows = np.concatenate([r for _, r in group]); ph = torch.tensor(heads_gpu(d, rows, a.heads), device="cuda")
        sc = np.concatenate([np.mean([m[s][1] for m in ent], 0) for s, _ in group])
        xh = torch.tensor(heads_gpu(d, rows, a.xpheads), device="cuda", dtype=torch.float64) if getattr(a, "xpheads", "") else None   # §42t
        if qfm is not None:                                  # q95 / betaN (and their axis heads) from the QF trunks
            sq = np.concatenate([np.mean([m[s][1] for m in qfm], 0) for s, _ in group])
            for j in (iq, ib):
                sc[:, j] = sq[:, j] if a.qb == "qf" else 0.5 * (sc[:, j] + sq[:, j])
        if a.qb == "qh50":                                   # §38s: q95 / betaN from the QH heads, 50/50 with the entry
            qh = heads_gpu(d, rows, a.qbheads)
            w_ = a.qbw; sc[:, iq] = (1 - w_) * sc[:, iq] + w_ * qh[:, 0]; sc[:, ib] = (1 - w_) * sc[:, ib] + w_ * qh[:, 1]   # §42a: head weight
        if getattr(a, "qbclamp", False):                     # §42z: q95 / betaN clamped to the labelled range (qb_clamp.json)
            B = json.loads((fx.OUT / "qb_clamp.json").read_text())
            sc[:, iq] = np.clip(sc[:, iq], *B["q95"]); sc[:, ib] = np.clip(sc[:, ib], *B["betaN"])
        axv = {"trunk": sc[:, [ir_, iz_]], "blend": 0.5 * (sc[:, [ir_, iz_]] + ph[:, :2].cpu().numpy()),
               "head": ph[:, :2].cpu().numpy()}[a.axis]                              # §38aa: head = the J1 heads' axis
        ax = torch.tensor(axv, device="cuda", dtype=torch.float64)
        for ps_, psim in psims.items():
            pc = np.concatenate([np.mean([m[s][0] for m in psim], 0) for s, _ in group])
            if shrink is not None and (ps_ is None or "s" in flags_(ps_)):
                pc = pc.copy(); sl, mt, mp = shrink; pc[:, :64] = mt + sl * (pc[:, :64] - mp)
            psi = torch.tensor((pc @ comps + mean).reshape(-1, 65, 65), device="cuda")
            if guard is not None and (ps_ is None or "g" in flags_(ps_)):
                off = torch.cat([(gp.axis(psi[c_]) - ph[c_, :2].to(psi.dtype)).abs().amax(1) for c_ in GP.chunks(len(psi), 4096)])
                bad = (off > guard[1]).cpu().numpy(); guard_n[0] += int(bad.sum()); guard_n[1] += len(bad)
                if bad.any():
                    pf = np.concatenate([np.mean([m[s][0] for m in guard[0]], 0) for s, _ in group])
                    pc = pc.copy(); pc[bad] = pf[bad]; psi = torch.tensor((pc @ comps + mean).reshape(-1, 65, 65), device="cuda")
                    print(f"[guard] {ps_}: {int(bad.sum())} / {len(bad)} frames -> fallback (O-point > {guard[1]} m from the head axis; "
                          f"max {float(off.max()):.3f} m); total {guard_n[0]} / {guard_n[1]}", flush=True)
            if os.environ.get("QW_HIORACLE"):                # §45z diagnostic (VT rows only): add the TRUE PCA coefficients [k, 256)
                k_ = int(os.environ["QW_HIORACLE"]); ct = (d.psi_of(rows).reshape(len(rows), -1).astype(np.float64) - mean) @ comps.T
                psi = psi + torch.tensor(((ct[:, k_:] - pc[:, k_:]) @ comps[k_:]).reshape(-1, 65, 65), device="cuda", dtype=psi.dtype)
            for nm_, kw, steps in cfg:
                pg = GP.post_all(gp, psi.clone(), ax, ph[:, 7:], ph[:, 2], ph[:, 6], kw=copy.deepcopy(kw), steps=steps, tri_t=ph[:, 3:5],
                                 heads=ph, xh=xh)
                if sink is not None:
                    sink(nm_ if ps_ is None else f"{ps_}_{nm_}", [s for s, _ in group], pg, sc[:, iq].copy(), sc[:, ib].copy()); del pg
                    continue
                pp = pg.cpu().numpy(); del pg; i = 0
                pp = pp if single else pp.astype(np.float32)               # multi-arm: submission precision
                for s, r in group:
                    j = slice(i, i + len(r)); i += len(r); out[nm_][s] = (pp[j], sc[j, iq], sc[j, ib])
            del psi
        group.clear()
    for s in shots:
        group.append((s, np.nonzero(d.shot == s)[0]))
        if sum(len(r) for _, r in group) > 20000:
            flush()
    if group:
        flush()
    return out[None] if single else out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["test", "public", "sample", "vt"], required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--psi", default="entry", choices=list(PSI)); ap.add_argument("--heads", default=V8); ap.add_argument("--kw", default="")
    ap.add_argument("--qb", default="entry", choices=["entry", "qf", "qf50", "qh50"]); ap.add_argument("--axis", default="trunk", choices=["trunk", "blend", "head"])
    ap.add_argument("--qbheads", default="+".join(f"head_qb_QH_s{s}" for s in range(1, 7)))
    ap.add_argument("--qbw", type=float, default=0.5, help="qh50: weight of the q/b heads vs the entry scalars (§42a)")
    ap.add_argument("--shrink", default="", help="§42k: npz with per-component slope / mean_true / mean_pred (coef_reliability.npz)")
    ap.add_argument("--guard", default="", help="§42m: '<fallback psi set>:<thr m>' axis guard (psi set '<name>~g', or --psi alone)")
    ap.add_argument("--xpheads", default="", help="§42t: '+'-joined X-point head stems (head_xp_*) for the xh step")
    ap.add_argument("--qbclamp", action="store_true", help="§42z: clamp q95 / betaN to the labelled range (qb_clamp.json)")
    ap.add_argument("--vt-n", type=int, default=0)
    ap.add_argument("--vt-shots", default="", help="--split vt smoke: comma-separated shot indices")
    ap.add_argument("--psis", default="", help="--split vt: comma-separated psi sets scored from one shared inference (arms <set>_<kw>)")
    ap.add_argument("--kws", default="", help="--split vt: JSON {arm: kw} run from one shared inference")
    a = ap.parse_args(); t = time.time()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    import sota
    sota.DEV = torch.device("cuda")
    import ens_select as ES
    d = sota.Data(); L = sota.LABELS
    if a.split == "sample":                                  # VALID sample: twin + official parts, paired vs the frozen base
        import pickle
        import strategies as ST
        shots, grp = ST.sample(); pe, nm = ST.load_entry(); P = predict(d, shots, a, sota, ES)
        R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask)
        psi = torch.tensor(np.concatenate([P[s][0] for s in shots]), device="cuda")
        q, b = np.concatenate([P[s][1] for s in shots]), np.concatenate([P[s][2] for s in shots])
        tw = GP.parts_all(gp, psi, q, b, d, shots); off = np.cumsum([0] + [len(P[s][1]) for s in shots])
        sl = {s: slice(off[i], off[i + 1]) for i, s in enumerate(shots)}; flat = psi.cpu().numpy().reshape(len(psi), -1)
        po = fx.frame_parts(d, shots, lambda s: (flat[sl[s]], q[sl[s]], b[sl[s]]), np.asarray(nm["c_mean"]))
        base = pickle.load(open(fx.OUT / "parts_GPU-frozen-official.pkl", "rb")); g = np.asarray([grp[s] for s in shots])
        sb = np.stack([fx.shot_sums(base[s]) for s in shots]); rep = {"args": vars(a)}
        for nmx, parts in (("twin", tw), ("official", po)):
            sn = np.stack([fx.shot_sums(parts[s]) for s in shots]); c = fx.composite(sn, nm); bt = fx.paired_boot(sb, sn, nm, groups=g)
            rep[nmx] = dict(c, boot=bt)
            print(f"GPU_SAMPLE {a.tag} {nmx}: S {c['S']:.5f} delta {bt['delta']:+.5f} [{bt['lo']:+.5f}, {bt['hi']:+.5f}] r2_psi {c['r2_psi']:.6f} "
                  f"r2_qb {c['r2_qb']:.5f} dlcfs {c['dlcfs']:.5f} cons {c['consistency']:.5f} | {time.time() - t:.0f}s", flush=True)
        pickle.dump(po, open(fx.OUT / f"parts_GPUS-{a.tag}.pkl", "wb"))
        (fx.OUT / f"gpu_sample_{a.tag}.json").write_text(json.dumps(rep, indent=1, default=float))
        return
    if a.split == "vt":                                      # §41y: all VALID + TEST shots, GPU twin parts (paired arm comparisons)
        import pickle
        import strategies as ST
        shots = [int(s) for s in np.unique(d.shot[d.split != 0])]; _, nm = ST.load_entry()
        shots = shots[:a.vt_n] if a.vt_n else shots                                  # smoke test
        shots = [int(x) for x in a.vt_shots.split(",")] if a.vt_shots else shots
        arms = list(json.loads(a.kws).items()) if a.kws else [(a.tag, a.kw)]           # --kws {"VB": {...}, ...}: one inference
        arms = [(n_, json.dumps(k_) if isinstance(k_, dict) else k_) for n_, k_ in arms]
        R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); psis = a.psis.split(",") if a.psis else None
        PARTS = {(n_ if psis is None else f"{ps_}_{n_}"): {} for ps_ in (psis or [None]) for n_, _ in arms}

        RC = {}

        def sink(tag, gshots, psi_g, q, b):                                       # score each row group as it is produced
            key = tuple(gshots)
            if key not in RC:                                                     # reference contours once per row group
                RC.clear(); RC[key] = GP.refs(d, gshots)
            PARTS[tag].update(GP.parts_all(gp, psi_g, q, b, d, gshots, ref=RC[key]))
        predict(d, shots, a, sota, ES, arms=arms, sink=sink, psi_sets=psis); print(f"[vt] inference + post-processing + parts {time.time() - t:.0f}s", flush=True)
        for tag, tw in PARTS.items():
            sn = np.stack([fx.shot_sums(tw[s]) for s in shots]); c = fx.composite(sn, nm)
            pickle.dump(tw, open(fx.OUT / f"parts_VT-{tag}.pkl", "wb"))
            print(f"GPU_VT {tag} twin: S {c['S']:.5f} r2_psi {c['r2_psi']:.6f} r2_qb {c['r2_qb']:.5f} dlcfs {c['dlcfs']:.5f} "
                  f"cons {c['consistency']:.5f} | {len(shots)} shots {time.time() - t:.0f}s", flush=True)
        return
    if a.split == "test":
        import train_hybrid as TH
        out_f = fx.OUT / f"gpu_test_{a.tag}.json"; assert not out_f.exists(), "TEST already scored once for this tag"
        shots = d.shots_of(2); P = predict(d, shots, a, sota, ES); gts, preds = [], []
        for s in shots:
            rows = np.nonzero(d.shot == s)[0]
            gts.append(({"psi": d.psi_of(rows).reshape(-1, 65, 65).astype(np.float32), "q95": d.S[rows, L.index("efit_q95")].astype(np.float64),
                         "betaN": d.S[rows, L.index("efit_beta_n")].astype(np.float64)}, d.meta["names"][s]))
            preds.append({"psirz": P[s][0], "q95": P[s][1], "betaN": P[s][2]})
        r = TH.official_score(gts, preds, SOTA / "ref_cache")
        out_f.write_text(json.dumps({"args": vars(a), "test": r}, indent=1, default=float))
        print(f"GPU_TEST {a.tag}: S {r['S']:.6f} r2_psi {r['r2_psi']:.6f} r2_qb {r['r2_qb']:.6f} dlcfs {r['dlcfs']:.5f} "
              f"cons {r['consistency']:.5f} | {time.time() - t:.0f}s", flush=True)
        return
    import submit_public as SP
    pub = SP.PublicData(d.meta); shots = pub.shots()
    pub._thz = np.load(fx.ROOT / "data" / "sophelio_proc" / f"thz_{FD.TAG}.npy", mmap_mode="r") if a.psi in ("psiT", "psiTA") else None   # §41f
    P = predict(pub, shots, a, sota, ES)
    out = SOTA / f"submission_{a.tag}"; out.mkdir(parents=True, exist_ok=True); preds = {}
    for s in shots:
        T = int((pub.shot == s).sum())
        preds[f"shot_{s:04d}_psirz"] = np.asarray(P[s][0], np.float32).reshape(T, 65, 65)
        preds[f"shot_{s:04d}_q95"] = np.asarray(P[s][1], np.float32); preds[f"shot_{s:04d}_betaN"] = np.asarray(P[s][2], np.float32)
        if not all(np.isfinite(preds[f"shot_{s:04d}_{k}"]).all() for k in ("psirz", "q95", "betaN")):
            raise RuntimeError(f"shot {s}: non-finite prediction")
    npz = out / f"diii_d_{FD.TAG}_test.npz"; np.savez_compressed(npz, **preds)
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(npz, arcname=npz.name)
    info = {"args": vars(a), "dtype": "float32", "shots": len(shots), "frames": int(len(pub.X)), "npz_bytes": npz.stat().st_size,
            "seconds": round(time.time() - t)}
    (out / "build.json").write_text(json.dumps(info, indent=1)); print("GPU_PUBLIC built", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
