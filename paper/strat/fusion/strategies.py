"""§37 fusion step 3: score strategies against the submitted entry on the registered stratified VALID sample.

    .venv/bin/python paper/strat/fusion/strategies.py --run G-qb G-coef ...   ->  runs/strat/fusion/parts_<name>.pkl
    .venv/bin/python paper/strat/fusion/strategies.py --report                ->  runs/strat/fusion/strategies.json

Sample = runs/strat/fusion/strata.json (9 shot strata x 20 shots, seed 0, fixed before any strategy was scored).
Every strategy is fit on TRAIN only; no VALID-based choice is made (all hyper-parameters fixed in advance).
Composite S uses the full-VALID normalisers (parts_entry.pkl), so per-stratum S reads as "the fold's S if every shot
looked like this stratum"; deltas get a paired bootstrap over shots (stratified by stratum for the overall number).

Strategies (all on top of the entry = ens_select_mlp.json, 7 members, sigma 0):
  G-qb / G-qb-noIp       CatBoost residual correction of q95 / betaN (gbdt.py), with / without re-aligned Ip features
  G-coef / G-coef-noIp   CatBoost residual correction of PCA coefficients 0-15 (psi re-scored with the official code)
  G-both                 G-qb + G-coef
  STACK                  per-output NNLS weights over the 7 members (bands 0-4, 4-16, 16-64, q95, betaN), fit on TRAIN
  R-Ip-s<k>              single model: t5-ew2 retrained with the re-aligned Ip (retrain_ip.py), vs trunk1_t5-ew2_seed<k>
  OLD-s<k>               the matching single model without the fix (paired control for R-Ip-s<k>)
  CPU-ship / CPU-fix / CPU-fixhist   matched CPU short-budget (6 epoch) single models of the t5-ew2 config with the
                         shipped Ip / re-aligned Ip / re-aligned Ip + true-ms Ip history (retrain_cpu.py)
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

import fx

PH = ("ramp-up", "flat-top", "ramp-down")


def load_entry():
    with open(fx.OUT / "parts_entry.pkl", "rb") as f:
        e = pickle.load(f)
    return e["parts"], e["norms"]


def sample():
    st = json.loads((fx.OUT / "strata.json").read_text())
    shots, grp = [], {}
    for name, v in st["strata"].items():
        for s in v["shots"]:
            shots.append(int(s)); grp[int(s)] = name
    return shots, grp


def entry_preds(d, sota):
    z = np.load(fx.OUT / "pred_valid.npz"); M = len(fx.ENTRY)
    comps = d.meta["comps"][:64].numpy().astype(np.float64); mean = d.meta["mean"].numpy().astype(np.float64)
    H = sota.HEADS

    def f(s, w=None):
        cs = np.stack([z[f"{i}_{s}_c"] for i in range(M)]).astype(np.float64)
        ss = np.stack([z[f"{i}_{s}_s"] for i in range(M)]).astype(np.float64)
        return cs, ss
    return f, comps, mean, H


def run(name: str) -> None:
    sota, d = fx.data()
    shots, _ = sample()
    parts_e, nm = load_entry()
    f, comps, mean, H = entry_preds(d, sota)
    iq, ib = H.index("efit_q95"), H.index("efit_beta_n")
    if name.startswith("G-"):
        tag = "_noip" if name.endswith("noIp") else ""
        g = np.load(fx.OUT / f"gbdt{tag}.npz"); pos = {int(r): i for i, r in enumerate(g["rows"])}
    do_qb = name in ("G-qb", "G-qb-noIp", "G-both"); do_coef = name in ("G-coef", "G-coef-noIp", "G-both")
    if name == "G-both":
        g = np.load(fx.OUT / "gbdt.npz"); pos = {int(r): i for i, r in enumerate(g["rows"])}
    if name in ("G-qb", "G-qb-noIp"):                      # psi unchanged: only the scalar residuals move
        out = {}
        for s in shots:
            rows = fx.valid_rows(d, s); ii = [pos[int(r)] for r in rows]
            cs, ss = f(s); sc = ss.mean(0)
            q = sc[:, iq] + g["q95"][ii]; b = sc[:, ib] + g["betaN"][ii]
            p = dict(parts_e[s]); p["q_res"] = (p["q_gt"] - q) ** 2; p["b_res"] = (p["b_gt"] - b) ** 2
            out[s] = p
    else:
        if name == "STACK":
            w = json.loads((fx.OUT / "stack_weights.json").read_text())

        def pred(s):
            rows = fx.valid_rows(d, s)
            cs, ss = f(s)
            if name == "STACK":
                c = np.zeros(cs.shape[1:])
                for band, (a, b) in {"0-4": (0, 4), "4-16": (4, 16), "16-64": (16, 64)}.items():
                    c[:, a:b] = np.tensordot(np.array(w[band]), cs[:, :, a:b], axes=(0, 0))
                q = np.tensordot(np.array(w["q95"]), ss[:, :, iq], axes=(0, 0)); b_ = np.tensordot(np.array(w["betaN"]), ss[:, :, ib], axes=(0, 0))
                return c @ comps + mean, q, b_
            c = cs.mean(0); sc = ss.mean(0); q, b_ = sc[:, iq], sc[:, ib]
            if do_coef or do_qb:
                ii = [pos[int(r)] for r in rows]
                if do_coef:
                    c = c.copy(); c[:, :16] += g["coef16"][ii]
                if do_qb:
                    q = q + g["q95"][ii]; b_ = b_ + g["betaN"][ii]
            return c @ comps + mean, q, b_

        if name.startswith("CPU-"):
            import retrain_cpu as RC
            arm = name.split("-", 1)[1]
            dd = RC.DataCols(arm in ("fix", "fixhist"), arm == "fixhist")
            mp_ = fx.member_predict(fx.OUT / f"cpu_{arm}_e6_s1.pt", dd, {s: fx.valid_rows(d, s) for s in shots})

            def pred(s):  # noqa: F811
                c, sc = mp_[s]
                return c.astype(np.float64) @ comps + mean, sc[:, iq].astype(np.float64), sc[:, ib].astype(np.float64)
        if name.startswith("R-Ip-s") or name.startswith("OLD-s"):
            k = int(name.split("-s")[-1])
            if name.startswith("R-Ip"):
                import retrain_ip as RI
                dd = RI.DataIp(); path = fx.OUT / f"t5-ew2-ipfix_seed{k}.pt"
            else:
                dd = d; path = fx.TRUNKS / f"trunk1_t5-ew2_seed{k}.pt"
            mp_ = fx.member_predict(path, dd, {s: fx.valid_rows(d, s) for s in shots})

            def pred(s):  # noqa: F811
                c, sc = mp_[s]
                return c.astype(np.float64) @ comps + mean, sc[:, iq].astype(np.float64), sc[:, ib].astype(np.float64)
        out = fx.frame_parts(d, shots, pred, np.asarray(nm["c_mean"]))
    with open(fx.OUT / f"parts_{name}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    st = np.stack([fx.shot_sums(out[s]) for s in shots]); se = np.stack([fx.shot_sums(parts_e[s]) for s in shots])
    print(f"[strat] {name}: sample S {fx.composite(st, nm)['S']:.5f} vs entry {fx.composite(se, nm)['S']:.5f}", flush=True)


def phase_mask(d, s, ph):
    pos = d.edge_feats()[fx.valid_rows(d, s), 1]
    return {"ramp-up": pos < 0.1, "flat-top": (pos >= 0.1) & (pos <= 0.9), "ramp-down": pos > 0.9}[ph]


def compare(parts_a, parts_b, shots, grp, nm, d, B=2000):
    g = np.array([grp[s] for s in shots])
    A = np.stack([fx.shot_sums(parts_a[s]) for s in shots]); Bm = np.stack([fx.shot_sums(parts_b[s]) for s in shots])
    ca, cb = fx.composite(A, nm), fx.composite(Bm, nm)
    res = dict(overall=dict(S_ref=ca["S"], S_new=cb["S"], **fx.paired_boot(A, Bm, nm, g, B),
                            d_terms={k: -(cb["loss"][k] - ca["loss"][k]) for k in ca["loss"]},
                            d_cons_each={k: cb["cons_each"][k] - ca["cons_each"][k] for k in fx.CONS},
                            d_r2_q95=cb["r2_q95"] - ca["r2_q95"], d_r2_betaN=cb["r2_betaN"] - ca["r2_betaN"]))
    res["strata"] = {}
    for name in sorted(set(g)):
        ii = np.nonzero(g == name)[0]
        res["strata"][name] = dict(S_ref=fx.composite(A[ii], nm)["S"], **fx.paired_boot(A[ii], Bm[ii], nm, None, B))
    res["phases"] = {}
    for ph in PH:
        Ap = np.stack([fx.shot_sums(parts_a[s], phase_mask(d, s, ph)) for s in shots])
        Bp = np.stack([fx.shot_sums(parts_b[s], phase_mask(d, s, ph)) for s in shots])
        res["phases"][ph] = dict(S_ref=fx.composite(Ap, nm)["S"], **fx.paired_boot(Ap, Bp, nm, g, B))
    return res


def report() -> None:
    sota, d = fx.data()
    shots, grp = sample()
    parts_e, nm = load_entry()
    out = {"sample_shots": len(shots), "entry_sample": fx.composite(np.stack([fx.shot_sums(parts_e[s]) for s in shots]), nm)}
    # sampling uncertainty of the entry's own S (bootstrap over all VALID shots): how much fold-to-fold noise to expect
    allv = d.shots_of(1); E = np.stack([fx.shot_sums(parts_e[s]) for s in allv]); rng = np.random.default_rng(0)
    bs = [fx.composite(E[rng.integers(0, len(allv), len(allv))], nm)["S"] for _ in range(1000)]
    out["entry_fullvalid_S_boot95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pairs = {"G-qb": "entry", "G-qb-noIp": "entry", "G-coef": "entry", "G-coef-noIp": "entry", "G-both": "entry",
             "STACK": "entry", "R-Ip-s1": "OLD-s1", "R-Ip-s2": "OLD-s2", "OLD-s1": "entry",
             "CPU-fix": "CPU-ship", "CPU-fixhist": "CPU-fix", "CPU-fixhist ": "CPU-ship"}
    for new, ref in pairs.items():
        new_ = new.strip(); pf = fx.OUT / f"parts_{new_}.pkl"
        if not pf.exists():
            continue
        pa = parts_e if ref == "entry" else pickle.load(open(fx.OUT / f"parts_{ref}.pkl", "rb"))
        pb = pickle.load(open(pf, "rb"))
        new = new_
        out[f"{new} vs {ref}"] = r = compare(pa, pb, shots, grp, nm, d)
        o = r["overall"]
        print(f"[report] {new} vs {ref}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})}", flush=True)
        for k, v in r["strata"].items():
            print(f"    {k:26s} ref {v['S_ref']:.5f} delta {v['delta']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]")
        for k, v in r["phases"].items():
            print(f"    phase {k:20s} ref {v['S_ref']:.5f} delta {v['delta']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]")
    (fx.OUT / "strategies.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs="*", default=[]); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    for n in a.run:
        run(n)
    if a.report:
        report()
