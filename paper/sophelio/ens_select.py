"""Sophelio: greedy forward ensemble selection (Caruana et al. 2004, with replacement) over saved trunks — VALID only.

    .venv/bin/python paper/sophelio/ens_select.py [--glob 'runs/nas/sophelio/trunks/trunk1_t*.pt'] [--n-select 100]
        ->  runs/sophelio_sota/ens_select.json

Members are averaged in the shared PCA basis (coefficients zero-padded to 256; the basis rows are shared, so this is the
psi average) and their scalar heads averaged. Selection score = official S on a fixed 100-shot VALID subset; the
selected ensemble (and its sigma = 1 temporal smoothing) is then confirmed on ALL VALID shots. TEST is not touched.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import postproc as PP  # noqa: E402
import sota  # noqa: E402

KMAX = 256


def member(path: str, data, shots) -> dict:
    ck = torch.load(path, map_location=sota.DEV, weights_only=False); c = ck["cfg"]
    m = sota.Hybrid(len(ck["mu"]), c["k"], c["width"], c["depth"], c["drop"], c["quantum"], c["qcfg"], c["k_lo"],
                    c["q_layers"], c.get("k_hi")).to(sota.DEV)
    m.load_state_dict(ck["model"]); m.eval()
    out = {}
    with torch.no_grad():
        for s in shots:
            rows = np.nonzero(data.shot == s)[0]
            xs = torch.tensor(np.clip((data.inputs(rows, ck["fm"], ck["cfg"].get("edge", False), thz=bool(ck["cfg"].get("thz"))) - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32, device=sota.DEV)
            pc, ps = m(xs)
            pad = torch.zeros(len(rows), KMAX, device=sota.DEV); pad[:, :pc.shape[1]] = pc
            out[s] = (pad.cpu().numpy(), ps.cpu().numpy() * ck["ssd"] + ck["smu"])
    del m; torch.cuda.empty_cache()
    return out


def score(data, shots, members: list[dict], sigma: float = 0.0) -> dict:
    import train_hybrid as TH
    comps, mean = data.meta["comps"][:KMAX].numpy(), data.meta["mean"].numpy()
    L, H = sota.LABELS, sota.HEADS
    gts, preds = [], []
    for s in shots:
        rows = np.nonzero(data.shot == s)[0]
        pc = np.mean([mb[s][0] for mb in members], axis=0); sc = np.mean([mb[s][1] for mb in members], axis=0)
        psi, sc = PP.smooth(pc @ comps + mean, sigma), PP.smooth(sc, sigma)
        gts.append(({"psi": data.psi_of(rows).reshape(-1, 65, 65).astype(np.float32),
                     "q95": data.S[rows, L.index("efit_q95")].astype(np.float64),
                     "betaN": data.S[rows, L.index("efit_beta_n")].astype(np.float64)}, data.meta["names"][s]))
        preds.append({"psirz": psi.reshape(-1, 65, 65), "q95": sc[:, H.index("efit_q95")], "betaN": sc[:, H.index("efit_beta_n")]})
    return TH.official_score(gts, preds, sota.ROOT / "runs" / "sophelio_sota" / "ref_cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=[str(sota.ROOT / "runs/nas/sophelio/trunks/trunk1_t*.pt")])
    ap.add_argument("--tag", default="", help="suffix of the output JSON (several pools)")
    ap.add_argument("--n-select", type=int, default=100); ap.add_argument("--max-size", type=int, default=8)
    ap.add_argument("--top", type=int, default=8, help="greedy pool = the top singles (each scoring pass is ~1-2 min)")
    a = ap.parse_args()
    data = sota.Data()
    valid = data.shots_of(1)
    sel = [valid[i] for i in np.random.default_rng(7).permutation(len(valid))[:a.n_select]]
    paths = sorted({p for g in a.glob for p in glob.glob(g)})
    mem = {p: member(p, data, valid) for p in paths}
    single = {p: score(data, sel, [mem[p]])["S"] for p in paths}
    for p, v in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"[ens] single {pathlib.Path(p).name}: S_sel {v:.4f}", flush=True)
    pool = sorted(single, key=single.get, reverse=True)[:a.top]
    chosen = [pool[0]]; best = single[chosen[0]]
    while len(chosen) < a.max_size:
        trial = {p: score(data, sel, [mem[q] for q in chosen + [p]])["S"] for p in pool}
        p, v = max(trial.items(), key=lambda kv: kv[1])
        if v <= best + 1e-5:
            break
        chosen.append(p); best = v
        print(f"[ens] + {pathlib.Path(p).name} -> S_sel {v:.4f} ({len(chosen)} members)", flush=True)
    full = {sg: score(data, valid, [mem[q] for q in chosen], sg) for sg in (0.0, 1.0)}
    res = dict(members=[pathlib.Path(p).name for p in chosen], S_select=best, n_select=len(sel), n_valid=len(valid),
               valid_full={str(sg): {k: v for k, v in r.items() if k != "per_shot"} for sg, r in full.items()},
               singles={pathlib.Path(p).name: v for p, v in single.items()})
    (sota.ROOT / "runs" / "sophelio_sota" / f"ens_select{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    for sg, r in full.items():
        print(f"[ens] selected {res['members']} sigma={sg}: FULL VALID S {r['S']:.4f} cons {r['consistency']:.4f} "
              f"psi {r['r2_psi']:.5f} qb {r['r2_qb']:.4f} dlcfs {r['dlcfs']:.4f}", flush=True)


if __name__ == "__main__":
    main()
