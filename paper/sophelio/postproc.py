"""Sophelio post-processing on saved trunks (VALID only): per-shot temporal smoothing of the predicted psi / scalars and
trunk ensembles, scored by the vendored official scorer on the NAS's fixed 40-shot VALID subset.

    .venv/bin/python paper/sophelio/postproc.py --trunks runs/nas/sophelio/trunks/trunk1_t0.pt [...] --sigmas 0 1 2 4

Smoothing uses model outputs only (never labels): a Gaussian over neighbouring frames of the same shot, with the kernel
renormalised at the shot edges. Frames are the challenge's own time base; features already include symmetric lags.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sota  # noqa: E402


def load(path: str, data: "sota.Data"):
    ck = torch.load(path, map_location=sota.DEV, weights_only=False); c = ck["cfg"]
    m = sota.Hybrid(len(ck["mu"]), c["k"], c["width"], c["depth"], c["drop"], c["quantum"], c["qcfg"], c["k_lo"],
                    c["q_layers"], c.get("k_hi")).to(sota.DEV)
    m.load_state_dict(ck["model"]); m.eval()
    comps, mean = data.meta["comps"][:c["k"]].to(sota.DEV), data.meta["mean"].to(sota.DEV)

    @torch.no_grad()
    def predict(rows):
        xs = torch.tensor(np.clip((data.inputs(rows, ck["fm"], ck["cfg"].get("edge", False)) - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32, device=sota.DEV)
        pc, ps = m(xs)
        return (pc @ comps + mean).cpu().numpy(), ps.cpu().numpy() * ck["ssd"] + ck["smu"]
    return predict


def smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 (time), edge-renormalised."""
    if sigma <= 0:
        return a
    r = int(np.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    n = len(a); out = np.empty_like(a, dtype=np.float64)
    for t in range(n):
        lo, hi = max(0, t - r), min(n, t + r + 1)
        w = k[lo - t + r: hi - t + r]
        out[t] = np.tensordot(w, a[lo:hi], axes=(0, 0)) / w.sum()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--trunks", nargs="+", required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 4]); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", type=int, default=1)
    a = ap.parse_args()
    import train_hybrid as TH
    data = sota.Data(); L = sota.LABELS; H = sota.HEADS
    shots = data.shots_of(a.split); pick = [shots[i] for i in np.random.default_rng(1).permutation(len(shots))[:a.n]]
    preds = [load(p, data) for p in a.trunks]
    cache = []
    for s in pick:
        rows = np.nonzero(data.shot == s)[0]
        outs = [f(rows) for f in preds]
        psi = np.mean([o[0] for o in outs], axis=0); sc = np.mean([o[1] for o in outs], axis=0)
        gt = ({"psi": data.psi_of(rows).reshape(-1, 65, 65).astype(np.float32),
               "q95": data.S[rows, L.index("efit_q95")].astype(np.float64), "betaN": data.S[rows, L.index("efit_beta_n")].astype(np.float64)},
              data.meta["names"][s])
        cache.append((gt, psi, sc))
    for sg in a.sigmas:
        gts, pr = [], []
        for gt, psi, sc in cache:
            p2, s2 = smooth(psi, sg), smooth(sc, sg)
            gts.append(gt); pr.append({"psirz": p2.reshape(-1, 65, 65), "q95": s2[:, H.index("efit_q95")], "betaN": s2[:, H.index("efit_beta_n")]})
        r = TH.official_score(gts, pr, sota.ROOT / "runs" / "sophelio_sota" / "ref_cache")
        print(f"[postproc] trunks={len(a.trunks)} sigma={sg}: S {r['S']:.4f} psi {r['r2_psi']:.5f} qb {r['r2_qb']:.4f} "
              f"dlcfs {r['dlcfs']:.4f} cons {r['consistency']:.4f} {({k: round(v, 3) for k, v in r['r2_cons_each'].items()})}", flush=True)


if __name__ == "__main__":
    main()
