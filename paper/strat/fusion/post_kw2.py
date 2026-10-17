"""Doc 35 §42y: second kappa pass on a built public submission (label-free realisation lever, like §42u's li2).
Re-applies the kappa warp to the saved psi toward the head kappa, then the li warp twice (li, li2) so li stays on its head target.
Diagnostics: |kappa(psi) - kappa_head| and |li(psi) - li_head| before / after (rms, frames > 0.05). q95 / betaN copied unchanged.
    .venv/bin/python paper/strat/fusion/post_kw2.py --src F7 --tag F7K
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile

import numpy as np
import torch

import fx

H_F7 = "+".join(f"head_boundary_v8_J1RLA_s{s}*3" for s in range(1, 9)) + "+head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2"


def gaps(gp, p, kt, lt):
    o = gp.g.derive(p); return (o["kappa"] - kt).cpu().numpy(), (o["li"] - lt).cpu().numpy()


def report(nm, g):
    g = np.nan_to_num(g); big = np.abs(g) > 0.05
    return f"{nm} rms {np.sqrt(np.mean(g ** 2)):.4f}, frames > 0.05: {big.sum()}"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--heads", default=H_F7); ap.add_argument("--kcap", type=float, default=0.10)
    ap.add_argument("--trs", type=float, default=0.0, help="§45t: tri_shift cap in m after the li passes (0 = off)"); a = ap.parse_args()
    sys.path.insert(0, str(fx.ROOT / "paper" / "sophelio")); import sota
    sota.DEV = torch.device("cuda"); import gpu_build as GB, axis_shift as AX, submit_public as SP, fold as FD
    from gpu_post import GPUPost
    t = time.time(); d = sota.Data(); pub = SP.PublicData(d.meta); shots = pub.shots(); rows = np.arange(len(pub.X))
    h = GB.heads_gpu(pub, rows, a.heads); kt_all = torch.tensor(h[:, 2], device="cuda", dtype=torch.float64)
    lt_all = torch.tensor(h[:, 6], device="cuda", dtype=torch.float64); tt_all = torch.tensor(h[:, 3:5], device="cuda", dtype=torch.float64)
    src = np.load(fx.ROOT / "runs" / "sophelio_sota" / f"submission_{a.src}" / f"diii_d_{FD.TAG}_test.npz")
    psi = np.concatenate([src[f"shot_{s:04d}_psirz"] for s in shots]); assert len(psi) == len(rows), (len(psi), len(rows))
    R, Z, mask = AX.grid(); gp = GPUPost(R, Z, mask); out = np.empty_like(psi); g0k, g0l, g1k, g1l = [], [], [], []
    for i in range(0, len(psi), 2048):
        p = torch.tensor(psi[i:i + 2048], device="cuda", dtype=torch.float64); kt = kt_all[i:i + 2048]; lt = lt_all[i:i + 2048]
        k, l = gaps(gp, p, kt, lt); g0k.append(k); g0l.append(l)
        q = gp.kappa_warp(p, kt, cap=a.kcap); q = gp.li_warp(q, lt, cap=0.9); q = gp.li_warp(q, lt, cap=0.9)
        if a.trs > 0:
            q = gp.tri_shift(q, tt_all[i:i + 2048], cap=a.trs)
        k, l = gaps(gp, q, kt, lt); g1k.append(k); g1l.append(l); out[i:i + 2048] = q.cpu().numpy().astype(np.float32)
    for nm, g in (("kappa before", g0k), ("kappa after", g1k), ("li before", g0l), ("li after", g1l)):
        print("[kw2] " + report(nm, np.concatenate(g)), flush=True)
    od = fx.ROOT / "runs" / "sophelio_sota" / f"submission_{a.tag}"; od.mkdir(parents=True, exist_ok=True); preds = {}; i = 0
    for s in shots:
        T = int((pub.shot == s).sum()); preds[f"shot_{s:04d}_psirz"] = out[i:i + T]; i += T
        preds[f"shot_{s:04d}_q95"] = src[f"shot_{s:04d}_q95"]; preds[f"shot_{s:04d}_betaN"] = src[f"shot_{s:04d}_betaN"]
        if not all(np.isfinite(preds[f"shot_{s:04d}_{k}"]).all() for k in ("psirz", "q95", "betaN")):
            raise RuntimeError(f"shot {s}: non-finite prediction")
    npz = od / f"diii_d_{FD.TAG}_test.npz"; np.savez_compressed(npz, **preds)
    with zipfile.ZipFile(od / "submission.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(npz, arcname=npz.name)
    info = {"args": vars(a), "post_pass": "kw2 (kappa warp re-applied, cap %.2f) + li x2" % a.kcap + (" + trs cap %.3f" % a.trs if a.trs > 0 else ""), "frames": int(len(rows)), "seconds": round(time.time() - t)}
    (od / "build.json").write_text(json.dumps(info, indent=1)); print("POST_KW2 built", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
