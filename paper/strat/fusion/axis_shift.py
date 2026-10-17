"""Doc 35 §37n: shift the predicted plasma core so the scorer's O-point lands on the model's own axis heads (CPU).

Inference only, no fitted parameters. Per frame: phi = AXIS_SIGN * psi_pred; x_ax = the scorer's find_o_point estimate;
psi'(x) = psi(x - delta * g(x)), g = exp(-|x - x_ax|^2 / (2 SIGMA^2)) (zero at the LCFS, so boundary terms stay put),
bicubic; delta += target - estimate(psi') for N_IT iterations, |delta| <= 2 cells per axis. Arms (registered):
Z-head, RZ-head, RZ-avg (target = (head + estimate) / 2). Scored on the 180-shot VALID sample against the entry.

    .venv/bin/python paper/strat/fusion/axis_shift.py            ->  runs/strat/fusion/parts_AX-<arm>.pkl, axis_shift.json
"""
from __future__ import annotations

import json
import pickle

import numpy as np
from scipy.ndimage import map_coordinates

import fx  # noqa: I001  (CPU only)
import strategies as ST

SIGMA, N_IT, CAP_CELLS = 0.15, 4, 2.0
ARMS = {"Z-head": ("Z", "head"), "RZ-head": ("RZ", "head"), "RZ-avg": ("RZ", "avg")}


def grid():
    m = np.load(fx.STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    return np.asarray(m["grid_R"], float), np.asarray(m["grid_Z"], float), m["mask_coarse"].astype(bool)


def make_shifter(R, Z, mask, sign):
    from o_point import find_o_point
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    IZ, IR = np.meshgrid(np.arange(len(Z)), np.arange(len(R)), indexing="ij")
    RR, ZZ = R[IR], Z[IZ]

    def est(psi):
        Ra, Za, *_ = find_o_point(sign * psi, R, Z, mask if mask.any() else None)
        return np.array([Ra, Za])

    def warp(psi, x0, delta):
        g = np.exp(-((RR - x0[0]) ** 2 + (ZZ - x0[1]) ** 2) / (2 * SIGMA ** 2))
        return map_coordinates(psi, [IZ - delta[1] * g / dZ, IR - delta[0] * g / dR], order=3, mode="nearest")

    def shift(psi, head, axes, mode):
        x0 = est(psi); tgt = head if mode == "head" else 0.5 * (head + x0)
        on = np.array([1.0 if "R" in axes else 0.0, 1.0 if "Z" in axes else 0.0])
        cap = CAP_CELLS * np.array([dR, dZ]); delta = np.zeros(2); out = psi
        for _ in range(N_IT):
            delta = np.clip(delta + on * (tgt - est(out)), -cap, cap)
            out = warp(psi, x0, delta)
        return out, delta
    return shift


def main() -> None:
    import sys
    sota, d = fx.data()
    sys.path.insert(0, str(fx.STARTER / "fusion_scoring"))
    from common import AXIS_SIGN
    shots, grp = ST.sample()
    parts_e, nm = ST.load_entry()
    f, comps, mean, H = ST.entry_preds(d, sota)
    iq, ib, ir_, iz_ = (H.index(h) for h in ("efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis"))
    R, Z, mask = grid()
    shifter = make_shifter(R, Z, mask, AXIS_SIGN["DIII-D"])
    rep = {}
    for arm, (axes, mode) in ARMS.items():
        pf = fx.OUT / f"parts_AX-{arm}.pkl"; deltas = []
        if pf.exists():
            parts = pickle.load(open(pf, "rb"))
        else:
            def pred(s):
                cs, ss = f(s); sc = ss.mean(0)
                psi = (cs.mean(0) @ comps + mean).reshape(-1, 65, 65)
                out = np.empty_like(psi)
                for k in range(len(psi)):
                    out[k], dl = shifter(psi[k], np.array([sc[k, ir_], sc[k, iz_]]), axes, mode); deltas.append(dl)
                return out.reshape(len(psi), -1), sc[:, iq], sc[:, ib]
            parts = fx.frame_parts(d, shots, pred, np.asarray(nm["c_mean"]))
            pickle.dump(parts, open(pf, "wb"))
            dl = np.abs(np.array(deltas))
            print(f"[s37n] {arm}: |delta| median R {np.median(dl[:, 0]):.4f} Z {np.median(dl[:, 1]):.4f} m; "
                  f"at cap R {np.mean(dl[:, 0] >= 2 * (R[1] - R[0]) - 1e-9):.4f} Z {np.mean(dl[:, 1] >= 2 * (Z[1] - Z[0]) - 1e-9):.4f}", flush=True)
        r = ST.compare(parts_e, parts, shots, grp, nm, d); rep[arm] = r; o = r["overall"]
        print(f"[s37n] {arm}: S {o['S_ref']:.5f} -> {o['S_new']:.5f} delta {o['delta']:+.5f} [{o['lo']:+.5f}, {o['hi']:+.5f}] "
              f"terms {({k: round(v, 5) for k, v in o['d_terms'].items()})} cons {({k: round(v, 4) for k, v in o['d_cons_each'].items()})}", flush=True)
    (fx.OUT / "axis_shift.json").write_text(json.dumps(rep, indent=1, default=float))
    best = max(rep, key=lambda a: rep[a]["overall"]["S_new"])
    print(f"S37N_VALID best={best} delta={rep[best]['overall']['delta']:+.5f} lo={rep[best]['overall']['lo']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
