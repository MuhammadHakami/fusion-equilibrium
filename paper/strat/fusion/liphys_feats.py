"""Doc 35 §42p: inputs-only current-profile proxies for li (internal inductance), from the fixed-coordinate Thomson block (thz).

Ohmic steady state: j ~ sigma E with Spitzer sigma ~ Te^1.5, so the current-profile width (which sets li) follows the Te^1.5 profile.
Per frame, on the vertical core chord (R = 1.94 m, Z grid 0.02-0.83 m) inside the plasma top z_e (highest Z with Te > 100 eV):
Te^1.5 and pressure (Te ne) centroid / rms width over z_e, Te / Te^1.5 / ne peaking (lowest covered point vs the chord mean),
log Te_core^1.5; on the tangential midplane chord (R 1.67-2.06 m): Te and Te^1.5 R-centroids and Te peaking. 13 columns, 0 where
undefined. Rows align with the thz block they are computed from (cache / public / private folds alike).
    .venv/bin/python paper/strat/fusion/liphys_feats.py        -> runs/strat/fusion/liphys_{all,public,private}.npy
"""
from __future__ import annotations

import numpy as np

import fx

ZC = np.linspace(0.02, 0.83, 28); RT = np.linspace(1.67, 2.06, 8); NF = 13


def feats(T: np.ndarray) -> np.ndarray:
    T = np.nan_to_num(np.asarray(T, np.float64))
    te, ne = np.clip(T[:, :28], 0, None) / 1e3, np.clip(T[:, 28:56], 0, None)           # keV, 1e19 m^-3
    tt = np.clip(T[:, 84:92], 0, None) / 1e3; ze = T[:, 100]
    ze = np.where(ze > 0, ze, np.where((te > 0).any(1), ZC[np.argmax(np.where(te > 0, ZC, 0), 1)], 0.0))
    ok = (te > 0) & (ZC[None] <= ze[:, None] + 1e-6); s, p = np.where(ok, te ** 1.5, 0), np.where(ok, te * ne, 0)
    zs = np.maximum(ze, 0.05)[:, None]
    def mom(w):
        m0 = w.sum(1) + 1e-12
        return (w * ZC).sum(1) / m0 / zs[:, 0], np.sqrt((w * ZC ** 2).sum(1) / m0) / zs[:, 0]
    cs, ws = mom(s); cp, wp = mom(p)
    first = np.argmax(ok, 1); r = np.arange(len(T)); nok = ok.sum(1) + 1e-12
    pk = lambda v: v[r, first] / (np.where(ok, v, 0).sum(1) / nok + 1e-9)
    okt = tt > 0; st = tt ** 1.5
    cen = lambda w: (w * RT).sum(1) / (w.sum(1) + 1e-12)
    F = np.stack([cs, ws, cp, wp, pk(te), pk(te ** 1.5), pk(ne), np.log(te[r, first] ** 1.5 + 1e-3), ze,
                  cen(np.where(okt, tt, 0)), cen(np.where(okt, st, 0)), tt.max(1) / (np.where(okt, tt, 0).sum(1) / (okt.sum(1) + 1e-12) + 1e-9),
                  np.log(s.sum(1) * (ZC[1] - ZC[0]) + 1e-3)], 1)
    has = ok.sum(1) >= 3
    F[~has, :9] = 0; F[~has, 12] = 0; F[okt.sum(1) < 2, 9:12] = 0
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    src = fx.ROOT / "data" / "sophelio_proc"
    for name, path in (("all", src / "cache_k256" / "thz.npy"), ("public", src / "thz_public.npy"), ("private", src / "thz_private.npy")):
        if not path.exists():
            continue
        T = np.load(path, mmap_mode="r"); out = np.zeros((len(T), NF), np.float32)
        for i in range(0, len(T), 200000):
            out[i:i + 200000] = feats(T[i:i + 200000])
        np.save(fx.OUT / f"liphys_{name}.npy", out)
        nz = (out != 0).mean(0)
        print(f"[liphys] {name}: {out.shape}, nonzero frac {np.round(nz, 2).tolist()}, means {np.round(out.mean(0), 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
