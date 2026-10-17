"""Fusion Equilibrium Challenge (Codabench 17739, Development phase): predictions of our registered entry on the public
DIII-D test fold. The entry is the ensemble chosen by the committed TEST rule (score_test.py: best full-VALID S), i.e.
runs/sophelio_sota/ens_select_mlp.json (7 members, sigma from full VALID), predicted with the same member loader
(ens_select.member) that produced our held-out TEST S 0.9910; only the frames come from the public test shots.

Shot order = the Hugging Face config's sorted file order (data/diii_d_public_test/*.parquet), checked against the local
copy; T per shot = len(efit_times) (checked); psirz float16 (never rounded), q95 / betaN float32.

    .venv/bin/python paper/sophelio/submit_public.py [--entry ens_select_reg.json --out submission_reg]
        ->  runs/sophelio_sota/<out>/{diii_d_public_test.npz, submission.zip}
--entry picks another VALID-selected ensemble (e.g. the register-coupled selection, ens_select_reg.json).
"""
from __future__ import annotations

import os

import argparse
import json
import pathlib
import sys
import zipfile

import numpy as np
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ens_select as ES  # noqa: E402
import postproc as PP  # noqa: E402
import sota  # noqa: E402

RAW = sota.ROOT / "data/sophelio/fusion-equilibrium-challenge/data" / os.environ.get("QW_FOLD", "diii_d_public_test")
PROC = sota.ROOT / "data/sophelio_proc" / os.environ.get("QW_FOLD", "diii_d_public_test")
OUT = sota.ROOT / "runs/sophelio_sota/submission"
ENTRY = sota.ROOT / "runs/sophelio_sota/ens_select_mlp.json"


class PublicData:
    """The slice of sota.Data that ens_select.member needs, over the public test frames (inputs only)."""

    def __init__(self, train_meta: dict):
        self.files = sorted(PROC.glob("*.npz"))
        raw = sorted(RAW.glob("*.parquet"))
        if [f.stem for f in self.files] != [r.stem for r in raw]:
            raise RuntimeError("public test features do not match the raw file order")
        xs, shot = [], []
        for i, (f, r) in enumerate(zip(self.files, raw)):
            x = np.load(f)["X"]
            T = len(pq.read_table(r, columns=["efit_times"]).column(0)[0].as_py())
            if len(x) != T or not np.isfinite(x).all():
                raise RuntimeError(f"{f.stem}: {len(x)} frames vs efit_times {T} or non-finite inputs")
            xs.append(x); shot.append(np.full(T, i, np.int32))
        self.X, self.shot = np.concatenate(xs), np.concatenate(shot)
        self.meta, self.fn, self._edge = train_meta, train_meta["feature_names"], None

    edge_feats = sota.Data.edge_feats
    inputs = sota.Data.inputs

    def shots(self) -> list[int]:
        return list(range(len(self.files)))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--entry", default=ENTRY.name); ap.add_argument("--out", default=OUT.name)
    a = ap.parse_args()
    entry, out = ENTRY.parent / a.entry, OUT.parent / a.out
    out.mkdir(parents=True, exist_ok=True)
    train = sota.Data()
    pub = PublicData(train.meta)
    sel = json.loads(entry.read_text()); vf = sel["valid_full"]; sigma = float(max(vf, key=lambda s: vf[s]["S"]))
    shots = pub.shots()
    members = [ES.member(str(sota.ROOT / "runs/nas/sophelio/trunks" / m), pub, shots) for m in sel["members"]]
    comps, mean = train.meta["comps"][:ES.KMAX].numpy(), train.meta["mean"].numpy()
    H = sota.HEADS
    preds = {}
    for i in shots:
        pc = np.mean([mb[i][0] for mb in members], axis=0); sc = np.mean([mb[i][1] for mb in members], axis=0)
        psi, sc = PP.smooth(pc @ comps + mean, sigma), PP.smooth(sc, sigma)
        T = int((pub.shot == i).sum())
        preds[f"shot_{i:04d}_psirz"] = np.asarray(psi, np.float32).reshape(T, 65, 65).astype(np.float16)
        preds[f"shot_{i:04d}_q95"] = np.asarray(sc[:, H.index("efit_q95")], np.float32)
        preds[f"shot_{i:04d}_betaN"] = np.asarray(sc[:, H.index("efit_beta_n")], np.float32)
        if not all(np.isfinite(preds[f"shot_{i:04d}_{k}"]).all() for k in ("psirz", "q95", "betaN")):
            raise RuntimeError(f"shot {i}: non-finite prediction")
    npz = out / "diii_d_public_test.npz"
    np.savez_compressed(npz, **preds)
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(npz, arcname=npz.name)                                          # the .npz at the archive root
    info = {"entry": entry.name, "members": sel["members"], "sigma": sigma, "shots": len(shots), "frames": int(len(pub.X)),
            "npz_bytes": npz.stat().st_size}
    (out / "build.json").write_text(json.dumps(info, indent=1)); print("[submit] built", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
