"""§37 fusion, strategy R-Ip: one-variable retrain of an entry-member config with the re-aligned plasma-current input.

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_fusion 16 .venv/bin/python \
        paper/strat/fusion/retrain_ip.py --seed 1      ->  runs/strat/fusion/t5-ew2-ipfix_seed1.{pt,json}

Config = trunk1_t5-ew2 (width 512, depth 4, k 64, 21 coil/Ip signals + t, psi loss x3 on shot edges), identical except
that the `mag_plasma_current` column is replaced by runs/strat/fusion/ip_fixed.npy[:, 0] (ip_feature.py). TRAIN fits;
no VALID score is computed here (score_val=0); scoring happens on the registered stratified sample (strategies.py),
paired against the existing trunk1_t5-ew2_seed<k>.pt of the same seed. TEST is never loaded.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
from gpucap import cap_from_env  # noqa: E402

cap_from_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sota  # noqa: E402

OUT = ROOT / "runs" / "strat" / "fusion"


class DataIp(sota.Data):
    """sota.Data with the Ip input column replaced by the re-aligned Ip (TRAIN/VALID rows only)."""

    def __init__(self):
        super().__init__()
        self.ip = np.load(OUT / "ip_fixed.npy", mmap_mode="r")
        self.ipcol = self.fn.index("mag_plasma_current")

    def inputs(self, rows, fm, edge=False):
        x = np.array(np.asarray(self.X[rows])[:, fm])
        assert (self.split[rows] != 2).all(), "TEST rows are never loaded"
        pos = int(np.nonzero(fm)[0].tolist().index(self.ipcol))
        x[:, pos] = np.asarray(self.ip[rows, 0])
        return np.concatenate([x, self.edge_feats()[rows]], 1) if edge else x


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--base", default="trunk1_t5-ew2_seed1.pt")
    a = ap.parse_args()
    ck = torch.load(ROOT / "runs/nas/sophelio/trunks" / a.base, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]); cfg["seed"] = a.seed
    data = DataIp()
    tag = f"t5-ew2-ipfix_seed{a.seed}"
    r = sota.train_eval(cfg, data, score_val=0, save=OUT / f"{tag}.pt")
    r["tag"] = tag; r["base"] = a.base
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=1, default=float))
    print(f"[retrain-ip] {tag}: VALID quick {r['valid_quick']} ({r['train_seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
