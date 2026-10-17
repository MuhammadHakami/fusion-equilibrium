"""§37 fusion, strategy R-Ip (CPU, matched short budget): the t5-ew2 entry-member config trained twice with identical
seed and a reduced budget, once with the shipped Ip column (arm `ship`) and once with the re-aligned Ip (arm `fix`), and once with the re-aligned Ip plus four Ip-history
inputs in true milliseconds (arm `fixhist`: Ip(t-200), Ip(t-50), Ip(t+20), dIp/dt; tests temporal context).
The GPU queue was saturated by the other §37 rows, so this matched pair runs on CPU (6 threads, < 6 GB).

    .venv/bin/python paper/strat/fusion/retrain_cpu.py --arm fix --epochs 6 --seed 1  ->  runs/strat/fusion/cpu_<arm>_e6_s1.pt

TRAIN fits only; score_val = 0 (the stratified-sample scoring is strategies.py). TEST rows are never loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "sophelio")); sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("STRAT_THREADS", "6")))
import sota  # noqa: E402

sota.DEV = torch.device("cpu")
OUT = ROOT / "runs" / "strat" / "fusion"


class DataCols(sota.Data):
    """Reads only the masked columns in row chunks (keeps RSS low); optionally swaps in the re-aligned Ip."""

    def __init__(self, fix: bool, hist: bool = False):
        super().__init__()
        self.fix, self.hist = fix, hist
        self.ip = np.load(OUT / "ip_fixed.npy", mmap_mode="r")
        self.ipcol = self.fn.index("mag_plasma_current")

    def inputs(self, rows, fm, edge=False):
        assert (self.split[rows] != 2).all(), "TEST rows are never loaded"
        cols = np.nonzero(fm)[0]
        x = np.concatenate([np.asarray(self.X[rows[i:i + 200000]])[:, cols] for i in range(0, len(rows), 200000)])
        if self.fix:
            x[:, cols.tolist().index(self.ipcol)] = np.asarray(self.ip[rows, 0])
        if self.hist:                          # + Ip(t-200 ms), Ip(t-50 ms), Ip(t+20 ms), dIp/dt (true ms)
            x = np.concatenate([x, np.asarray(self.ip[rows, 1:5])], 1)
        return np.concatenate([x, self.edge_feats()[rows]], 1) if edge else x


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["ship", "fix", "fixhist"], required=True)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    ck = torch.load(ROOT / "runs/nas/sophelio/trunks/trunk1_t5-ew2_seed1.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]) | {"seed": a.seed, "epochs": a.epochs}
    tag = f"cpu_{a.arm}_e{a.epochs}_s{a.seed}"
    r = sota.train_eval(cfg, DataCols(a.arm in ("fix", "fixhist"), a.arm == "fixhist"), score_val=0, save=OUT / f"{tag}.pt")
    r["tag"] = tag
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=1, default=float))
    print(f"[cpu-retrain] {tag}: VALID quick {r['valid_quick']} ({r['train_seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
