"""Fusion §41q final build F1: retrain the 7 entry-ensemble members (ens_select_mlp.json; they carry 50% of q95 / betaN in the
HJl9 recipe) on all 7,041 labelled shots with their stored configs (all_data=True). The coupled_c13-mlp member is the hybrid
classical + 30-atom register model (quantum "srr"), whose stage-1 init trunk1_t5 is retrained first. Same inputs as the
originals (cached block). Writes runs/nas/sophelio/trunks_all/<member>.pt; gpu_build reads them with QW_ENTRY_DIR=trunks_all.

    python paper/strat/fusion/entry_alldata.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper" / "sophelio")); sys.path.insert(0, str(ROOT / "paper"))
import sota  # noqa: E402

SRC = ROOT / "runs" / "nas" / "sophelio" / "trunks"; DST = SRC.parent / "trunks_all"


def train(name: str, data, init: str | None = None, clip: float = 0.0) -> None:
    out = DST / name
    if out.exists():
        print(f"[entry-all] {name} exists", flush=True); return
    cf = os.environ.get("QW_ENTRY_CFGS", "")                           # reproduction: exported configs (repro/fusion/entry_configs.json)
    c = dict(json.loads(pathlib.Path(cf).read_text())[name]) if cf else dict(torch.load(SRC / name, map_location="cpu", weights_only=False)["cfg"])
    c["all_data"] = True
    if clip > 0:                                                       # §41q: refit of members that diverged without clipping
        c["clip"] = clip
    if c.get("init"):
        c["init"] = str(DST / pathlib.Path(c["init"]).name)          # the all-data stage-1 trunk
        assert pathlib.Path(c["init"]).exists(), c["init"]
    res = sota.train_eval(c, data, score_val=0, save=out)
    print(f"[entry-all] {name}: quick (in-sample) {res['valid_quick']}", flush=True)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True); sota.DEV = torch.device("cuda")
    members = json.loads((ROOT / "runs" / "sophelio_sota" / "ens_select_mlp.json").read_text())["members"]
    data = sota.Data()
    redo = [m for m in os.environ.get("QW_ENTRY_REDO", "").split(",") if m]           # e.g. trunk1_t19-ew2_seed1.pt (clip 1.0)
    for m in redo:
        (DST / m).unlink(missing_ok=True); train(m, data, clip=1.0)
    train("trunk1_t5.pt", data)                                        # stage 1 of the register member
    for m in members:
        train(m, data)
    print("ENTRY_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
