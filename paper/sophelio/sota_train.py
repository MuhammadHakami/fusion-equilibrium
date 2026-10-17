"""Single-run CLI over paper/sophelio/sota.py (cached full DIII-D data, hybrid model, official scorer).

    .venv/bin/python paper/sophelio/sota_train.py --tag v1 [--cfg '{"quantum": "none", "epochs": 30}'] [--score-test 0]

VALID drives every choice; --score-test > 0 scores TEST once and is reserved for final, pre-selected runs.
"""
from __future__ import annotations

import argparse
import os
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
import sota  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--cfg", default="{}")
    ap.add_argument("--score-val", type=int, default=40); ap.add_argument("--score-test", type=int, default=0)
    a = ap.parse_args()
    sota.build_cache()
    data = sota.Data()
    cfg = json.loads(a.cfg)
    if cfg.get("inputs") == "magfix":                           # doc 35 §38g: corrected magnetics block (Ip + true-ms lags)
        sys.path.insert(0, str(ROOT / "paper" / "strat" / "fusion"))
        import inputs_fix
        inputs_fix.use(data)
    if cfg.get("canon"):                                        # §41q: Ip-sign canonical current inputs (targets invariant)
        sys.path.insert(0, str(ROOT / "paper" / "strat" / "fusion"))
        import canon
        canon.use(data)
    if cfg.get("qphys"):                                        # §42h: physics scalings (+ coil / Ip ratios if 2), after inputs_fix / canon
        sys.path.insert(0, str(ROOT / "paper" / "strat" / "fusion"))
        import qphys_inputs
        qphys_inputs.use(data, ratios=int(cfg["qphys"]) == 2)
    if cfg.get("hfull"):                                        # §42o: + full-rate current history (24 columns, appended last)
        sys.path.insert(0, str(ROOT / "paper" / "strat" / "fusion"))
        import hfull_inputs
        hfull_inputs.use(data)
    out = ROOT / "runs" / "sophelio_sota"; out.mkdir(parents=True, exist_ok=True)
    res = sota.train_eval(cfg, data, score_val=a.score_val, score_test=a.score_test, save=out / f"{a.tag}.pt")
    res["tag"] = a.tag
    (out / f"{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[sota] {a.tag}: VALID quick {res['valid_quick']}", flush=True)
    for k in ("valid_official", "test_official"):
        if k in res:
            print(f"[sota] {a.tag}: {k} {res[k]}", flush=True)


if __name__ == "__main__":
    main()
