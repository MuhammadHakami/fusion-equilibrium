# Fusion Equilibrium Challenge — reproduction package (DIII-D, Challenge 1)

MIT-licensed reproduction package (exact runtime inputs with sha256: MANIFEST_runtime.txt; weights: WEIGHTS.sha256 + fetch_weights.py). Three candidates, selected with `CAND`; **F17K is the adopted Final configuration** (doc 35, 2026-10-17):
- **F15K** (Development public S 0.995949, 2026-10-17): F14K with Huber-loss (delta 0.3) shape heads J1RLHA and q/b heads
  QHAR2Hu (doc 35 §45l). F14K = F8 (QHAR2 q/b heads) + three fast-path register shape heads J1QH (§43l) + a gradient-boosted li tree
  blended into the heads' li at w 0.2 (§45a) + a second kappa / li pass (kw2).
- **F16K** (Development public S 0.996167, 2026-10-17): F15K + Ip-sign routing - on reversed-Ip shots the MLP shape heads are the canonical
  twins J1RLHACS s1-4 and the q/b heads QHAR2HuCS s1-2 (§45o); normal-Ip predictions are identical to F15K.
- **F17K** (default `CAND`; adopted for the Final; decided on VALID+TEST and label-free evidence, doc 35 §45s / §45w; not uploaded in Development): F16K with the register heads J1QH and the X-point
  heads XPA replaced on every shot by their Ip-sign canonical twins J1QHCS s1-3 / XPACS s1-3 (same recipes + canonical current inputs;
  same training compute as F16K).
It will be frozen at the Final-phase submission. Every number below is reproducible from the scripts named here.

## What the model is

A hybrid classical + quantum-register pipeline predicting, per EFIT frame, the 65x65 poloidal flux map psi, q95 and betaN
from the challenge inputs only (coil currents, Ip, Thomson scattering):

| Stage | Component | Algorithm | Trained on |
|---|---|---|---|
| psi | `PSIFA_s{2,1,3}` + `PQ2A_s{2,1,3}` (mean of 6) | MLP trunks (2048 x 2) on 64 PCA coefficients of psi, Ip-sign-canonical inputs; PQ2A add the `qphys_inputs.py` scalings and coil / \|Ip\| ratios; axis guard: frames whose O-point lies > 0.25 m from the head axis use the PSIFA mean | all 7,041 labelled shots |
| shape heads | `J1RLHA_s1..s8` (F16K: + `J1RLHACS_s1..s4` on reversed-Ip shots) | MLP (4096 x 4), Huber loss delta 0.3: 7 consistency scalars + 64 LCFS rays; corrected magnetics + fixed-coordinate Thomson + full-rate current history + physics scalings / coil-Ip ratios + Te^1.5 current-profile proxies (`liphys_feats.py`) | all shots |
| register heads | `J1QPIpi_s1, s2` | PI-KAN head with a 30-atom Rydberg register layer (simulated; cascade link) and a volume physics term | TRAIN split |
| register heads | `J1QH_s1..s3` | MLP (2048 x 4) with a 30-atom Rydberg register layer (simulated, layered fast path; independent link; no physics loss), J1RLA inputs | TRAIN split |
| li tree | `gbdt_shape_li_all.cbm` | CatBoost 1.2.10 (GPU; depth 8, 3000 iterations, lr 0.08) on the J1RLA input set; blended into the shape heads' li at w 0.2 | every 3rd frame of all shots |
| X-point heads | `XPA_s1..s3` | MLP (4096 x 4, J1RLA inputs) predicting the lower / upper X-point (continuous Newton saddle of the true psi, `xp_labels.py`) | all shots |
| q95 / betaN | `QHAR2Hu_s1..s3` (80%; F16K: `QHAR2HuCS_s1..s2` on reversed-Ip shots) + NAS entry ensemble (20%) | MLP heads (Huber loss delta 0.3) on corrected magnetics + inputs-only physics scalings (log Bt/Ip, pressure moments, betaN / beta_p / Greenwald-like ratios) and the 19 coil currents / \|Ip\| (`qphys_inputs.py`); 7 NAS trunks (`ens_select_mlp.json`) | all shots |
| post-processing | `gpu_build.py` | psi edits toward the heads: axis shift, boundary warp, continuous X-point placement (xhe), kappa warp, X-point warp, joint Gauss-Newton shape fit, li warp, second li pass; then `post_kw2.py`: second kappa pass + li x2; q95 / betaN clamped to the labelled range (`qb_clamp.json`: q95 = all-shot label min / max, betaN = label quantiles 1e-4 / 1 - 1e-4) | (no training) |

Head ensemble weights: 24 J1RLHA : 4 J1QPIpi : 12 J1QH (each J1RLHA seed 3; F16K reversed-Ip shots: each J1RLHACS seed 6, each J1QPIpi 2, each J1QH 4; the VALID+TEST-validated
proportions 6 : 1 : 3 of doc 35 §43l, scaled to the seed counts); the li output is then blended with the li tree, 0.8 heads + 0.2 tree
(5-fold shot cross-fitted weight on VALID+TEST, §45a). q/b blend weight 0.8 on the heads (VALID+TEST study, doc 35 §42a).

## Hardware and compute (declared)

- Reference machine: NVIDIA DGX Spark (GB10, 121 GB unified memory, aarch64), CUDA 13.0, PyTorch 2.13.0, Python 3.13.12.
- Wall-clock per component on the GB10 (training, one job alone): psi trunk ~10-20 min (x6); MLP head ~32 min (J1RLHA x8, QHAR2Hu x3,
  XPA x3 = 14 heads; F16K + J1RLHACS x4 + QHAR2HuCS x2 = 20 heads); PI-KAN register head ~4 h 10 min (quantum-layer simulation, x2); fast-path register heads J1QH x3 2 h 50 min
  together (concurrent on one GPU); li tree ~3 min (CatBoost GPU); NAS entry members
  (all 7 + stage-1) ~2 h; feature extraction / labels / X-point labels (mostly CPU) ~2 h; inference + post-processing on a test
  fold ~1 h.
- GPU total from scratch, sequential on one GB10: model training F15K ~ 1.5 + 14 x 0.53 + 2 x 4.2 + 2.8 + 0.05 + 2 h ~ 22.2 GB10-hours;
  F16K ~ 25.4 (6 more MLP heads). GPU data generation (the rest of the data step is CPU-only): X-point labels (xp_labels.py) measured
  0.09 h for all 1.56 M frames (+5% check) and the psi PCA in build_cache (SVD of 90k x 4225 in 25 s; per-file projections are small
  matmuls inside a CPU-bound load loop), counted conservatively as <= 0.5 h -> **F15K <= 22.8 GB10-hours** (+ ~1 h test-fold inference). Bound for the 24 GPU-hour ceiling (an A100-class GPU; NOT timed - no A100 available): the components are
  dense bf16 MLPs and a simulated register layer, limited by matmul throughput and memory bandwidth, where an A100-80GB has ~2.5x (bf16
  dense) and ~7x (2.0 TB/s vs 273 GB/s) the GB10's peak. Measured on the GB10 (doc 35 §45r): an MLP head epoch (1.556 M samples x ~822 MFLOP)
  runs at ~41 TFLOPS achieved; the MLP-type parts (~14.1 GB10-h) fit if an A100 sustains >= ~44 TFLOPS (14% of peak) on 4096-wide GEMMs.
  Concurrency does NOT help on the GB10 (4 heads at once: 14% slower per head than alone). The register heads (11.2 GB10-h) are a simulated
  30-atom layer; profiled (doc 35 §45r): ~44% of its GPU time is complex FP32 GEMMs on SIMT cores (GB10 ~31 vs A100 19.5 TFLOPS), ~50% small
  kernels - so ~1.2x SLOWER on an A100. Component estimate: **F16K / F17K ~20, F15K ~19 A100-hours** (~4 h under 24). Not timed on an A100.

## Data provenance

- Challenge data v1.1.0 (`diii_d_train`, the test fold) and the organizers' `data_fixes.fix_d3d_ip_times` (final data per the
  2026-10-11 starter README). No external datasets, no pre-trained models, no MAST data of any kind.
- Our own internal split of `diii_d_train` (TRAIN / VALID / TEST by shot) is used for model selection; the final components
  are refit on all 7,041 shots.

## Rebuild from scratch

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r repro/fusion/requirements.lock
uv venv -p 3.12 .venv-tdc && uv pip install --python .venv-tdc/bin/python catboost==1.2.10 numpy==2.5.2     # the li tree (train_all.sh step 8, final_fold.sh)
# 1 data: per-shot features, cache + PCA, corrected inputs, feature blocks, scorer-derived shape labels
.venv/bin/python paper/sophelio/extract.py --split diii_d_train --workers 8
.venv/bin/python -c "import sys; sys.path.insert(0, 'paper/sophelio'); import sota; sota.build_cache()"
.venv/bin/python paper/strat/fusion/mag_fix.py --target trainvalid && .venv/bin/python paper/strat/fusion/mag_fix.py --target test
.venv/bin/python paper/strat/fusion/ip_fix.py
.venv/bin/python paper/strat/fusion/thz_feats.py --all
.venv/bin/python paper/strat/fusion/hist_full.py
STRAT_NANG=64 STRAT_LABEL_SUFFIX=_v8 .venv/bin/python paper/strat/fusion/boundary_labels.py 1
.venv/bin/python paper/strat/fusion/boundary_labels_vt.py
.venv/bin/python paper/strat/fusion/liphys_feats.py
.venv/bin/python paper/strat/fusion/xp_labels.py                         # X-point labels (continuous saddle of the true psi)
# 2 models: repro/fusion/train_all.sh (entry ensemble, psi trunks, shape / register / q-b heads; exact env per component)
CAND=F17K repro/fusion/train_all.sh            # the adopted Final; or CAND=F16K (J1QH / XPA in place of the twins), or CAND=F15K (no Ip-sign twins)
```

## Regenerate the submitted predictions

```bash
QW_FOLD=diii_d_private_test runs/final_fold.sh            # inputs-only feature blocks for the test fold
QW_FOLD=diii_d_private_test CAND=F17K repro/fusion/infer.sh <tag>   # (or CAND=F16K / F15K) -> runs/sophelio_sota/submission_<tag>K/diii_d_private_test.npz
```

Determinism: q95 / betaN reproduce to ~1e-6; psi to ~1e-3 absolute (GPU reduction order in the batched post-processing),
checked by rebuilding the Development entry on a 5-shot fold. infer.sh (CAND=F15K / F16K / F17K) reproduces runs/final_f15k.sh / final_f16k.sh / TAG=F17 REGALL=1 XPALL=1 final_f16k.sh exactly on the organizers'
smoke fold (diii_d_fold_smoke, 864 frames; agreement psi / q95 / betaN 1.000000, 2026-10-17). The submitted file is submission_<tag>K.
Package assembly: `.venv/bin/python repro/fusion/export.py <dir>` writes the clean package tree (61 Python sources = the import closure of
every entry script, the shell entrypoints, this README, the lockfile, MANIFEST_source.txt with sha256; weights / caches listed in
MANIFEST_runtime.txt are symlinked, or copied with --copy-weights). The exported tree, run in place (runs/final_fold.sh + infer.sh CAND=F17K
on the smoke fold), reproduces the repository's F17K smoke predictions bit for bit (2026-10-17).

## Methods disclosure

AI coding assistants (Claude Code) were used for code, experiments and analysis under the entrant's direction and
responsibility (rules update 2026-09-23, section 4).
