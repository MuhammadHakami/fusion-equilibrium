#!/bin/bash
# Fusion reproduction: train every model component from scratch (after the data step in README.md). Exact recipes of the Final candidates
# (seeds as shipped):  CAND=F15K = F14K with Huber-loss (delta 0.3) shape heads J1RLHA and q/b heads QHAR2Hu
# (doc 35 §45l);  CAND=F16K = F15K + Ip-sign canonical twins J1RLHACS s1-4 / QHAR2HuCS s1-2 that replace the heads on reversed-Ip shots
# (doc 35 §45o);  CAND=F17K (default; the adopted Final configuration, 2026-10-17) = F16K with the Ip-sign canonical register heads J1QHCS (replacing J1QH) and X-point heads XPACS (replacing XPA)
# on every shot (doc 35 §45s / §45w; same compute as F16K);  CAND=F18K (default; the Final configuration, 2026-10-18) = F17K + the VisionHOPE +
# input-dependent Rydberg-register shape heads V60 s1-2 (step 9; doc 35 §47c / §47f; +3.1 GB10-hours). Plus the fast-path register heads J1QH (§43l), the gradient-boosted li tree
# (§45a), kw2 at inference.
# Reference machine: one NVIDIA GB10 (CUDA). GPU time alone per component: psi trunks ~1.5 h, MLP heads ~32 min each, PI-KAN register heads
# ~4.2 h each, the three J1QH heads 2.8 h together, li tree ~3 min, NAS entry ensemble ~2 h -> F15K ~22.2 GB10-hours, F16K ~25.4 (6 more
# MLP heads). See README.md for the A100 conversion.
set -e; cd "$(dirname "$0")/../.."; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1; P=.venv/bin/python; F=paper/strat/fusion
CAND=${CAND:-F18K}; case "$CAND" in F15K|F16K|F17K|F18K) ;; *) echo "CAND must be F15K, F16K, F17K or F18K"; exit 2;; esac
TW=0; [ "$CAND" = F15K ] || TW=1                                           # the Ip-sign canonical twins (F16K, F17K)
CN=""; QT=J1QH; XT=_XPA; { [ "$CAND" = F17K ] || [ "$CAND" = F18K ]; } && { CN="STRAT_CANON=2"; QT=J1QHCS; XT=_XPACS; }   # F17K / F18K: canonical register / X-point heads
# 1 NAS entry ensemble (7 members incl. the srr register model coupled_c13-mlp, + its stage-1 init trunk1_t5) on all labelled shots
QW_ENTRY_CFGS=repro/fusion/entry_configs.json $P $F/entry_alldata.py      # configs as shipped (t19-ew2 members: clip 1.0)
# 2 psi trunks (PSI-F config + Ip-sign canonical inputs + all data), seeds 2, 1, 3
CFG='"k": 64, "width": 2048, "depth": 2, "drop": 0.1, "lr": 0.0025322139193052723, "batch": 4096, "epochs": 90, "w_scal": 0.0, "gamma": 0.3152355297039971, "lags": "sym", "thomson": true, "quantum": "none", "k_lo": 32, "q_layers": 4, "edge_w": 2.0, "clip": 1.0, "fast": true, "inputs": "magfix", "canon": true, "all_data": true'
for s in 2 1 3; do $P paper/sophelio/sota_train.py --tag PSIFA_s$s --score-val 0 --cfg "{$CFG, \"seed\": $s}"; done
# 2b psi trunks PQ2A (as 2 + inputs-only physics scalings and coil / |Ip| ratios, qphys_inputs.py), seeds 2, 1, 3; psiMixA averages PQ2A + PSIFA
for s in 2 1 3; do $P paper/sophelio/sota_train.py --tag PQ2A_s$s --score-val 0 --cfg "{$CFG, \"qphys\": 2, \"seed\": $s}"; done
E="STRAT_LABEL_SUFFIX=_v8 STRAT_CLIP=1.0 STRAT_FAST=1 STRAT_EPOCHS=60 STRAT_WIDTH=4096 STRAT_DEPTH=4 STRAT_INPUTS=magfix QW_MAGFIX=X_magfix"
J="STRAT_ALL=1 STRAT_SCAL_W=9.142857 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 STRAT_HUBER=0.3"
# 4 shape heads J1RLHA (J1AH inputs + physics scalings / coil-Ip ratios + Te^1.5 current-profile proxies, all data, Huber delta 0.3), seeds 1-8
for s in 1 2 3 4 5 6 7 8; do env $E $J STRAT_OUT_TAG=_J1RLHA $P $F/head_boundary.py $s; done
# 4c (F16K, F17K) Ip-sign canonical twins J1RLHACS (= J1RLHA + STRAT_CANON=2: currents x sign(Ip) of the shot + the sign as an input), seeds 1-4
[ "$TW" = 1 ] && for s in 1 2 3 4; do env $E $J STRAT_CANON=2 STRAT_OUT_TAG=_J1RLHACS $P $F/head_boundary.py $s; done
# 5 register heads (PI-KAN + simulated 30-atom Rydberg layer, volume physics term), seeds 1-2
PK='{"block": "mlp", "width": 4096, "depth": 4, "drop": 0.1, "lr": 0.0025322139193052723, "w_pi": 0.1, "q_layers": 4, "q_in": 128, "link": "cascade", "alpha_mult": 1.0, "alpha_init": 0.1}'
for s in 1 2; do $P $F/hpo_qpikan.py --fixed "$PK" --tag J1QPIpi --epochs 60 --seed $s; done
# 5b register heads J1QH (§43j trial 10: MLP 2048 x 4 + 30-atom Rydberg layer, independent link, no physics loss; fast layered register
#    path QW_QFAST=1; J1RL inputs; TRAIN split; F17K: + STRAT_CANON=2 -> J1QHCS), seeds 1-3 - run concurrently on one GPU (2.8 h together on the GB10)
QH='{"block": "mlp", "width": 2048, "depth": 4, "drop": 0.1, "q_layers": 4, "q_in": 64, "link": "independent", "lr": 0.0012505271791067153, "alpha_mult": 1.0, "alpha_init": 0.1, "w_pi": 0.0}'
pids=(); for s in 1 2 3; do env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 $CN $P $F/hpo_qpikan.py --fixed "$QH" --tag $QT --epochs 60 --seed $s & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done        # per-PID wait: a failed seed stops the rebuild under set -e (a bare `wait` returns 0)
# 6 q95 / betaN heads QHAR2Hu (physics scalings + coil / |Ip| ratios, all data, Huber delta 0.3), seeds 1-3
for s in 1 2 3; do env $E STRAT_ALL=1 STRAT_QPHYS=2 STRAT_HUBER=0.3 STRAT_OUT_TAG=_QHAR2Hu $P $F/head_boundary.py $s qb; done
# 6c (F16K, F17K) canonical q/b twins QHAR2HuCS (= QHAR2Hu + STRAT_CANON=2), seeds 1-2
[ "$TW" = 1 ] && for s in 1 2; do env $E STRAT_ALL=1 STRAT_QPHYS=2 STRAT_HUBER=0.3 STRAT_CANON=2 STRAT_OUT_TAG=_QHAR2HuCS $P $F/head_boundary.py $s qb; done
# 7 X-point heads XPA (J1RLA inputs; targets = continuous saddle of the true psi, xp_labels.npz from the data step; F17K: + STRAT_CANON=2 -> XPACS), seeds 1-3
for s in 1 2 3; do env $E STRAT_ALL=1 STRAT_THZ=1 STRAT_HFULL=1 STRAT_QPHYS=2 STRAT_LIPHYS=1 $CN STRAT_OUT_TAG=$XT $P $F/head_boundary.py $s xp; done
# 8 li tree (CatBoost 1.2.10 GPU, depth 8, 3000 it, lr 0.08; J1RL inputs; every 3rd TRAIN + every 3rd VALID/TEST frame) -> gbdt_shape_li_all.cbm
#    (PT = a Python with catboost==1.2.10; ours is .venv-tdc). Test-fold predictions come from runs/final_fold.sh (dumppub / litreepred).
PT=${PT:-.venv-tdc/bin/python}; export QW_TREE_STEM=head_boundary_v8_J1RLHA_s1     # tree inputs = the J1RLHA input pipeline (identical to J1RL's)
$P $F/gbdt_shape.py dump && $PT $F/gbdt_shape.py litree
# 9 (F18K) VisionHOPE + register shape heads V60 (quota_fusion.py: 2 VisionHOPE-T blocks, width 128, 8 tokens per frame, on 32-frame crops of
#   128 shots per step; zero-init gate into the J1QHCS-recipe MLP trunk 2048 x 4; head = linear + skip + alpha x 3 x 10-atom register;
#   TRAIN split, 60 epochs = 20.4 k steps, ~1.5 h per seed alone on the GB10; VisionHOPE CUDA kernels are JIT-built with ninja), seeds 1-2
V='{"backbone": "vhope", "seq": true, "crop": 32, "shots": 128, "bb_width": 128, "bb_depth": 2, "tokens": 8}'
[ "$CAND" = F18K ] && for s in 1 2; do env QW_QFAST=1 QW_MAGFIX=X_magfix STRAT_THZ=1 STRAT_HFULL=1 STRAT_LIPHYS=1 STRAT_QPHYS=2 STRAT_CANON=2 \
  $P $F/quota_fusion.py final V60 "$V" $s 60; done
echo "TRAIN_ALL_DONE $CAND $(date -Is)"
