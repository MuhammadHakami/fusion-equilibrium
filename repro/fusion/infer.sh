#!/bin/bash
# Fusion reproduction: predictions for a test fold with the shipped (or rebuilt) weights. Run runs/final_fold.sh first with the
# same QW_FOLD (it builds the fold's thz / hfull / qphys / liphys / qshape blocks and the X_ipfix sign block).
#   QW_FOLD=diii_d_private_test [CAND=F18K|F17K|F16K|F15K] repro/fusion/infer.sh <tag>
set -e; cd "$(dirname "$0")/../.."; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1; : "${QW_FOLD:?set QW_FOLD}"
CAND=${CAND:-F18K}; case "$CAND" in F15K|F16K|F17K|F18K) ;; *) echo "CAND must be F15K, F16K, F17K or F18K"; exit 2;; esac
# Final candidate F15K (default): psi = mean of PQ2A s1-3 + PSIFA s1-3 (psiMixA) with the axis guard (frames whose O-point is > 0.25 m from the
#     head axis take the PSIFA coefficients); shape heads 24 J1RLHA : 4 J1QPIpi : 12 J1QH (each J1RLHA seed weight 3, J1QPIpi 2, J1QH 4; doc 35
#     §43l), li blended with the gradient-boosted li tree at w 0.2 (QW_LITREE, §45a; litree_<tag>.npz from runs/final_fold.sh); q/b = 0.8 x
#     QHAR2Hu s1-3 + 0.2 x entry; post-processing adds the continuous X-point step (xhe, XPA heads, no gate, weight 0.75; §42t step 4) and a
#     second li pass (li2); then post_kw2.py: a second kappa pass + li x2 toward the same heads (§42y); q95 / betaN clamped to the labelled
#     range (qb_clamp.json, §42z). Same chain as runs/final_f15k.sh (rehearsed on the organizers' smoke fold, 2026-10-17).
# CAND=F16K: as F15K, but on reversed-Ip shots (sign of the shot's mean corrected Ip, X_ipfix block) the MLP shape heads are the canonical
#     twins J1RLHACS s1-4 (weight 6 each; the register heads unchanged) and the q/b heads are QHAR2HuCS s1-2 (doc 35 §45o; gpu_build "route:"
#     stems). Same chain as runs/final_f16k.sh.
# CAND=F17K: as F16K, with the Ip-sign canonical register heads J1QHCS s1-3 in place of J1QH and the canonical X-point heads XPACS s1-3
#     in place of XPA, on every shot (doc 35 §45s / §45w). Same chain as TAG=F17 REGALL=1 XPALL=1 runs/final_f16k.sh.
# CAND=F18K (default; the Final configuration, 2026-10-18): F17K + the VisionHOPE + input-dependent Rydberg-register shape heads V60 s1-2
#     (paper/strat/fusion/quota_fusion.py: VisionHOPE-T blocks on the per-shot frames x tokens map -> zero-init gate into the frame's MLP
#     trunk -> linear + skip + alpha x 30-atom register read-out; 60 epochs) at weight 6.75 each on every shot (share 13.5 / 53.5 = 0.25;
#     doc 35 §47c / §47f). Same chain as TAG=F18 REGALL=1 XPALL=1 VH=1 runs/f17.sh (public) / runs/final_f16k.sh (test folds).
REG="head_boundary_v8_J1QPIpi_s1*2+head_boundary_v8_J1QPIpi_s2*2+head_boundary_v8_J1QH_s1*4+head_boundary_v8_J1QH_s2*4+head_boundary_v8_J1QH_s3*4"
{ [ "$CAND" = F17K ] || [ "$CAND" = F18K ]; } && REG=${REG//J1QH_/J1QHCS_}
H=$(for s in 1 2 3 4 5 6 7 8; do printf "head_boundary_v8_J1RLHA_s$s*3+"; done)"$REG"
Q=$(for s in 1 2 3; do printf "head_qb_QHAR2Hu_s$s+"; done | sed 's/+$//')
if [ "$CAND" != F15K ]; then
  H="route:$H|$(for s in 1 2 3 4; do printf "head_boundary_v8_J1RLHACS_s$s*6+"; done)$REG"
  Q="route:$Q|head_qb_QHAR2HuCS_s1+head_qb_QHAR2HuCS_s2"
fi
if [ "$CAND" = F18K ]; then                                                  # + V60 s1-2 on both routes (every shot)
  VV="head_boundary_v8_V60_s1*6.75+head_boundary_v8_V60_s2*6.75"; H="${H%%|*}+$VV|${H#*|}+$VV"
fi
export QW_LITREE=$PWD/runs/strat/fusion/$([ "$QW_FOLD" = diii_d_public_test ] && echo litree.npz || echo litree_private.npz):0.2
XH="head_xp_XPA_s1+head_xp_XPA_s2+head_xp_XPA_s3"; { [ "$CAND" = F17K ] || [ "$CAND" = F18K ]; } && XH=${XH//XPA_/XPACS_}
KW='{"_steps":["shift","bw","xhe","kw","xp","jgn","li","li2"],"xp":{"sigma":0.15,"cap":0.1},"li":{"cap":0.9},"xhe":{"gate":0.0,"w":0.75}}'
QW_ENTRY_DIR=trunks_all .venv/bin/python paper/strat/fusion/gpu_build.py --split public --tag "$1" --psi psiMixA --guard psiFA:0.25 --qb qh50 --qbw 0.8 --qbheads "$Q" \
  --heads "$H" --xpheads "$XH" --kw "$KW" --axis head --qbclamp
.venv/bin/python paper/strat/fusion/post_kw2.py --src "$1" --tag "${1}K" --heads "$H"     # -> submission_<tag>K (the submitted file)
