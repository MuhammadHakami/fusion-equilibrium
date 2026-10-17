#!/bin/bash
# Fusion Final-phase runbook, part 1: every inputs-only feature block for a test fold (QW_FOLD=<data dir name>, e.g.
# diii_d_private_test). Files land as *_private.npy (or *_public.npy for diii_d_public_test). Then gpu_build.py --split public with the
# same QW_FOLD writes submission_<tag>/diii_d_private_test.npz.   QW_FOLD=diii_d_private_test runs/final_fold.sh [thp_psi]
cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1; : "${QW_FOLD:?set QW_FOLD}"; export QW_FOLD
P=.venv/bin/python; F=paper/strat/fusion; set -e
$P paper/sophelio/extract.py --split $QW_FOLD --workers 8                  # per-shot npz (inputs only)
$P $F/mag_fix.py --target public                                          # X_magfix_<tag>.npy (corrected magnetics)
$P $F/ip_fix.py --fold-only                                               # X_ipfix_<tag>.npy (organizers' Ip time fix; canon sign)
$P $F/thz_feats.py --public                                               # thz_<tag>.npy (fixed-coordinate Thomson)
$P $F/hist_full.py --fold-only                                            # H_full_<tag>.npy (full-rate history)
$P $F/fut_feats.py --fold-only                                            # F_fut_<tag>.npy (future context)
$P $F/thp_feats.py raw --public                                           # thp channels
TAG=$([ "$QW_FOLD" = diii_d_public_test ] && echo public || echo private)
$P $F/liphys_feats.py                                                     # liphys_<tag>.npy (li current-profile proxies, from thz; §42p, F5c+)
[ -n "$QW_QSHAPE" ] && $P $F/qshape_feats.py --fold $TAG               # opt-in: qshape_<tag>.npy (QHAR2S only, needs the J1AH heads; F14K-F16K never read it)
[ -n "$1" ] && $P $F/thp_feats.py feat --psi $1 --public                  # thp_<tag>_<psi>.npy (needs the GPU)
QW_TREE_STEM=${QW_TREE_STEM:-head_boundary_v8_J1RLHA_s1} $P $F/gbdt_shape.py dumppub   # gbdt_shape_X<tag>.npy: li-tree inputs (J1RLHA pipeline = J1RL, bit-identical)
.venv-tdc/bin/python $F/gbdt_shape.py litreepred                          # litree_<tag>.npz: SAVED all-data li tree (no refit; F14+)
echo "FINAL_FOLD_FEATURES_DONE $QW_FOLD $(date -Is)"
