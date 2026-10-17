# Fusion Equilibrium Challenge 2026 — DIII-D entry (F18K)

Predicts, for every EFIT frame of a DIII-D shot, the 65×65 poloidal flux map ψ(R, Z), q95 and βN from the challenge inputs only
(magnetic coil currents, Ip, Thomson scattering). The model is a hybrid classical + quantum ensemble. Its newest member is a
**VisionHOPE-T backbone that reads the whole shot as a frames × tokens map**, gated into a per-frame MLP trunk, with a
**simulated 30-atom Rydberg register whose detunings and drives depend on each input** in the head. Next to it run MLP heads,
MLP + register heads, X-point heads and q95 / βN heads. The predicted ψ is then moved onto the predicted plasma shape.

![F18K architecture: inputs, the VisionHOPE + Rydberg-register shape head, and the ensemble / psi assembly](repro/fusion/model.png)

<sub>Every size in the figure is read from the trained PyTorch checkpoints and checked against them by
[`repro/fusion/fig_model.py`](repro/fusion/fig_model.py) (`.venv/bin/python repro/fusion/fig_model.py`).</sub>

## Inputs and outputs (per EFIT frame)

| | |
|---|---|
| **Inputs** (599 for the shape heads) | 21 magnetic channels (18 F-coil currents, E-coil, B-coil, Ip) at t and 13 offsets from −200 ms to +20 ms, corrected (294); Thomson Te, ne, pe core / edge statistics and profiles (132); time (1); Thomson on a fixed R / Z grid (105); full-rate current history (24); Spitzer current-profile proxies Te^1.5 (13); physics scalings and the 19 coil currents / \|Ip\| (29); sign(Ip) (1). Reversed-Ip shots use currents × sign(Ip). |
| **VisionHOPE head view** | All frames of the shot at once: 32-frame crops in training, 32-frame tiles at stride 16 at inference (overlaps averaged). Each frame adds its time since the shot start and a valid flag (601 channels). |
| **Internal targets** | 7 shape scalars (R and Z of the axis, κ, upper / lower triangularity, volume, li), 64 LCFS rays, the lower / upper X-points (R, Z), q95, βN, and 64 PCA coefficients of ψ. |
| **Submitted outputs** | ψ(R, Z) 65 × 65, q95, βN. |
| **Score** | S = 0.55 R²ψ + 0.15 R²(q95, βN) + 0.10 (1 − D_LCFS) + 0.20 · mean R² of the 7 shape scalars. |

## Results

| Configuration | Development public S | Paired gain vs F17K, 150 held-out training shots |
|---|---|---|
| F17K (MLP + register heads, Ip-sign canonical twins) | 0.996223 | — |
| **F18K** = F17K + VisionHOPE + register heads (Final) | pending | S +0.000084, 95% CI [+0.000036, +0.000134] |

## How the configuration was chosen

- **Shot-level stratified cross-validation on the training shots only.** Strata are Ip sign × li quartile. The organisers' test folds are never used for fitting or selection.
- **Optuna (TPE):** head learning rate, width and depth; register read-in and layers.
- **NAS:** the q95 / βN entry-ensemble trunks ([`repro/fusion/entry_configs.json`](repro/fusion/entry_configs.json)).
- **Lean 4 + Mathlib** ([`formal/`](formal/), `cd formal && lake build`) proves the rules the selection uses:
  - `CompetitionBlend`: a head can join the blend only if ⟨e_M, e_X⟩ < ‖e_M‖², and the best weight's gain is (A − C)² / (A + B − 2C).
  - `BandGate`: the zero-init gate is never worse than the trunk alone.
  - `ScoreCeiling`: 1 − S splits into weighted per-term deficits, and a label-noise floor caps the reachable S.
  - `NoiseRobust`: the floor-plus-excess split, the Huber bound, and the certified register-noise bound 2|α| Σ|W|.
- **Adoption rule:** a component is kept only if its paired gain on 150 held-out shots has a 95% CI above zero, measured through the full deployed chain.

## Run

```bash
# 1 environment (the VisionHOPE CUDA kernels are compiled on first use: needs nvcc + ninja, pinned in the lockfile)
uv venv .venv && uv pip install --python .venv/bin/python -r repro/fusion/requirements.lock
uv venv -p 3.12 .venv-tdc && uv pip install --python .venv-tdc/bin/python catboost==1.2.10 numpy==2.5.2

# 2 data (challenge dataset) + caches: step 1 of repro/fusion/README.md
hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset --local-dir data/sophelio/fusion-equilibrium-challenge

# 3 weights (release assets, sha256-checked against repro/fusion/WEIGHTS.sha256)
.venv/bin/python repro/fusion/fetch_weights.py --base https://github.com/MuhammadHakami/fusion-equilibrium/releases/download/f18k-weights

# 4 inference -> runs/sophelio_sota/submission_F18K/diii_d_private_test.npz
QW_FOLD=diii_d_private_test runs/final_fold.sh
QW_FOLD=diii_d_private_test CAND=F18K repro/fusion/infer.sh F18

# or train every model from scratch (~23 A100-hours, estimated; GB10 timings in repro/fusion/README.md)
CAND=F18K repro/fusion/train_all.sh
```

`CAND=F17K` reproduces the previous configuration (no VisionHOPE heads).

## Layout

| Path | Contents |
|---|---|
| `paper/strat/fusion/quota_fusion.py` | VisionHOPE + register shape head (training: `final`; inference: `predict_rows`) |
| `paper/strat/fusion/head_boundary.py`, `hpo_qpikan.py` | MLP shape / X-point / q95-βN heads; MLP + register heads |
| `paper/qlayer.py` | Simulated Rydberg registers (exact statevector, per-input detunings and drive gates) |
| `paper/strat/fusion/gpu_build.py`, `post_kw2.py` | Inference, ensemble, and ψ edits toward the heads |
| `baseline/VisionHOPE/` | VisionHOPE runtime subset (MIT, vendored) |
| `baseline/sophelio-starter/` | The organisers' starter kit (MIT) |
| `formal/` | Lean 4 proofs used by the selection rules |
| `repro/fusion/` | Entry scripts, manifests (sha256), and the full component / compute / provenance README |

Details (every component, compute per component, data provenance): [`repro/fusion/README.md`](repro/fusion/README.md).
MIT licence. AI coding assistants (Claude Code) were used under the entrant's direction and responsibility.
