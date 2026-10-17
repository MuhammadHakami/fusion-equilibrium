"""Fusion §42a: inputs-only physics scalings for the q95 / betaN heads, appended to a data object's inputs.

q95 ~ a^2 B_t / (R I_p) * f(kappa) and betaN = beta_t a B_t / I_p with beta_t ~ <p> / B_t^2: both are products / ratios of
inputs that an MLP on standardised raw currents must otherwise approximate. Computed on the fly from the (already corrected:
inputs_fix / canon run first) base columns, lag 0, so the TRAIN and public / private paths are the same code:
  log|B| - log|Ip| (q-like), log|Ip|, log|B|, log <ne Te> (chord moments), log <ne Te> - log|Ip| - log|B| (betaN-like),
  log <ne Te> - 2 log|Ip| (beta_p-like), log <ne> - log|Ip| (Greenwald-like), log Te_p50, log ne_p50.
    import qphys_inputs as QP; QP.use(data)        # wrap after inputs_fix / canon / hfull_inputs
"""
from __future__ import annotations

import numpy as np

NAMES = ["q_like", "log_ip", "log_b", "log_p_mean", "log_p_prof", "betan_like", "betap_like", "greenwald_like",
         "log_te_p50", "log_ne_p50"]


def feats(x: np.ndarray, cols: list[str]) -> np.ndarray:
    ix = {n: i for i, n in enumerate(cols)}; g = lambda n: np.asarray(x[:, ix[n]], np.float64)
    ip, b = np.abs(g("mag_plasma_current")) + 10.0, np.abs(g("mag_bcoil")) + 1.0          # kA; floors keep ramps finite
    te, ne = np.clip(g("thomson_core_Te_mean"), 0, None), np.clip(g("thomson_core_ne_mean"), 0, None) / 1e19
    tp = np.stack([np.clip(g(f"thomson_core_Te_prof{k}"), 0, None) for k in range(24) if f"thomson_core_Te_prof{k}" in ix], 1)
    npf = np.stack([np.clip(g(f"thomson_core_ne_prof{k}"), 0, None) for k in range(24) if f"thomson_core_ne_prof{k}" in ix], 1) / 1e19
    pm, pp = np.log(te * ne + 1.0), np.log((tp[:, :npf.shape[1]] * npf[:, :tp.shape[1]]).mean(1) + 1.0)
    lip, lb = np.log(ip), np.log(b)
    F = np.stack([lb - lip, lip, lb, pm, pp, pp - lip - lb, pp - 2 * lip, np.log(ne + 0.01) - lip,
                  np.log(np.clip(g("thomson_core_Te_p50"), 0, None) + 1.0), np.log(np.clip(g("thomson_core_ne_p50"), 0, None) / 1e19 + 0.01)], 1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


COILS = ["mag_ECOILA"] + [f"mag_F{i}{s}" for i in range(1, 10) for s in "AB"]


def feats_ratio(x: np.ndarray, cols: list[str]) -> np.ndarray:
    """§42e: coil currents over |Ip| (lag 0). Radial force balance: B_v ~ mu0 Ip (ln 8R/a + beta_p + li/2 - 3/2) / (4 pi R), and the
    vertical field is a fixed linear combination of the coil currents, so beta_p + li/2 is LINEAR in I_c / Ip (Shafranov)."""
    ix = {n: i for i, n in enumerate(cols)}
    ip = np.abs(np.asarray(x[:, ix["mag_plasma_current"]], np.float64)) + 100.0          # kA; floor keeps ramps bounded
    F = np.stack([np.asarray(x[:, ix[c]], np.float64) / ip for c in COILS if c in ix], 1)
    return np.clip(np.nan_to_num(F), -50, 50).astype(np.float32)


def use(data, ratios: bool = False):
    inner, fn = data.inputs, list(data.fn)

    def inputs(rows, fm, edge=False, hist=False, thz=False):
        x = inner(rows, fm, edge, hist=hist, thz=thz) if (hist or thz) else inner(rows, fm, edge)
        cols = [n for n, m in zip(fn, fm) if m]; base = np.asarray(x[:, :len(cols)])
        extra = [feats(base, cols)] + ([feats_ratio(base, cols)] if ratios else [])
        return np.concatenate([np.asarray(x, np.float32)] + extra, 1)
    data.inputs = inputs
    return data
