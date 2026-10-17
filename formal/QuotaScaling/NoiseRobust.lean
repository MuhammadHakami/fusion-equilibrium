import Mathlib

/-!
# Reaching the ceiling under noise (doc 35 §48c-d; user Oct 17: "1.0 under uncertainty", noise modelling and robustness)

`ScoreCeiling` shows the labels carry slice noise `e` (EFIT fits every time slice separately), so 1.0 is not reachable; the best
any model can do is the noise floor. This file states what "reaching the highest possible under noise" means and which knobs of the
VisionHOPE + input-dependent register hybrid act on it.

1. `excess_decomp`, `reach_floor_iff`: with labels `y = m + e` and noise uncorrelated with the predictable part, the squared error
   splits into the floor `Σ e²` plus the excess `Σ (m - f)²`; the floor is reached iff the predictor equals `m` on every frame.
   Everything a model choice can win is in the excess.
2. `hybrid_le_classical`: a register branch with a free amplitude α (α = 0 is the classical model) never costs capacity - the
   fitted hybrid is at most the classical model's loss (the paper's zero-init gate argument, any loss).
3. `stratum_blend_le`: blend weights chosen per noise regime (ramp-up / flat-top / ramp-down, Ip sign) do at least as well as one
   pooled weight (CompetitionBlend per stratum).
4. `antithetic_noise`: input-noise injection on a linear read-out equals the clean loss plus a sensitivity penalty `⟪w, ε⟫²` - training
   with sensor noise buys robustness to sensor noise (Tikhonov form).
5. `huber_le_sq`, `huber_le_lin`: the Huber loss never exceeds the squared loss and grows only linearly, so a label jump of size R
   (ELM, sawtooth) costs at most δR instead of R²/2 (bounded influence).
6. `register_shift_le`: the register enters as α Σ W_j ⟨Z_j⟩ with every ⟨Z_j⟩ ∈ [-1, 1]; any physical noise realisation (atom
   positions x, y, z, detuning, Rabi amplitude, read-out) moves the hybrid's output by at most 2 |α| Σ |W_j| - a certified bound
   that the run reports from the trained weights.
-/

namespace NoiseRobust

open Finset

/-- 1a. Floor + excess split when the noise is orthogonal to the excess (finite sample). -/
theorem excess_decomp {ι : Type*} (s : Finset ι) (m e f : ι → ℝ) (horth : ∑ i ∈ s, e i * (m i - f i) = 0) :
    ∑ i ∈ s, (m i + e i - f i) ^ 2 = ∑ i ∈ s, e i ^ 2 + ∑ i ∈ s, (m i - f i) ^ 2 := by
  have : ∀ i ∈ s, (m i + e i - f i) ^ 2 = e i ^ 2 + (m i - f i) ^ 2 + 2 * (e i * (m i - f i)) := fun i _ => by ring
  rw [sum_congr rfl this, sum_add_distrib, sum_add_distrib, ← mul_sum, horth]; ring

/-- 1b. The floor is reached iff the predictor equals the predictable part on every frame. -/
theorem reach_floor_iff {ι : Type*} (s : Finset ι) (m e f : ι → ℝ) (horth : ∑ i ∈ s, e i * (m i - f i) = 0) :
    ∑ i ∈ s, (m i + e i - f i) ^ 2 = ∑ i ∈ s, e i ^ 2 ↔ ∀ i ∈ s, f i = m i := by
  rw [excess_decomp s m e f horth]
  constructor
  · intro h i hi
    have h0 : ∑ j ∈ s, (m j - f j) ^ 2 = 0 := by linarith
    have := (sum_eq_zero_iff_of_nonneg fun j _ => sq_nonneg (m j - f j)).1 h0 i hi
    have : m i - f i = 0 := by simpa using this
    linarith
  · intro h
    have : ∑ j ∈ s, (m j - f j) ^ 2 = 0 := sum_eq_zero fun j hj => by rw [h j hj]; ring
    linarith

/-- 2. The fitted register amplitude is never worse than switching the register off (α = 0 = the classical model). -/
theorem hybrid_le_classical {ι : Type*} (L : (ι → ℝ) → ℝ) (c q : ι → ℝ) (αs : ℝ)
    (hopt : ∀ α : ℝ, L (fun i => c i + αs * q i) ≤ L (fun i => c i + α * q i)) : L (fun i => c i + αs * q i) ≤ L c := by
  have := hopt 0; simpa using this

/-- 3. Per-stratum blend weights are at least as good as one pooled weight. -/
theorem stratum_blend_le {σ : Type*} (S : Finset σ) (F : σ → ℝ → ℝ) (ws : σ → ℝ) (w : ℝ)
    (hbest : ∀ s ∈ S, F s (ws s) ≤ F s w) : ∑ s ∈ S, F s (ws s) ≤ ∑ s ∈ S, F s w :=
  sum_le_sum hbest

/-- 4. Antithetic input noise on a linear read-out: clean loss plus a sensitivity penalty. -/
theorem antithetic_noise {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (w x ε : E) (y : ℝ) :
    ((@inner ℝ _ _ w (x + ε) - y) ^ 2 + (@inner ℝ _ _ w (x - ε) - y) ^ 2) / 2
      = (@inner ℝ _ _ w x - y) ^ 2 + (@inner ℝ _ _ w ε) ^ 2 := by
  rw [inner_add_right, inner_sub_right]; ring

/-- The Huber loss with threshold δ. -/
noncomputable def huber (δ r : ℝ) : ℝ := if |r| ≤ δ then r ^ 2 / 2 else δ * (|r| - δ / 2)

/-- 5a. Huber never exceeds the squared loss. -/
theorem huber_le_sq (δ r : ℝ) : huber δ r ≤ r ^ 2 / 2 := by
  unfold huber; split_ifs with h
  · exact le_rfl
  · push_neg at h; nlinarith [sq_abs r, sq_nonneg (|r| - δ)]

/-- 5b. Huber grows at most linearly (bounded influence of label jumps). -/
theorem huber_le_lin (δ r : ℝ) (hδ : 0 ≤ δ) : huber δ r ≤ δ * |r| := by
  unfold huber; split_ifs with h
  · nlinarith [abs_nonneg r, sq_abs r]
  · nlinarith

/-- 6. Certified worst-case effect of any register noise realisation: read-outs ⟨Z_j⟩, ⟨Z_j⟩' in [-1, 1]. -/
theorem register_shift_le {n : ℕ} (α : ℝ) (W z z' : Fin n → ℝ) (hz : ∀ j, |z j| ≤ 1) (hz' : ∀ j, |z' j| ≤ 1) :
    |α * ∑ j, W j * z j - α * ∑ j, W j * z' j| ≤ 2 * |α| * ∑ j, |W j| := by
  have key : |∑ j, (W j * z j - W j * z' j)| ≤ 2 * ∑ j, |W j| :=
    calc |∑ j, (W j * z j - W j * z' j)| ≤ ∑ j, |W j * z j - W j * z' j| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ j, 2 * |W j| := sum_le_sum fun j _ => by
          rw [← mul_sub, abs_mul]
          have : |z j - z' j| ≤ 2 := by
            calc |z j - z' j| ≤ |z j| + |z' j| := abs_sub _ _
              _ ≤ 2 := by linarith [hz j, hz' j]
          nlinarith [abs_nonneg (W j)]
      _ = 2 * ∑ j, |W j| := by rw [mul_sum]
  calc |α * ∑ j, W j * z j - α * ∑ j, W j * z' j| = |α| * |∑ j, (W j * z j - W j * z' j)| := by
        rw [← mul_sub, abs_mul, ← sum_sub_distrib]
    _ ≤ |α| * (2 * ∑ j, |W j|) := mul_le_mul_of_nonneg_left key (abs_nonneg α)
    _ = 2 * |α| * ∑ j, |W j| := by ring

end NoiseRobust
