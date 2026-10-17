import Mathlib
import QuotaScaling.BandGate

/-!
# When does a new model belong in the ensemble? (doc 35 §47, Fusion weak-spot rule)

The Fusion heads' errors are 0.90-0.97 correlated across algorithms (§41s), so a new backbone earns its place only through
the part of its error the ensemble does not share. Write `u` for the current ensemble's error vector on a target (held-out
frames) and `v` for the candidate's. A blend with weight `w` on the candidate has error `(1 - w) • u + w • v = u - w • d`,
`d = u - v`, so every statement reduces to three out-of-fold statistics A = ‖u‖², B = ‖v‖², C = ⟪u, v⟫:
⟪u, d⟫ = A - C and ‖d‖² = A + B - 2C.

1. `blend_error_eq`: the blended error is the gated residual of `BandGate` (`u - w • d`).
2. `blend_helps_iff`: some positive weight lowers the SSE iff ⟪u, d⟫ > 0, i.e. C < A: the candidate's error must be
   less aligned with the ensemble's error than the ensemble's error is with itself. Correlation alone does not decide.
3. `blend_best_gain`: the best weight w* = (A - C) / (A + B - 2C) lowers the SSE by exactly (A - C)² / (A + B - 2C), and no
   weight does better (`BandGate.gate_opt_le`). The §47 read-out reports this number per target from out-of-fold errors.

Nothing here predicts that any model helps; the data supply A, B, C.
-/

namespace CompetitionBlend

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- 1. A convex blend's error is the gated residual with `d = u - v`. -/
theorem blend_error_eq (u v : E) (w : ℝ) : (1 - w) • u + w • v = u - w • (u - v) := by
  rw [sub_smul, one_smul, smul_sub]; abel

/-- 2. A candidate helps at some positive weight iff ⟪u, u - v⟫ > 0 (C < A). -/
theorem blend_helps_iff (u v : E) :
    (∃ w : ℝ, 0 < w ∧ ‖u - w • (u - v)‖ ^ 2 < ‖u‖ ^ 2) ↔ 0 < @inner ℝ _ _ u (u - v) := by
  constructor
  · rintro ⟨w, hw, h⟩
    rw [BandGate.gate_energy_exact] at h
    by_contra hn; push_neg at hn
    nlinarith [sq_nonneg w, sq_nonneg ‖u - v‖, mul_nonneg (sq_nonneg w) (sq_nonneg ‖u - v‖), mul_nonneg hw.le (neg_nonneg.mpr hn)]
  · intro hp
    have hd : u - v ≠ 0 := by
      intro h0; rw [h0, inner_zero_right] at hp; exact lt_irrefl 0 hp
    have hD : 0 < ‖u - v‖ ^ 2 := by positivity
    refine ⟨@inner ℝ _ _ u (u - v) / ‖u - v‖ ^ 2, div_pos hp hD, ?_⟩
    rw [BandGate.gate_energy_exact]
    have key : 2 * (@inner ℝ _ _ u (u - v) / ‖u - v‖ ^ 2) * @inner ℝ _ _ u (u - v)
        - (@inner ℝ _ _ u (u - v) / ‖u - v‖ ^ 2) ^ 2 * ‖u - v‖ ^ 2 = (@inner ℝ _ _ u (u - v)) ^ 2 / ‖u - v‖ ^ 2 := by
      field_simp; ring
    have pos : 0 < (@inner ℝ _ _ u (u - v)) ^ 2 / ‖u - v‖ ^ 2 := div_pos (by positivity) hD
    linarith

/-- 3. The best blend: its SSE is ‖u‖² - ⟪u, d⟫² / ‖d‖² and no weight is lower. -/
theorem blend_best_gain (u v : E) (hd : u - v ≠ 0) (w : ℝ) :
    ‖u - (@inner ℝ _ _ u (u - v) / ‖u - v‖ ^ 2) • (u - v)‖ ^ 2
        = ‖u‖ ^ 2 - (@inner ℝ _ _ u (u - v)) ^ 2 / ‖u - v‖ ^ 2
      ∧ ‖u‖ ^ 2 - (@inner ℝ _ _ u (u - v)) ^ 2 / ‖u - v‖ ^ 2 ≤ ‖u - w • (u - v)‖ ^ 2 := by
  have hD : 0 < ‖u - v‖ ^ 2 := by positivity
  refine ⟨?_, BandGate.gate_opt_le u (u - v) hd w⟩
  rw [BandGate.gate_energy_exact]; field_simp; ring

end CompetitionBlend
