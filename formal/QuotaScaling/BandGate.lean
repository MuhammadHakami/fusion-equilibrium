import Mathlib

/-!
# Band-wise ceilings and the persistence/trunk gate (docs 35 §8, 36)

Formal backbone of the frequency-aware co-processor (ADR-C v2) and of the classical per-window gate. Every band is
the range of an orthogonal projector; inside it the co-processor works on an orthonormal family of band-localized
modes, and the gate rescales the trunk's increment over persistence per band. Nothing here claims that any learned
component predicts well, and nothing is a claim of quantum advantage.

1. `band_span_le_energy` (Bessel inside a band): the energy on any orthonormal family of modes is at most the band's
   energy, so span_b ≤ full_b. With `Utilization.gain_le_ceiling` (captured_b ≤ span_b) this is the per-band chain
   captured_b ≤ span_b ≤ full_b reported after convergence.
2. `gate_energy_exact` / `gate_coeff_exact`: for any gate α the residual energy and every mode coefficient of
   u − α d (u = P_b(y − x_t), d = P_b(p − x_t)) are exact functions of three cached band statistics and two cached
   coefficients, which is why the gate trains on the exact metric.
3. `gate_opt_le`: the best scalar gate is at least as good as every scalar gate, in particular persistence (α = 0)
   and the trunk (α = 1): the per-window band-gate ceiling dominates both anchors band by band.
-/

namespace BandGate

variable {ι E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Bessel inside a band: energy on orthonormal band modes ≤ the band's energy (span_b ≤ full_b). -/
theorem band_span_le_energy {v : ι → E} (hv : Orthonormal ℝ v) (s : Finset ι) (x : E) :
    ∑ i ∈ s, (@inner ℝ _ _ (v i) x) ^ 2 ≤ ‖x‖ ^ 2 := by
  have h := hv.sum_inner_products_le (s := s) (x := x)
  simpa [Real.norm_eq_abs, sq_abs] using h

/-- The gated residual's energy from three cached statistics: ‖u‖², ⟪u, d⟫, ‖d‖². -/
theorem gate_energy_exact (u d : E) (α : ℝ) :
    ‖u - α • d‖ ^ 2 = ‖u‖ ^ 2 - 2 * α * @inner ℝ _ _ u d + α ^ 2 * ‖d‖ ^ 2 := by
  rw [@norm_sub_sq_real, inner_smul_right, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  ring

/-- Every mode coefficient of the gated residual from two cached coefficients. -/
theorem gate_coeff_exact (φ u d : E) (α : ℝ) :
    @inner ℝ _ _ φ (u - α • d) = @inner ℝ _ _ φ u - α * @inner ℝ _ _ φ d := by
  rw [inner_sub_right, inner_smul_right]

/-- The optimal scalar gate dominates every gate, hence both persistence (α = 0) and the trunk (α = 1). -/
theorem gate_opt_le (u d : E) (hd : d ≠ 0) (α : ℝ) :
    ‖u‖ ^ 2 - (@inner ℝ _ _ u d) ^ 2 / ‖d‖ ^ 2 ≤ ‖u - α • d‖ ^ 2 := by
  have hD : 0 < ‖d‖ ^ 2 := by positivity
  rw [gate_energy_exact, sub_le_iff_le_add, ← sub_le_iff_le_add', le_div_iff₀ hD]
  nlinarith [sq_nonneg (α * ‖d‖ ^ 2 - @inner ℝ _ _ u d)]

/-- Specializations: the per-band oracle gate is no worse than persistence or the trunk. -/
theorem gate_opt_le_persistence (u d : E) (hd : d ≠ 0) :
    ‖u‖ ^ 2 - (@inner ℝ _ _ u d) ^ 2 / ‖d‖ ^ 2 ≤ ‖u‖ ^ 2 := by
  simpa using gate_opt_le u d hd 0

theorem gate_opt_le_trunk (u d : E) (hd : d ≠ 0) :
    ‖u‖ ^ 2 - (@inner ℝ _ _ u d) ^ 2 / ‖d‖ ^ 2 ≤ ‖u - d‖ ^ 2 := by
  simpa using gate_opt_le u d hd 1

end BandGate
