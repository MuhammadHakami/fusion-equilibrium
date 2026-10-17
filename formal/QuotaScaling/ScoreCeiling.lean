import Mathlib

/-!
# How high can the Fusion score go? (doc 35 §48, user Oct 17: "reach 0.9999 or 1.0 mathematically")

The official composite is `S = 0.55 R²ψ + 0.15 R²qb + 0.10 (1 - D) + 0.20 C` (each R² clipped at 0, `D` = mean normalised
LCFS Hausdorff distance, `C` = mean of the seven consistency R²). Three facts settle what any model can reach:

1. `deficit_eq`, `target_each`: `1 - S` is the weighted sum of the four deficits, so `S ≥ 1 - ε` forces every deficit
   below `ε / w` - for `ε = 10⁻⁴`: ψ ≤ 1.8·10⁻⁴, q/b ≤ 6.7·10⁻⁴, D ≤ 10⁻³, consistency ≤ 5·10⁻⁴ (each alone; jointly the
   weighted sum ≤ 10⁻⁴).
2. `pair_floor`, `sse_floor`, `cell_floor`: the data themselves bound every predictor from below. Two frames whose inputs a
   predictor cannot tell apart by more than `η` but whose labels differ by `Δ` cost it at least `(Δ - η)₊² / 2`, and disjoint
   pairs add up (a matching); frames with identical inputs cost at least their within-group sum of squares (the finite
   form of `E Var(Y | X)`). A label that is not a function of the inputs has an error floor no architecture removes.
3. `r2_ceiling`, `cons_ceiling`, `score_mono`, `unreachable`: floors turn into per-term R² ceilings, the clipped mean
   consistency inherits them, and the composite is monotone - so a ceiling below the target proves the target is
   unreachable from these inputs, whatever the model.

The floors are data (doc 35 §48: cross-shot nearest-neighbour pairs on stratified shot halves, Gamma-test intercepts);
nothing here estimates them.
-/

namespace ScoreCeiling

/-- The official composite on unclipped terms (the clipping enters through `cons_ceiling`). -/
noncomputable def score (rpsi rqb d cons : ℝ) : ℝ := 0.55 * rpsi + 0.15 * rqb + 0.10 * (1 - d) + 0.20 * cons

/-- 1a. `1 - S` is the weighted sum of the four deficits. -/
theorem deficit_eq (rpsi rqb d cons : ℝ) :
    1 - score rpsi rqb d cons = 0.55 * (1 - rpsi) + 0.15 * (1 - rqb) + 0.10 * d + 0.20 * (1 - cons) := by
  unfold score; ring

/-- 1b. Reaching `1 - ε` forces each deficit below `ε / weight` (deficits are nonnegative for R² ≤ 1, D ≥ 0). -/
theorem target_each (rpsi rqb d cons ε : ℝ) (h : 1 - ε ≤ score rpsi rqb d cons)
    (hp : rpsi ≤ 1) (hq : rqb ≤ 1) (hd : 0 ≤ d) (hc : cons ≤ 1) :
    1 - rpsi ≤ ε / 0.55 ∧ 1 - rqb ≤ ε / 0.15 ∧ d ≤ ε / 0.10 ∧ 1 - cons ≤ ε / 0.20 := by
  have e := deficit_eq rpsi rqb d cons
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [le_div_iff₀ (by norm_num)] <;> nlinarith

/-- 2a. Pair floor: if a predictor's outputs on two inputs differ by at most `η`, its two squared errors sum to at least
`(|y₁ - y₂| - η)₊² / 2`. -/
theorem pair_floor (y1 y2 f1 f2 η : ℝ) (hf : |f1 - f2| ≤ η) :
    (max 0 (|y1 - y2| - η)) ^ 2 / 2 ≤ (y1 - f1) ^ 2 + (y2 - f2) ^ 2 := by
  have h1 : |y1 - y2| ≤ |(y1 - f1) - (y2 - f2)| + |f1 - f2| := by
    have : y1 - y2 = ((y1 - f1) - (y2 - f2)) + (f1 - f2) := by ring
    rw [this]; exact abs_add _ _
  have hm : max 0 (|y1 - y2| - η) ≤ |(y1 - f1) - (y2 - f2)| := max_le (abs_nonneg _) (by linarith)
  have h0 : 0 ≤ max 0 (|y1 - y2| - η) := le_max_left _ _
  have hsq : (max 0 (|y1 - y2| - η)) ^ 2 ≤ ((y1 - f1) - (y2 - f2)) ^ 2 := by
    rw [← sq_abs ((y1 - f1) - (y2 - f2))]; nlinarith
  nlinarith [sq_nonneg ((y1 - f1) + (y2 - f2))]

/-- 2b. Matching floor: disjoint pairs `(a i, b i)` (the map `Sum.elim a b` is injective) whose pair costs are bounded
below by `lb i` give a lower bound on the total squared error over all frames. -/
theorem sse_floor {α ι : Type*} [Fintype α] [Fintype ι] [DecidableEq α] (e : α → ℝ) (he : ∀ x, 0 ≤ e x)
    (a b : ι → α) (hinj : Function.Injective (Sum.elim a b)) (lb : ι → ℝ) (hlb : ∀ i, lb i ≤ e (a i) + e (b i)) :
    ∑ i, lb i ≤ ∑ x, e x := by
  calc ∑ i, lb i ≤ ∑ i, (e (a i) + e (b i)) := Finset.sum_le_sum fun i _ => hlb i
    _ = ∑ s : ι ⊕ ι, e (Sum.elim a b s) := by rw [Fintype.sum_sum_type]; simp [Finset.sum_add_distrib]
    _ = ∑ x ∈ Finset.univ.image (Sum.elim a b), e x := (Finset.sum_image fun x _ y _ h => hinj h).symm
    _ ≤ ∑ x, e x := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun x _ _ => he x

/-- 2c. Cell floor (identical inputs): any single value `c` predicted for a group costs at least the group's sum of
squares about its own mean. -/
theorem cell_floor {ι : Type*} (s : Finset ι) (y : ι → ℝ) (c : ℝ) (hs : s.Nonempty) :
    ∑ i ∈ s, (y i - (∑ j ∈ s, y j) / s.card) ^ 2 ≤ ∑ i ∈ s, (y i - c) ^ 2 := by
  set m := (∑ j ∈ s, y j) / s.card
  have hn : (0 : ℝ) < s.card := by exact_mod_cast hs.card_pos
  have hsum : ∑ i ∈ s, (y i - m) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]; simp only [m]; field_simp
  have key : ∑ i ∈ s, (y i - c) ^ 2 = ∑ i ∈ s, (y i - m) ^ 2 + s.card * (m - c) ^ 2 := by
    have : ∀ i ∈ s, (y i - c) ^ 2 = (y i - m) ^ 2 + 2 * (m - c) * (y i - m) + (m - c) ^ 2 := fun i _ => by ring
    rw [Finset.sum_congr rfl this, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hsum,
      Finset.sum_const, nsmul_eq_mul]; ring
  rw [key]; nlinarith [sq_nonneg (m - c)]

/-- 3a. A floor on the error is a ceiling on R². -/
theorem r2_ceiling (sse sst floor : ℝ) (hsst : 0 < sst) (h : floor ≤ sse) : 1 - sse / sst ≤ 1 - floor / sst := by
  have := div_le_div_of_nonneg_right h hsst.le; linarith

/-- 3b. The clipped mean consistency inherits per-scalar ceilings `c k ≥ 0`. -/
theorem cons_ceiling (r c : Fin 7 → ℝ) (hr : ∀ k, r k ≤ c k) (hc : ∀ k, 0 ≤ c k) :
    (∑ k, max 0 (r k)) / 7 ≤ (∑ k, c k) / 7 := by
  have : ∑ k, max 0 (r k) ≤ ∑ k, c k := Finset.sum_le_sum fun k _ => max_le (hc k) (hr k)
  linarith

/-- 3c. The composite is monotone in every term. -/
theorem score_mono {rpsi rqb d cons cpsi cqb dmin ccons : ℝ} (hp : rpsi ≤ cpsi) (hq : rqb ≤ cqb) (hd : dmin ≤ d)
    (hc : cons ≤ ccons) : score rpsi rqb d cons ≤ score cpsi cqb dmin ccons := by
  unfold score; nlinarith

/-- 3d. A ceiling below the target makes the target unreachable for every predictor obeying the floors. -/
theorem unreachable {rpsi rqb d cons cpsi cqb dmin ccons t : ℝ} (hp : rpsi ≤ cpsi) (hq : rqb ≤ cqb) (hd : dmin ≤ d)
    (hc : cons ≤ ccons) (ht : score cpsi cqb dmin ccons < t) : score rpsi rqb d cons < t :=
  lt_of_le_of_lt (score_mono hp hq hd hc) ht

end ScoreCeiling
