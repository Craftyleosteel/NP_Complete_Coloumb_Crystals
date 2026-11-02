import Mathlib

/-!
# The four-point diagonal selector at the Coulomb exponent `s = 1`

Four candidate sites at `c + (±b, ±b)`. All energies below are in units of `b⁻¹`:
a side pair (distance `2b`) costs `1/2`, a diagonal pair (distance `2√2 b`) costs `1/(2√2)`.
`I j` is the minimum internal energy of `j` selected corners. Every repair move used in the
two-of-four forcing lemma (Emmerich, Lemma 4.1) has a strictly positive internal gain.
-/

namespace CoulombHardness

noncomputable def sidePair : ℝ := 1 / 2
noncomputable def diagPair : ℝ := 1 / (2 * Real.sqrt 2)

noncomputable def I2 : ℝ := diagPair
noncomputable def I3 : ℝ := 2 * sidePair + diagPair
noncomputable def I4 : ℝ := 4 * sidePair + 2 * diagPair

lemma sqrt_two_gt_five_fourths : (5 / 4 : ℝ) < Real.sqrt 2 := by
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have h0 := Real.sqrt_nonneg 2
  nlinarith

lemma diagPair_pos : 0 < diagPair := by
  unfold diagPair; positivity

lemma diagPair_lt_two_fifths : diagPair < 2 / 5 := by
  have h := sqrt_two_gt_five_fourths
  have := one_div_lt_one_div_of_lt (by norm_num : (0 : ℝ) < 5 / 2) (by linarith : (5 / 2 : ℝ) < 2 * Real.sqrt 2)
  unfold diagPair
  norm_num at this ⊢
  linarith

/-- Side pair → diagonal pair (the smallest gain, `≈ 0.146 b⁻¹`). -/
lemma gain_side_to_diag : 1 / 10 < sidePair - diagPair := by
  unfold sidePair; linarith [diagPair_lt_two_fifths]

/-- Occupancy transfer `(3,1) → (2,2)`. -/
lemma gain_31_22 : 0 < I3 - 2 * I2 := by
  unfold I3 I2 sidePair; linarith [diagPair_lt_two_fifths]

/-- Occupancy transfer `(3,0) → (2,1)`. -/
lemma gain_30_21 : 0 < I3 - I2 := by
  unfold I3 I2 sidePair; linarith [diagPair_pos]

/-- Occupancy transfer `(4,1) → (3,2)`. -/
lemma gain_41_32 : 0 < I4 - I3 - I2 := by
  unfold I4 I3 I2 sidePair; linarith

/-- Occupancy transfer `(4,0) → (3,1)`. -/
lemma gain_40_31 : 0 < I4 - I3 := by
  unfold I4 I3 sidePair; linarith [diagPair_pos]

/-- With `M` cells and `b ≤ 1/(40 M)`, the smallest internal gain `> 0.1/b` beats the
adverse cross-cell change, which is bounded by `4M` (cell centres pairwise ≥ 1 apart). -/
lemma forcing_condition (M : ℕ) (b : ℝ) (hb : 0 < b) (hbM : b * (40 * M) ≤ 1) :
    4 * (M : ℝ) < (sidePair - diagPair) / b := by
  rw [lt_div_iff₀ hb]
  have := gain_side_to_diag
  nlinarith

end CoulombHardness
