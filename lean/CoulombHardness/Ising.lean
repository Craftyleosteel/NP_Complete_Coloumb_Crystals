import Mathlib

/-!
# Spin-level facts used by the geometric reduction

* `two_spin_fourier`: every function of two `±1` spins is `A + h₁σ + h₂τ + Jστ` exactly
  (write-up §6). This is why the energy of any legal selector configuration is an Ising energy.
* `energy_flip`: flipping the spins of a set `C` changes an Ising energy by
  `−2 · (couplings across the cut) − 2 · (fields inside C)`.
* `flip_decreases`: if a ferromagnetic edge `ab` across the cut is in a defect state and its
  coupling dominates every other cross-cut coupling plus every field inside `C`, the flip strictly
  lowers the energy. This is the combinatorial core of tree normalisation (write-up Lemma 9.2);
  the geometry only has to supply the numerical budget (Lemma 9.1).
-/

namespace CoulombHardness

/-! ### Two-spin Fourier decomposition -/

section Fourier

variable (f : ℝ → ℝ → ℝ)

noncomputable def fA : ℝ := (f 1 1 + f 1 (-1) + f (-1) 1 + f (-1) (-1)) / 4
noncomputable def fH₁ : ℝ := (f 1 1 + f 1 (-1) - f (-1) 1 - f (-1) (-1)) / 4
noncomputable def fH₂ : ℝ := (f 1 1 - f 1 (-1) + f (-1) 1 - f (-1) (-1)) / 4
noncomputable def fJ : ℝ := (f 1 1 - f 1 (-1) - f (-1) 1 + f (-1) (-1)) / 4

theorem two_spin_fourier {σ τ : ℝ} (hσ : σ = 1 ∨ σ = -1) (hτ : τ = 1 ∨ τ = -1) :
    f σ τ = fA f + fH₁ f * σ + fH₂ f * τ + fJ f * σ * τ := by
  rcases hσ with rfl | rfl <;> rcases hτ with rfl | rfl <;> (unfold fA fH₁ fH₂ fJ; ring)

end Fourier

/-! ### Ising energies and component flips -/

section Flip

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Ising energy with coupling matrix `J` (each unordered pair counted once via the `1/2`) and
fields `h`. -/
noncomputable def isingEnergy (J : ι → ι → ℝ) (h : ι → ℝ) (σ : ι → ℝ) : ℝ :=
  (∑ i, ∑ j, J i j * σ i * σ j) / 2 + ∑ i, h i * σ i

/-- Flip every spin in `C`. -/
def flipOn (C : Finset ι) (σ : ι → ℝ) : ι → ℝ := fun i => if i ∈ C then -σ i else σ i

/-- Coupling sum between two blocks. -/
noncomputable def block (J : ι → ι → ℝ) (σ : ι → ℝ) (s t : Finset ι) : ℝ :=
  ∑ i ∈ s, ∑ j ∈ t, J i j * σ i * σ j

lemma sum_split (C : Finset ι) (g : ι → ι → ℝ) :
    (∑ i, ∑ j, g i j) = (∑ i ∈ C, ∑ j ∈ C, g i j) + (∑ i ∈ C, ∑ j ∈ Cᶜ, g i j)
      + (∑ i ∈ Cᶜ, ∑ j ∈ C, g i j) + (∑ i ∈ Cᶜ, ∑ j ∈ Cᶜ, g i j) := by
  have h1 : ∀ i, (∑ j, g i j) = (∑ j ∈ C, g i j) + ∑ j ∈ Cᶜ, g i j :=
    fun i => (Finset.sum_add_sum_compl C (g i)).symm
  simp only [h1]
  rw [← Finset.sum_add_sum_compl C]
  simp only [Finset.sum_add_distrib]
  ring

theorem energy_flip (J : ι → ι → ℝ) (hsym : ∀ i j, J i j = J j i) (h : ι → ℝ) (σ : ι → ℝ)
    (C : Finset ι) :
    isingEnergy J h (flipOn C σ) =
      isingEnergy J h σ - 2 * block J σ C Cᶜ - 2 * ∑ i ∈ C, h i * σ i := by
  have hin : ∀ i ∈ C, flipOn C σ i = -σ i := fun i hi => by simp [flipOn, hi]
  have hout : ∀ i ∈ Cᶜ, flipOn C σ i = σ i := fun i hi => by
    simp [flipOn, Finset.mem_compl.mp hi]
  -- the two mixed blocks are equal by symmetry of `J`
  have hswap : block J σ Cᶜ C = block J σ C Cᶜ := by
    unfold block
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [hsym]; ring
  unfold isingEnergy
  rw [sum_split C (fun i j => J i j * flipOn C σ i * flipOn C σ j),
    sum_split C (fun i j => J i j * σ i * σ j)]
  have eCC : (∑ i ∈ C, ∑ j ∈ C, J i j * flipOn C σ i * flipOn C σ j)
      = ∑ i ∈ C, ∑ j ∈ C, J i j * σ i * σ j :=
    Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by
      rw [hin i hi, hin j hj]; ring
  have eCO : (∑ i ∈ C, ∑ j ∈ Cᶜ, J i j * flipOn C σ i * flipOn C σ j) = -block J σ C Cᶜ := by
    unfold block; rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun i hi => ?_
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun j hj => by rw [hin i hi, hout j hj]; ring
  have eOC : (∑ i ∈ Cᶜ, ∑ j ∈ C, J i j * flipOn C σ i * flipOn C σ j) = -block J σ Cᶜ C := by
    unfold block; rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun i hi => ?_
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun j hj => by rw [hout i hi, hin j hj]; ring
  have eOO : (∑ i ∈ Cᶜ, ∑ j ∈ Cᶜ, J i j * flipOn C σ i * flipOn C σ j)
      = ∑ i ∈ Cᶜ, ∑ j ∈ Cᶜ, J i j * σ i * σ j :=
    Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by
      rw [hout i hi, hout j hj]
  have eh : (∑ i, h i * flipOn C σ i) = (∑ i, h i * σ i) - 2 * ∑ i ∈ C, h i * σ i := by
    have s1 : (∑ i ∈ C, h i * flipOn C σ i) + (∑ i ∈ Cᶜ, h i * flipOn C σ i)
        = ∑ i, h i * flipOn C σ i := Finset.sum_add_sum_compl _ _
    have s2 : (∑ i ∈ C, h i * σ i) + (∑ i ∈ Cᶜ, h i * σ i) = ∑ i, h i * σ i :=
      Finset.sum_add_sum_compl _ _
    have e1 : (∑ i ∈ C, h i * flipOn C σ i) = -∑ i ∈ C, h i * σ i := by
      rw [← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl fun i hi => by rw [hin i hi]; ring
    have e2 : (∑ i ∈ Cᶜ, h i * flipOn C σ i) = ∑ i ∈ Cᶜ, h i * σ i :=
      Finset.sum_congr rfl fun i hi => by rw [hout i hi]
    linarith
  rw [eCC, eCO, eOC, eOO, eh]
  unfold block at hswap ⊢
  linarith

/-- A dominated ferromagnetic defect edge across the cut makes the flip strictly favourable. -/
theorem flip_decreases (J : ι → ι → ℝ) (hsym : ∀ i j, J i j = J j i) (h : ι → ℝ)
    (σ : ι → ℝ) (hσ : ∀ i, σ i = 1 ∨ σ i = -1) (C : Finset ι) (a b : ι)
    (ha : a ∈ C) (hb : b ∉ C) (hdef : σ a * σ b = -1) (hferro : J a b < 0)
    (hdom : (∑ i ∈ C, ∑ j ∈ Cᶜ, if i = a ∧ j = b then 0 else |J i j|)
      + ∑ i ∈ C, |h i| < -J a b) :
    isingEnergy J h (flipOn C σ) < isingEnergy J h σ := by
  rw [energy_flip J hsym h σ C]
  have habs : ∀ i, |σ i| = 1 := fun i => by rcases hσ i with h | h <;> simp [h]
  -- split off the defect edge from the cross-cut block
  have hb' : b ∈ Cᶜ := Finset.mem_compl.mpr hb
  have hsplit : block J σ C Cᶜ = J a b * σ a * σ b +
      ∑ i ∈ C, ∑ j ∈ Cᶜ, (if i = a ∧ j = b then 0 else J i j * σ i * σ j) := by
    unfold block
    have : ∀ i j, J i j * σ i * σ j =
        (if i = a ∧ j = b then J a b * σ a * σ b else 0)
          + (if i = a ∧ j = b then 0 else J i j * σ i * σ j) := by
      intro i j
      by_cases hij : i = a ∧ j = b
      · obtain ⟨rfl, rfl⟩ := hij; simp
      · simp [hij]
    rw [Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => this i j]
    simp only [Finset.sum_add_distrib]
    congr 1
    rw [Finset.sum_eq_single a]
    · rw [Finset.sum_eq_single b]
      · simp
      · intro j _ hj; simp [hj]
      · intro h'; exact absurd hb' h'
    · intro i _ hi
      exact Finset.sum_eq_zero fun j _ => by simp [hi]
    · intro h'; exact absurd ha h'
  -- bound the remaining cross terms and the fields
  have hrest : -(∑ i ∈ C, ∑ j ∈ Cᶜ, if i = a ∧ j = b then 0 else |J i j|) ≤
      ∑ i ∈ C, ∑ j ∈ Cᶜ, (if i = a ∧ j = b then 0 else J i j * σ i * σ j) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun j _ => ?_
    split_ifs
    · simp
    · have h1 : |J i j * σ i * σ j| = |J i j| := by rw [abs_mul, abs_mul, habs, habs]; ring
      have := neg_abs_le (J i j * σ i * σ j)
      linarith
  have hfield : -(∑ i ∈ C, |h i|) ≤ ∑ i ∈ C, h i * σ i := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun i _ => ?_
    have h1 : |h i * σ i| = |h i| := by rw [abs_mul, habs]; ring
    have := neg_abs_le (h i * σ i)
    linarith
  have hedge : J a b * σ a * σ b = -J a b := by rw [mul_assoc, hdef]; ring
  rw [hsplit, hedge]
  linarith

end Flip

end CoulombHardness
