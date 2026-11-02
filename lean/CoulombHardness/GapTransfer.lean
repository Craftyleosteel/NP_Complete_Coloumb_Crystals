import Mathlib

/-!
# Gap transfer: the logical skeleton of the reduction

Emmerich (arXiv:2608.23506, Sections 10–11) proves correctness of the geometric reduction in two
steps:

* **normalisation**: every feasible subset can be improved (energy never increases) to a normal
  form that decodes to a spin configuration;
* **simulation**: on normal forms the energy is `E0 + λ H(σ) + R` with `|R| < λ/20`.

Together with an integer gap of 4 in the source Hamiltonian, any threshold in the window
`[E0 + λ(H0+1), E0 + λ(H0+3)]` decides the source instance. The window (rather than a single
threshold) is what lets us output a *rational* threshold even though `E0` and `λ` are irrational
sums of reciprocal square roots at `s = 1`.
-/

namespace CoulombHardness

variable {Cfg Spin : Type*}

/-- Abstract simulation of an integer Hamiltonian `H` by an energy on configurations. -/
structure Simulation (Cfg Spin : Type*) where
  energy : Cfg → ℝ
  H : Spin → ℤ
  normalize : Cfg → Cfg
  decode : Cfg → Spin
  encode : Spin → Cfg
  E0 : ℝ
  lam : ℝ
  lam_pos : 0 < lam
  normalize_le : ∀ c, energy (normalize c) ≤ energy c
  sim_normal : ∀ c, |energy (normalize c) - (E0 + lam * H (decode (normalize c)))| < lam / 20
  sim_encode : ∀ σ, |energy (encode σ) - (E0 + lam * H σ)| < lam / 20

/-- Correctness of the reduction for any threshold in the admissible window. -/
theorem Simulation.reduction_correct (S : Simulation Cfg Spin) (H0 : ℤ) (τ : ℝ)
    (hτlo : S.E0 + S.lam * ((H0 : ℝ) + 1) ≤ τ)
    (hτhi : τ ≤ S.E0 + S.lam * ((H0 : ℝ) + 3))
    (hgap : (∃ σ, S.H σ ≤ H0) ∨ (∀ σ, H0 + 4 ≤ S.H σ)) :
    (∃ c, S.energy c ≤ τ) ↔ (∃ σ, S.H σ ≤ H0) := by
  constructor
  · rintro ⟨c, hc⟩
    rcases hgap with h | h
    · exact h
    · exfalso
      have h1 := S.normalize_le c
      have h4 := (abs_lt.mp (S.sim_normal c)).1
      have h3 : (H0 : ℝ) + 4 ≤ (S.H (S.decode (S.normalize c)) : ℝ) := by
        exact_mod_cast h (S.decode (S.normalize c))
      have hl := S.lam_pos
      nlinarith
  · rintro ⟨σ, hσ⟩
    refine ⟨S.encode σ, ?_⟩
    have h2 := (abs_lt.mp (S.sim_encode σ)).2
    have h3 : (S.H σ : ℝ) ≤ H0 := by exact_mod_cast hσ
    have hl := S.lam_pos
    nlinarith

end CoulombHardness
