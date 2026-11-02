import Mathlib
import CoulombHardness.Barahona
import CoulombHardness.GapTransfer

/-!
# The pinned Coulomb-crystal problem and the main theorem

`CoulombSubset P k τ`: some `k` of the candidate sites `P` have Coulomb energy `≤ τ`.

## What is proved here, and what is assumed

`GeometricReduction G K` bundles exactly what the geometric half of the write-up provides for a
planar cubic graph `G` and target `K`:

* a finite point set `P ⊂ ℝ²`, a cardinality `k` and a rational threshold `τ`;
* a `Simulation` of `2 H_B` by the Coulomb energy of `k`-subsets of `P`. Its normalisation map is
  the repair procedure (write-up Lemmas 5.2 and 9.2), and its simulation bound is the Barahona
  normal form (Prop. 11.1, from Lemmas 6.1, 6.2, 8.1, 9.1, 10.1 and 10.2);
* the threshold window (proof of Thm 1.2).

`GeometricReduction.correct` proves, with no further assumptions, that any such object is a
correct reduction from independent set. Constructing a `GeometricReduction` for every `(G, K)`,
in polynomial time, is the geometric content that remains to be formalised; polynomial-time
computability is argued in the write-up, since Mathlib has no reduction framework.

The energy here is `2 H_B` (`twiceHB`), so the source gap is 8. That is more than the 4 that
`Simulation.reduction_correct` needs.
-/

namespace CoulombHardness

/-- Coulomb (Riesz `s = 1`) energy; each unordered pair counted once. -/
noncomputable def coulombEnergy {d : ℕ} (S : Finset (EuclideanSpace ℝ (Fin d))) : ℝ :=
  (∑ p ∈ S, ∑ q ∈ S.erase p, 1 / dist p q) / 2

def CoulombSubset {d : ℕ} (P : Finset (EuclideanSpace ℝ (Fin d))) (k : ℕ) (τ : ℝ) : Prop :=
  ∃ S ⊆ P, S.card = k ∧ coulombEnergy S ≤ τ

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V) [DecidableRel G.Adj]

/-- The `k`-subsets of `P`: configurations of the pinned crystal. -/
abbrev PinnedConfig (P : Finset (EuclideanSpace ℝ (Fin 2))) (k : ℕ) :=
  {S : Finset (EuclideanSpace ℝ (Fin 2)) // S ⊆ P ∧ S.card = k}

/-- Spin configurations of the source Ising model. -/
abbrev SpinConfig (V : Type*) := {σ : V → ℤ // IsSpin σ}

/-- Source threshold for `2 H_B`: `n − 8K`. -/
def sourceThreshold (K : ℕ) : ℤ := (Fintype.card V : ℤ) - 8 * K

/-- Everything the geometric construction must deliver for the instance `(G, K)`. -/
structure GeometricReduction (K : ℕ) where
  P : Finset (EuclideanSpace ℝ (Fin 2))
  k : ℕ
  τ : ℚ
  sim : Simulation (PinnedConfig P k) (SpinConfig V)
  sim_energy : ∀ c, sim.energy c = coulombEnergy c.val
  sim_H : ∀ σ, sim.H σ = twiceHB G σ.val
  window_lo : sim.E0 + sim.lam * (((sourceThreshold (V := V) K : ℤ) : ℝ) + 1) ≤ (τ : ℝ)
  window_hi : (τ : ℝ) ≤ sim.E0 + sim.lam * (((sourceThreshold (V := V) K : ℤ) : ℝ) + 3)

/-- **Main theorem (conditional on the geometry).** Any geometric reduction for `(G, K)` is
correct: the pinned-crystal instance is a yes-instance iff `G` has an independent set of size
`≥ K`. -/
theorem GeometricReduction.correct (hG : G.IsRegularOfDegree 3) {K : ℕ}
    (R : GeometricReduction G K) :
    CoulombSubset R.P R.k (R.τ : ℝ) ↔
      ∃ s : Finset V, G.IsIndepSet (s : Set V) ∧ K ≤ s.card := by
  have hgap : (∃ σ, R.sim.H σ ≤ sourceThreshold (V := V) K) ∨
      (∀ σ, sourceThreshold (V := V) K + 4 ≤ R.sim.H σ) := by
    by_cases h : ∃ s : Finset V, G.IsIndepSet (s : Set V) ∧ K ≤ s.card
    · left
      obtain ⟨σ, hσs, hσ⟩ := (barahona_yes G hG K).mpr h
      exact ⟨⟨σ, hσs⟩, by rw [R.sim_H]; exact hσ⟩
    · right
      intro σ
      have := barahona_gap G hG K h σ.val σ.prop
      rw [R.sim_H]
      unfold sourceThreshold
      omega
  have key := R.sim.reduction_correct (sourceThreshold (V := V) K) R.τ R.window_lo
    R.window_hi hgap
  constructor
  · rintro ⟨S, hSP, hcard, hE⟩
    obtain ⟨σ, hσ⟩ := key.mp ⟨⟨S, hSP, hcard⟩, by rw [R.sim_energy]; exact hE⟩
    rw [R.sim_H] at hσ
    exact (barahona_yes G hG K).mp ⟨σ.val, σ.prop, hσ⟩
  · intro h
    obtain ⟨σ, hσs, hσ⟩ := (barahona_yes G hG K).mpr h
    obtain ⟨c, hc⟩ := key.mpr ⟨⟨σ, hσs⟩, by rw [R.sim_H]; exact hσ⟩
    exact ⟨c.val, c.prop.1, c.prop.2, by rw [← R.sim_energy]; exact hc⟩

end CoulombHardness
