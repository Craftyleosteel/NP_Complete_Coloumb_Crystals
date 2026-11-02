import Mathlib

/-!
# Barahona's source Hamiltonian on cubic graphs

`H_B(σ) = Σ_{uv ∈ E} σ_u σ_v + Σ_v σ_v`. To stay in `ℤ` we work with `twiceHB = 2 H_B`, summing
the edge term over *ordered* adjacent pairs. For a cubic graph, with `A` the set of `+1` spins,

  `2 H_B(σ) = n + 4 · pairsIn A − 8 · |A|`                                  (`twiceHB_eq`)

where `pairsIn A` counts ordered adjacent pairs inside `A`. Every `A` contains an independent set
`I` with `2|A| ≤ 2|I| + pairsIn A` (`exists_indepSet_large`, greedy deletion). Hence

* `G` has an independent set of size `≥ K`  ↔  `∃ σ, 2 H_B(σ) ≤ n − 8K`    (`barahona_yes`)
* otherwise every `σ` has `2 H_B(σ) ≥ n − 8K + 8`                         (`barahona_gap`)

which is the integer gap the geometric construction must preserve.
-/

namespace CoulombHardness

open Finset

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V) [DecidableRel G.Adj]

def IsSpin (σ : V → ℤ) : Prop := ∀ v, σ v = 1 ∨ σ v = -1

/-- Twice Barahona's Hamiltonian (edge term over ordered adjacent pairs). -/
def twiceHB (σ : V → ℤ) : ℤ :=
  (∑ u, ∑ v, if G.Adj u v then σ u * σ v else 0) + 2 * ∑ v, σ v

/-- The `+1` spins. -/
def upSet (σ : V → ℤ) : Finset V := univ.filter fun v => σ v = 1

/-- Ordered adjacent pairs with both endpoints in `A` (twice the number of edges inside `A`). -/
def pairsIn (A : Finset V) : ℕ :=
  (univ.filter fun p : V × V => G.Adj p.1 p.2 ∧ p.1 ∈ A ∧ p.2 ∈ A).card

/-! ### The counting identity -/

omit [DecidableEq V] in
lemma sum_adj_one (u : V) :
    (∑ v, if G.Adj u v then (1 : ℤ) else 0) = G.degree u := by
  rw [← SimpleGraph.card_neighborFinset_eq_degree, SimpleGraph.neighborFinset_eq_filter,
    Finset.card_filter]
  push_cast
  rfl

omit [DecidableEq V] in
/-- `Σ_u Σ_{v ~ u} f u = 3 Σ_u f u` on a cubic graph. -/
lemma sum_adj_left (hG : G.IsRegularOfDegree 3) (f : V → ℤ) :
    (∑ u, ∑ v, if G.Adj u v then f u else 0) = 3 * ∑ u, f u := by
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  have h : ∀ v, (if G.Adj u v then f u else 0) = f u * (if G.Adj u v then 1 else 0) := by
    intro v; split_ifs <;> simp
  simp only [h, ← Finset.mul_sum, sum_adj_one, hG u]
  push_cast
  ring

omit [DecidableEq V] in
/-- Same with the summand depending on the other endpoint (uses symmetry of adjacency). -/
lemma sum_adj_right (hG : G.IsRegularOfDegree 3) (f : V → ℤ) :
    (∑ u, ∑ v, if G.Adj u v then f v else 0) = 3 * ∑ u, f u := by
  rw [Finset.sum_comm]
  simp only [G.adj_comm]
  exact sum_adj_left G hG f

theorem twiceHB_eq (hG : G.IsRegularOfDegree 3) (σ : V → ℤ) (hσ : IsSpin σ) :
    twiceHB G σ =
      Fintype.card V + 4 * (pairsIn G (upSet σ) : ℤ) - 8 * ((upSet σ).card : ℤ) := by
  -- indicator of the `+1` spins
  set x : V → ℤ := fun v => if σ v = 1 then 1 else 0 with hx
  have hσx : ∀ v, σ v = 2 * x v - 1 := by
    intro v; rcases hσ v with h | h <;> simp [hx, h]
  have hpt : ∀ u v, (if G.Adj u v then σ u * σ v else 0) =
      4 * (if G.Adj u v then x u * x v else 0) - 2 * (if G.Adj u v then x u else 0)
        - 2 * (if G.Adj u v then x v else 0) + (if G.Adj u v then 1 else 0) := by
    intro u v
    by_cases h : G.Adj u v
    · simp only [ite_eq_left h, hσx]; ring
    · simp only [ite_eq_right h]; ring
  have hpairs : (pairsIn G (upSet σ) : ℤ) = ∑ u, ∑ v, if G.Adj u v then x u * x v else 0 := by
    rw [pairsIn, Finset.card_filter, ← Finset.univ_product_univ, Finset.sum_product]
    push_cast
    refine Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun v _ => ?_
    by_cases hu : σ u = 1 <;> by_cases hv : σ v = 1 <;> by_cases ha : G.Adj u v <;>
      simp [upSet, hx, hu, hv, ha]
  have hcard : ((upSet σ).card : ℤ) = ∑ v, x v := by
    rw [upSet, Finset.card_filter]; push_cast; rfl
  have hn : (∑ u, ∑ v, if G.Adj u v then (1 : ℤ) else 0) = 3 * Fintype.card V := by
    have := sum_adj_left G hG (fun _ => (1 : ℤ))
    simpa using this
  have hlin : (∑ v, σ v) = 2 * (∑ v, x v) - Fintype.card V := by
    simp only [hσx, Finset.sum_sub_distrib, ← Finset.mul_sum]
    simp
  unfold twiceHB
  simp only [hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [sum_adj_left G hG x, sum_adj_right G hG x, hn, hlin, hpairs, hcard]
  ring

/-! ### Greedy independent sets -/

/-- Every vertex set `A` contains an independent set `I` with `2|A| ≤ 2|I| + pairsIn A`:
delete one endpoint of an internal edge; this loses one vertex and at least two ordered pairs. -/
theorem exists_indepSet_large (A : Finset V) :
    ∃ I ⊆ A, G.IsIndepSet (I : Set V) ∧ 2 * A.card ≤ 2 * I.card + pairsIn G A := by
  induction A using Finset.strongInduction with
  | H A ih =>
    by_cases hind : G.IsIndepSet (A : Set V)
    · exact ⟨A, subset_rfl, hind, by omega⟩
    obtain ⟨u, hu, v, hv, hne, huv⟩ : ∃ u ∈ A, ∃ v ∈ A, u ≠ v ∧ G.Adj u v := by
      by_contra hcon
      push Not at hcon
      exact hind fun x hx y hy hxy => hcon x hx y hy hxy
    obtain ⟨I, hIA, hI, hbound⟩ := ih (A.erase u) (Finset.erase_ssubset hu)
    refine ⟨I, hIA.trans (Finset.erase_subset u A), hI, ?_⟩
    -- removing `u` destroys the two ordered pairs `(u,v)` and `(v,u)`
    have hpairs : pairsIn G (A.erase u) + 2 ≤ pairsIn G A := by
      set S' := univ.filter fun p : V × V => G.Adj p.1 p.2 ∧ p.1 ∈ A.erase u ∧ p.2 ∈ A.erase u
      set S := univ.filter fun p : V × V => G.Adj p.1 p.2 ∧ p.1 ∈ A ∧ p.2 ∈ A
      have h2 : ({(u, v), (v, u)} : Finset (V × V)).card = 2 :=
        Finset.card_pair fun h => hne (congrArg Prod.fst h)
      have hdisj : Disjoint S' {(u, v), (v, u)} := by
        rw [Finset.disjoint_left]
        intro p hp hp2
        simp only [S', Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_erase] at hp
        simp only [Finset.mem_insert, Finset.mem_singleton] at hp2
        rcases hp2 with rfl | rfl
        · exact hp.2.1.1 rfl
        · exact hp.2.2.1 rfl
      have hsub : S' ∪ {(u, v), (v, u)} ⊆ S := by
        intro p hp
        simp only [Finset.mem_union, Finset.mem_insert, Finset.mem_singleton] at hp
        simp only [S, Finset.mem_filter, Finset.mem_univ, true_and]
        rcases hp with hp | rfl | rfl
        · simp only [S', Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_erase] at hp
          exact ⟨hp.1, hp.2.1.2, hp.2.2.2⟩
        · exact ⟨huv, hu, hv⟩
        · exact ⟨huv.symm, hv, hu⟩
      have := Finset.card_le_card hsub
      rw [Finset.card_union_of_disjoint hdisj, h2] at this
      exact this
    have hcard := Finset.card_erase_of_mem hu
    have hpos : 0 < A.card := Finset.card_pos.mpr ⟨u, hu⟩
    omega

/-! ### Threshold form and gap -/

/-- `pairsIn` of an independent set vanishes. -/
lemma pairsIn_eq_zero_of_indep {s : Finset V} (hs : G.IsIndepSet (s : Set V)) :
    pairsIn G s = 0 := by
  rw [pairsIn, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  rintro ⟨a, b⟩ - ⟨hab, ha, hb⟩
  exact hs ha hb (G.ne_of_adj hab) hab

theorem barahona_yes (hG : G.IsRegularOfDegree 3) (K : ℕ) :
    (∃ σ, IsSpin σ ∧ twiceHB G σ ≤ Fintype.card V - 8 * K) ↔
      ∃ s : Finset V, G.IsIndepSet (s : Set V) ∧ K ≤ s.card := by
  constructor
  · rintro ⟨σ, hσ, hle⟩
    obtain ⟨I, -, hI, hb⟩ := exists_indepSet_large G (upSet σ)
    rw [twiceHB_eq G hG σ hσ] at hle
    exact ⟨I, hI, by omega⟩
  · rintro ⟨s, hs, hK⟩
    let σ : V → ℤ := fun v => if v ∈ s then 1 else -1
    have hσ : IsSpin σ := fun v => by by_cases h : v ∈ s <;> simp [σ, h]
    have hup : upSet σ = s := by
      ext v; by_cases h : v ∈ s <;> simp [upSet, σ, h]
    refine ⟨σ, hσ, ?_⟩
    rw [twiceHB_eq G hG σ hσ, hup, pairsIn_eq_zero_of_indep G hs]
    push_cast
    omega

theorem barahona_gap (hG : G.IsRegularOfDegree 3) (K : ℕ)
    (hno : ¬ ∃ s : Finset V, G.IsIndepSet (s : Set V) ∧ K ≤ s.card) :
    ∀ σ, IsSpin σ → (Fintype.card V : ℤ) - 8 * K + 8 ≤ twiceHB G σ := by
  intro σ hσ
  obtain ⟨I, -, hI, hb⟩ := exists_indepSet_large G (upSet σ)
  have hIK : I.card < K := by
    by_contra h
    exact hno ⟨I, hI, by omega⟩
  rw [twiceHB_eq G hG σ hσ]
  omega

end CoulombHardness
