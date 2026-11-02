# When is finding the ground state of an ion Coulomb crystal NP-hard?

Code, data, and formal proofs accompanying the paper by **Paco Navarro** (Department of Physics,
UC Santa Barbara), *"When is finding the ground state of an ion Coulomb crystal NP-hard?"*
(arXiv link to be added).

**Short version:** for identical ions in a fixed trap ("free" crystals), the ground-state problem
cannot be NP-hard unless the polynomial hierarchy collapses, even though it is hard in practice. When
the ions are pinned to a prescribed set of sites, choosing which sites to fill so as to minimize the
Coulomb energy is NP-hard in two and three dimensions, and NP-complete at finite precision.

## New to this? Start with the explainer

[`explainer/explainer.pdf`](explainer/explainer.pdf) is a short, accessible version written for
physicists with no background in computational complexity. It covers why the problem is interesting,
what "NP-hard" means in physics terms, the quadrupole trick behind the construction, and the
simulations. The same piece is in [`explainer/explainer.html`](explainer/explainer.html); download it
together with the `explainer_img/` folder and open it in a browser.

## Contents

| Folder | What it contains | Reproduces |
|---|---|---|
| `explainer/` | The accessible explainer (PDF and HTML) | – |
| `checks/` | Symbolic and numerical checks of the proof's constants | Lemma 4 angular law, field coefficients, λ, sensitivity constants |
| `checks/certify_budget.py` | Interval-arithmetic certification of the lattice sums | Table II and the cut budget (adverse load ≤ 42.225 b⁴ < 47.999 b⁴) |
| `figures/` | Scripts, saved results (`*.json`), and rendered figures | Every figure and quoted simulation number in the paper |
| `lean/` | Lean 4 + Mathlib formalization | The machine-checked components listed in Appendix A |

## Reproducing the results

Python 3.12 or later (tested on 3.14):

```bash
pip install -r requirements.txt

python checks/multipole_law.py        # angular law at s = 1 and s = 2 (symbolic)
python checks/s1_constants.py         # field coefficient, lambda bounds, sensitivity constants
python checks/certify_budget.py       # interval-arithmetic certificate, under a minute

python figures/make_figures.py        # schematic and analytic figures, seconds
python figures/simulations.py pinned  # pinned cube-graph instance: collapse, annealing, barrier path
python figures/free_crystals.py       # free-crystal census and schematic; uses all cores, ~1.5 min on 24
```

The `*.json` files in `figures/` hold the results quoted in the paper, so the figures can be
replotted without rerunning the simulations.

### Lean

Install [elan](https://github.com/leanprover/elan), then:

```bash
cd lean
lake exe cache get   # prebuilt Mathlib (several GB)
lake build
```

The toolchain is pinned in `lean/lean-toolchain` (Lean v4.36.0-rc2). `#print axioms` on the main
theorems reports only `propext`, `Classical.choice`, and `Quot.sound`.

## What is and is not machine-checked

The Lean development checks the source-problem combinatorics, the gap-transfer logic, the selector
inequalities, the spin-level flip criterion, and the assembly of the main theorem. That assembly is
**conditional** on a `GeometricReduction` object, which Lean does not construct. The geometric estimates
(multipole bounds, lattice sums, field compensation, layout) are proved by hand in the paper; the
tightest of them, the lattice sums, are certified independently by `checks/certify_budget.py`.

## Use of AI tools

This work was prepared with extensive assistance from a large language model (Claude, Anthropic, via
Claude Code), including the code, the Lean formalization, derivations, and drafting. The author directed
the project, checked the results, and takes full responsibility for them. See the acknowledgments of the
paper for details.

## License

MIT. See [`LICENSE`](LICENSE).
