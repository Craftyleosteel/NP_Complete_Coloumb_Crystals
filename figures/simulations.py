"""Simulation figures for the PRA manuscript.

fig_free_landscape : N ions in a 2D isotropic harmonic trap; distinct local minima reached from
                     random starts, and the fraction of quenches that find the lowest one.
fig_collapse       : the pinned-crystal instance built from the cube graph Q3 (Fig. 1). Exact
                     Coulomb energy of all 2^8 tree-uniform states vs Barahona's H_B.
fig_anneal         : simulated annealing (single-selector flips) on the same instance.

The pinned instance uses demonstration parameters (b = 0.02, D = 8, grid scale S = 12), much
milder than the proof's b = (1e8 M^3)^-1, so the asymptotic separations are visible at a size
that can be enumerated. All energies are exact sums of 1/r over the selected sites.
"""
import json
import os
import pathlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve().parent
UP, DOWN, AF = "#2f6db5", "#b8661a", "#8a3fb0"
INK, MUTED, GRID = "#1b1f24", "#6b7280", "#d4d8de"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 400,
})
RESULTS = {}


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# =========================================================== free crystal census
@njit(cache=True)
def _trap_eg(x, n):
    E = 0.0
    g = np.zeros(2 * n)
    for i in range(n):
        xi, yi = x[2 * i], x[2 * i + 1]
        E += 0.5 * (xi * xi + yi * yi)
        g[2 * i] += xi
        g[2 * i + 1] += yi
        for j in range(i + 1, n):
            dx, dy = xi - x[2 * j], yi - x[2 * j + 1]
            r2 = dx * dx + dy * dy
            ir = 1.0 / np.sqrt(r2)
            E += ir
            c = ir * ir * ir
            g[2 * i] -= c * dx
            g[2 * i + 1] -= c * dy
            g[2 * j] += c * dx
            g[2 * j + 1] += c * dy
    return E, g


def trap_energy_grad(x, n):
    return _trap_eg(x, n)


def census(n, starts, rng):
    energies = []
    for _ in range(starts):
        x0 = rng.normal(scale=np.sqrt(n) * 0.6, size=2 * n)
        res = minimize(trap_energy_grad, x0, args=(n,), jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 20000})
        energies.append(res.fun)
    e = np.sort(np.array(energies))
    distinct = [e[0]]
    for v in e[1:]:
        if v - distinct[-1] > 1e-7 * abs(v):
            distinct.append(v)
    p_ground = np.mean(np.abs(e - e[0]) <= 1e-7 * abs(e[0]))
    return len(distinct), p_ground


def free_landscape():
    rng = np.random.default_rng(1)
    Ns = list(range(10, 101, 10))
    starts = 300
    out = [census(n, starts, rng) for n in Ns]
    nmin = [o[0] for o in out]
    pg = [o[1] for o in out]
    RESULTS["free"] = {"N": Ns, "distinct_minima": nmin, "p_ground": pg, "starts": starts}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.375, 1.6))
    a1.semilogy(Ns, nmin, "o-", color=UP, ms=3, lw=1.2)
    a1.set_yticks([1, 3, 10, 30, 100])
    a1.set_yticklabels(["1", "3", "10", "30", "100"])
    a1.minorticks_off()
    a1.set_xlabel("ions $N$")
    a1.set_ylabel("distinct minima found")
    a1.set_title("(a)", loc="left")
    a2.semilogy(Ns, pg, "o-", color=DOWN, ms=3, lw=1.2)
    a2.set_xlabel("ions $N$")
    a2.set_ylabel("quenches reaching ground")
    a2.set_title("(b)", loc="left")
    for a in (a1, a2):
        clean(a)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_free_landscape")


# =========================================================== pinned Q3 instance
V = {"o0": (4, 0), "o1": (8, 4), "o2": (4, 8), "o3": (0, 4),
     "i0": (4, 2), "i1": (6, 4), "i2": (4, 6), "i3": (2, 4)}
VERTS = list(V)
ROUTES = [
    ("o0", "o1", [(4, 0), (8, 0), (8, 4)]), ("o1", "o2", [(8, 4), (8, 8), (4, 8)]),
    ("o2", "o3", [(4, 8), (0, 8), (0, 4)]), ("o3", "o0", [(0, 4), (0, 0), (4, 0)]),
    ("i0", "i1", [(4, 2), (6, 2), (6, 4)]), ("i1", "i2", [(6, 4), (6, 6), (4, 6)]),
    ("i2", "i3", [(4, 6), (2, 6), (2, 4)]), ("i3", "i0", [(2, 4), (2, 2), (4, 2)]),
    ("o0", "i0", [(4, 0), (4, 2)]), ("o1", "i1", [(8, 4), (6, 4)]),
    ("o2", "i2", [(4, 8), (4, 6)]), ("o3", "i3", [(0, 4), (2, 4)]),
]
EDGES = [(VERTS.index(u), VERTS.index(v)) for u, v, _ in ROUTES]


def lattice_path(poly):
    """all integer points along an axis-parallel polyline, in order"""
    pts = [np.array(poly[0])]
    for a, c in zip(poly[:-1], poly[1:]):
        a, c = np.array(a), np.array(c)
        d = np.sign(c - a)
        for k in range(1, int(np.abs(c - a).sum()) + 1):
            pts.append(a + k * d)
    return pts


def build_instance(S=10, D=4):
    cells, tree, role = [], [], []          # role: 0 wire, 1 root, 2 p-prefix, 3 q-lane/step
    for k, (v, p) in enumerate(V.items()):
        cells.append(np.array(p) * S); tree.append(k); role.append(1)
    boxes = []
    for (u, v, poly) in ROUTES:
        P = [np.array(q) * S for q in poly]
        path = lattice_path([tuple(q) for q in P])
        # longest straight segment hosts the objective box
        segs = [(np.abs(P[i + 1] - P[i]).sum(), i) for i in range(len(P) - 1)]
        L_k, k = max(segs)
        t0 = int(sum(np.abs(P[i + 1] - P[i]).sum() for i in range(k)))
        tm = t0 + (L_k - (D + 2)) // 2
        m = path[tm]
        d = np.sign(P[k + 1] - P[k])
        nrm = np.array([-d[1], d[0]])
        iu, iv = VERTS.index(u), VERTS.index(v)
        p_idx, q_idx = [], []
        for t in range(1, tm + 1):                      # tree u along the route up to p0 = m
            cells.append(path[t]); tree.append(iu); role.append(2 if t > tm - D - 1 else 0)
            if t > tm - D - 1:
                p_idx.append(len(cells) - 1)
        for t in range(tm + D + 2, len(path) - 1):      # tree v continuation on the route
            cells.append(path[t]); tree.append(iv); role.append(0)
        cells.append(m + (D + 1) * d); tree.append(iv); role.append(3)   # step cell
        for j in range(D + 1):                          # lane q_j = m + (1+j) d + n
            cells.append(m + (1 + j) * d + nrm); tree.append(iv); role.append(3)
            q_idx.append(len(cells) - 1)
        boxes.append((p_idx, q_idx))
    return np.array(cells, float), np.array(tree), np.array(role), boxes


def selector_points(c, b, eta):
    """(M, 2 states, 2 points, 2 coords); state 0 = +1 diagonal, state 1 = -1 diagonal"""
    M = len(c)
    pts = np.empty((M, 2, 2, 2))
    s = (b + eta)[:, None]
    pts[:, 0, 0] = c + s * np.array([1, 1]); pts[:, 0, 1] = c - s * np.array([1, 1])
    pts[:, 1, 0] = c + b * np.array([1, -1]); pts[:, 1, 1] = c - b * np.array([1, -1])
    return pts


def couplings(c, b, eta):
    pts = selector_points(c, b, eta)
    F = np.zeros((2, 2, len(c), len(c)))
    for si in range(2):
        for sj in range(2):
            acc = 0.0
            for a in range(2):
                for e in range(2):
                    d = pts[:, si, a][:, None, :] - pts[:, sj, e][None, :, :]
                    r = np.sqrt((d ** 2).sum(-1))
                    np.fill_diagonal(r, np.inf)
                    acc = acc + 1.0 / r
            F[si, sj] = acc
    J = (F[0, 0] - F[0, 1] - F[1, 0] + F[1, 1]) / 4
    h1 = (F[0, 0] + F[0, 1] - F[1, 0] - F[1, 1]) / 4          # field on i from pair (i, j)
    A = (F[0, 0] + F[0, 1] + F[1, 0] + F[1, 1]) / 4
    internal_plus = 1.0 / (2 * np.sqrt(2) * (b + eta))
    internal_minus = 1.0 / (2 * np.sqrt(2) * b)
    H = (internal_plus - internal_minus) / 2 + h1.sum(1)
    const = A.sum() / 2 + ((internal_plus + internal_minus) / 2).sum()
    np.fill_diagonal(J, 0.0)
    return J, H, const


def pinned_instance():
    b, D, S = 0.02, 8, 12
    c, tree, role, boxes = build_instance(S, D)
    M = len(c)
    eta = np.zeros(M)
    J, H, const = couplings(c, b, eta)
    lam = np.mean([J[np.ix_(p, q)].sum() for p, q in boxes])
    g = np.where(role == 1, lam, 0.0)
    beta = -1.0 / (4 * np.sqrt(2) * b ** 2)
    for _ in range(4):                                   # field compensation (Newton on eta)
        eta = eta + (g - H) / beta
        J, H, const = couplings(c, b, eta)
    resid_field = np.abs(H - g).sum()
    # exact energy of every tree-uniform configuration
    n = len(V)
    HB, E = [], []
    for code in range(2 ** n):
        sig = np.array([1 if (code >> k) & 1 else -1 for k in range(n)])
        s = sig[tree].astype(float)
        E.append(const + H @ s + 0.5 * s @ J @ s)
        HB.append(sum(sig[u] * sig[v] for u, v in EDGES) + sig.sum())
    HB, E = np.array(HB), np.array(E)
    slope, icpt = np.polyfit(HB, E, 1)
    resid = E - (icpt + slope * HB)
    b4 = b ** 4
    RESULTS["pinned"] = {
        "b": b, "D": D, "S": S, "M_cells": int(M), "sites": int(4 * M), "ions": int(2 * M),
        "lambda_box_over_b4": lam / b4, "lambda_fit_over_b4": slope / b4,
        "max_residual_over_lambda": float(np.abs(resid).max() / slope),
        "residual_field_over_b4": resid_field / b4, "Jnn_over_b4": float(J[0, :][J[0, :] < 0].min() / b4),
        "HB_min": int(HB.min()), "ground_states": int((HB == HB.min()).sum()),
    }
    fig, ax = plt.subplots(figsize=(3.375, 2.1))
    levels = np.unique(HB)
    for L in levels:
        ax.axhline(L, color=GRID, lw=0.5, zorder=0)
    y = (E - icpt) / slope
    jitter = (np.random.default_rng(0).random(len(HB)) - 0.5) * 0.9
    ax.scatter(HB + jitter, y, s=5, color=UP, lw=0, zorder=2)
    ax.plot([levels[0] - 1, levels[-1] + 1], [levels[0] - 1, levels[-1] + 1], color=MUTED, lw=0.7,
            ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel(r"Barahona energy $H_B(\sigma)$")
    ax.set_ylabel(r"$(E_1(S_\sigma)-E_0)/\lambda$")
    ax.text(0.03, 0.95, f"{2 ** n} tree states, {2 * M} ions on {4 * M} sites\n"
            f"max deviation {RESULTS['pinned']['max_residual_over_lambda']:.1e}$\\,\\lambda$",
            transform=ax.transAxes, va="top", fontsize=6.8, color=INK)
    clean(ax)
    save(fig, "fig_collapse")
    return J, H, const, tree, slope, icpt, HB, E


# =========================================================== annealing
@njit(cache=True)
def anneal(J, H, s, T0, T1, sweeps, seed):
    np.random.seed(seed)
    M = len(s)
    f = J @ s
    E = H @ s + 0.5 * s @ f
    trace = np.empty(sweeps)
    for k in range(sweeps):
        T = T0 * (T1 / T0) ** (k / (sweeps - 1))
        for _ in range(M):
            i = np.random.randint(M)
            dE = -2.0 * s[i] * (H[i] + f[i])
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                s[i] = -s[i]
                E += dE
                for j in range(M):
                    f[j] += 2.0 * s[i] * J[j, i]
        trace[k] = E
    return trace, s


def annealing(J, H, const, tree, lam, icpt, HB, E):
    b4 = 0.02 ** 4
    E_ground = E.min()
    runs, sweeps = 24, 3000
    traces, finals, mono, walls = [], [], [], []
    # nearest-neighbour consistency edges: same tree, unit distance
    nn_pairs = [(i, j) for i in range(len(H)) for j in np.nonzero(J[i] < -40 * b4)[0] if i < j
                and tree[i] == tree[j]]
    for r in range(runs):
        rng = np.random.default_rng(100 + r)
        s0 = rng.choice([-1.0, 1.0], size=len(H))
        tr, s = anneal(J, H, s0.copy(), 200 * b4, 0.05 * b4, sweeps, 100 + r)
        tr = tr + const
        traces.append((tr - E_ground) / lam)
        finals.append((tr[-1] - E_ground) / lam)
        mono.append(all(len(set(s[tree == v])) == 1 for v in range(len(V))))
        walls.append(int(sum(s[i] != s[j] for i, j in nn_pairs)))
    finals = np.array(finals)
    RESULTS["anneal"] = {"runs": runs, "sweeps": sweeps, "T0_over_b4": 200, "T1_over_b4": 0.05,
                         "reached_ground": int((finals < 0.5).sum()),
                         "final_monochromatic": int(sum(mono)),
                         "final_excess_over_lambda": [float(x) for x in finals],
                         "domain_walls": walls,
                         "wall_cost_over_lambda": float(2 * 48.05 * b4 / lam)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.375, 1.7), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(1, sweeps + 1)
    for k, tr in enumerate(traces[:10]):
        a1.semilogy(x, np.maximum(tr, 1e-2), color=UP if finals[k] < 0.5 else DOWN, lw=0.6, alpha=0.9)
    a1.set_xlabel("sweep")
    a1.set_ylabel(r"$(E-E_{\rm gs})/\lambda$")
    a1.set_title("(a)", loc="left")
    a1.set_ylim(5, 5e3)
    w = np.array(walls)
    a2.hist(w, bins=np.arange(w.min() - 0.5, w.max() + 1.5, 1), color=MUTED, edgecolor="white", lw=0.6)
    a2.set_xlabel("domain walls left")
    a2.set_ylabel("runs")
    a2.set_title("(b)", loc="left")
    for a in (a1, a2):
        clean(a)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "fig_anneal")


def barrier_path(J, H, const, tree, lam, E):
    """Energy along a path of single-selector flips between the two ground states of Q3.

    Each tree is flipped cell by cell in breadth-first order from one of its leaves, so a domain
    wall enters at the leaf, sweeps to the root, splits, and leaves through the other two branches.
    """
    b = RESULTS["pinned"]["b"]
    b4 = b ** 4
    n = len(V)
    nn = [(i, j) for i in range(len(H)) for j in np.nonzero(J[i] < -40 * b4)[0] if i < j and tree[i] == tree[j]]
    adj = {i: [] for i in range(len(H))}
    for i, j in nn:
        adj[i].append(j); adj[j].append(i)
    sigA = np.array([SPIN_GS[v] for v in VERTS])
    s = sigA[tree].astype(float)
    f = J @ s
    E_now = const + H @ s + 0.5 * s @ f
    E_gs = E.min()
    trace, marks = [(E_now - E_gs) / lam], []
    for v in range(n):
        cells = [i for i in range(len(H)) if tree[i] == v]
        leaf = next(i for i in cells if len(adj[i]) == 1)
        order, seen, queue = [], {leaf}, [leaf]
        while queue:
            i = queue.pop(0); order.append(i)
            for j in adj[i]:
                if j not in seen:
                    seen.add(j); queue.append(j)
        for i in order:
            dE = -2.0 * s[i] * (H[i] + f[i])
            s[i] = -s[i]
            f += 2.0 * s[i] * J[:, i]
            E_now += dE
            trace.append((E_now - E_gs) / lam)
        marks.append(len(trace) - 1)
    trace = np.array(trace)
    intra = (0.5 - 1 / (2 * np.sqrt(2))) / b / lam
    RESULTS["barrier"] = {"max_over_lambda": float(trace.max()), "end_over_lambda": float(trace[-1]),
                          "intra_cell_over_lambda": float(intra), "flips": int(len(trace) - 1)}
    fig, ax = plt.subplots(figsize=(3.375, 1.9))
    for L in range(0, int(trace.max()) + 8, 4):
        ax.axhline(L, color=GRID, lw=0.4, zorder=0)
    ax.plot(np.arange(len(trace)), trace, color=UP, lw=0.9)
    for m in marks[:-1]:
        ax.axvline(m, color=MUTED, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel("single-selector flips along the path")
    ax.set_ylabel(r"$(E-E_{\rm gs})/\lambda$")
    ex = int(np.floor(np.log10(intra)))
    ax.text(0.98, 0.95, f"each flip also crosses an illegal selector\nstate $\\approx{intra / 10 ** ex:.1f}\\times10^{{{ex}}}\\,\\lambda$ higher (off scale)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6, color=INK)
    ax.set_xlim(0, len(trace) - 1)
    ax.set_ylim(-2, trace.max() * 1.35)
    clean(ax)
    save(fig, "fig_barrier")


SPIN_GS = {v: (+1 if v in ("o0", "o2", "i1", "i3") else -1) for v in V}


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["pinned", "free"]
    path = HERE / "simulation_results.json"
    if path.exists():
        RESULTS.update(json.loads(path.read_text()))
    if "pinned" in parts:
        J, H, const, tree, lam, icpt, HB, E = pinned_instance()
        print("pinned", json.dumps(RESULTS["pinned"], indent=1))
        annealing(J, H, const, tree, lam, icpt, HB, E)
        print("anneal", {k: v for k, v in RESULTS["anneal"].items() if k != "final_excess_over_lambda"})
        barrier_path(J, H, const, tree, lam, E)
        print("barrier", RESULTS["barrier"])
    if "free" in parts:
        free_landscape()
        print("free", RESULTS["free"])
    path.write_text(json.dumps(RESULTS, indent=1, default=float))
