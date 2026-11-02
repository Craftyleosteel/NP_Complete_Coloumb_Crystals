"""Free-crystal difficulty and the free-vs-pinned schematic for the PRA manuscript.

fig_free_landscape : N ions in a 2D isotropic harmonic trap, E = sum r^2/2 + sum 1/r_ij.
    (a) distinct minima per batch of quenches, (b) success probability per attempt for a random
    quench and for simulated annealing, (c) time to solution TTS = t_attempt / p_success.
    Error bars: standard error over independent batches.
fig_free_vs_pinned : schematic of the two problem classes (Definition 1 of the paper).

Usage: python free_crystals.py [census] [schematic]
"""
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numba import njit
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve().parent
NS = list(range(10, 101, 10))
BATCHES, QUENCHES_PER_BATCH, SA_PER_BATCH = 10, 400, 16
SA_SWEEPS, SA_T0, SA_T1 = 4000, 0.05, 1e-5


@njit(cache=True)
def trap_eg(x, n):
    E = 0.0
    g = np.zeros(2 * n)
    for i in range(n):
        xi, yi = x[2 * i], x[2 * i + 1]
        E += 0.5 * (xi * xi + yi * yi)
        g[2 * i] += xi
        g[2 * i + 1] += yi
        for j in range(i + 1, n):
            dx, dy = xi - x[2 * j], yi - x[2 * j + 1]
            ir = 1.0 / np.sqrt(dx * dx + dy * dy)
            E += ir
            c = ir * ir * ir
            g[2 * i] -= c * dx
            g[2 * i + 1] -= c * dy
            g[2 * j] += c * dx
            g[2 * j + 1] += c * dy
    return E, g


@njit(cache=True)
def anneal_positions(n, seed, sweeps, T0, T1):
    np.random.seed(seed)
    x = np.random.normal(0.0, np.sqrt(n) * 0.6, 2 * n)
    for k in range(sweeps):
        T = T0 * (T1 / T0) ** (k / (sweeps - 1))
        step = max(0.6 * np.sqrt(T / T0), 0.003)
        for _ in range(n):
            i = np.random.randint(n)
            ox, oy = x[2 * i], x[2 * i + 1]
            nx, ny = ox + step * np.random.normal(), oy + step * np.random.normal()
            dE = 0.5 * (nx * nx + ny * ny - ox * ox - oy * oy)
            for j in range(n):
                if j != i:
                    px, py = x[2 * j], x[2 * j + 1]
                    dE += 1.0 / np.sqrt((nx - px) ** 2 + (ny - py) ** 2) - 1.0 / np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                x[2 * i], x[2 * i + 1] = nx, ny
    return x


def quench(x0, n):
    res = minimize(trap_eg, x0, args=(n,), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 20000})
    return res.fun, res.x


def batch_job(args):
    n, b = args
    rng = np.random.default_rng(1000 * n + b)
    t0 = time.perf_counter()
    eq = []
    for _ in range(QUENCHES_PER_BATCH):
        e, _ = quench(rng.normal(scale=np.sqrt(n) * 0.6, size=2 * n), n)
        eq.append(e)
    tq = (time.perf_counter() - t0) / QUENCHES_PER_BATCH
    t0 = time.perf_counter()
    es = []
    best_x = None
    for r in range(SA_PER_BATCH):
        x = anneal_positions(n, 7919 * n + 101 * b + r, SA_SWEEPS, SA_T0, SA_T1)
        e, xq = quench(x, n)
        es.append(e)
        if best_x is None or e < min(es[:-1] or [np.inf]):
            best_x = xq
    ts = (time.perf_counter() - t0) / SA_PER_BATCH
    return n, b, eq, es, tq, ts, best_x.tolist()


def distinct(e, tol=1e-7):
    e = np.sort(np.asarray(e))
    d = [e[0]]
    for v in e[1:]:
        if v - d[-1] > tol * abs(v):
            d.append(v)
    return len(d)


def census():
    from multiprocessing import Pool
    anneal_positions(10, 0, 10, SA_T0, SA_T1)            # compile once before forking work
    jobs = [(n, b) for n in NS for b in range(BATCHES)]
    with Pool(min(24, os.cpu_count())) as pool:
        out = pool.map(batch_job, jobs, chunksize=1)
    R = {}
    for n in NS:
        rows = [o for o in out if o[0] == n]
        egs = min(min(min(o[2]), min(o[3])) for o in rows)
        tol = 1e-7 * abs(egs)
        nmin = [distinct(o[2]) for o in rows]
        pq = [np.mean(np.abs(np.array(o[2]) - egs) <= tol) for o in rows]
        ps = [np.mean(np.abs(np.array(o[3]) - egs) <= tol) for o in rows]
        tq = float(np.mean([o[4] for o in rows]))
        ts = float(np.mean([o[5] for o in rows]))
        best = min(rows, key=lambda o: min(o[3]))
        R[n] = {"E_gs": egs, "distinct_per_batch": nmin, "p_quench": pq, "p_anneal": ps,
                "t_quench": tq, "t_anneal": ts, "gs_positions": best[6] if n in (20, 60) else None}
    (HERE / "free_census.json").write_text(json.dumps(R, indent=1, default=float))
    return R


# ------------------------------------------------------------------ plotting
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 400})
    return plt


UP, DOWN, INK, MUTED, GRID = "#2f6db5", "#b8661a", "#1b1f24", "#6b7280", "#d4d8de"


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight", pad_inches=0.02)


def mean_sem(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def plot_census(R):
    plt = _style()
    Ns = np.array(sorted(int(k) for k in R))
    get = lambda key: np.array([mean_sem(R[str(n)][key]) for n in Ns])
    nmin, pq, ps = get("distinct_per_batch"), get("p_quench"), get("p_anneal")
    tq = np.array([R[str(n)]["t_quench"] for n in Ns])
    ts = np.array([R[str(n)]["t_anneal"] for n in Ns])
    # time to solution with the same relative error as p
    with np.errstate(divide="ignore", invalid="ignore"):
        ttsq, ttss = tq / pq[:, 0], ts / ps[:, 0]
        ettsq, ettss = ttsq * pq[:, 1] / pq[:, 0], ttss * ps[:, 1] / ps[:, 0]
    fit = {}
    for name, y in (("minima", nmin[:, 0]), ("tts_quench", ttsq), ("tts_anneal", ttss)):
        ok = np.isfinite(y) & (y > 0) & (Ns >= 30)
        slope, icpt = np.polyfit(Ns[ok], np.log(y[ok]), 1)
        fit[name] = (slope, icpt)
    fig, axs = plt.subplots(3, 1, figsize=(3.375, 4.6), sharex=True)
    a = axs[0]
    a.errorbar(Ns, nmin[:, 0], yerr=nmin[:, 1], fmt="o", ms=3.5, color=UP, ecolor=UP, elinewidth=0.8, capsize=2)
    xx = np.linspace(30, 100, 50)
    a.plot(xx, np.exp(fit["minima"][1] + fit["minima"][0] * xx), color=MUTED, lw=0.8, ls=(0, (3, 2)))
    a.set_yscale("log")
    a.set_ylabel(f"distinct minima\nper {QUENCHES_PER_BATCH} quenches")
    a.text(0.03, 0.88, "(a)", transform=a.transAxes)
    a = axs[1]
    a.errorbar(Ns, pq[:, 0], yerr=pq[:, 1], fmt="o", ms=3.5, color=DOWN, ecolor=DOWN, elinewidth=0.8, capsize=2,
               label="random quench")
    a.errorbar(Ns + 1.2, ps[:, 0], yerr=ps[:, 1], fmt="s", ms=3.2, color=UP, ecolor=UP, elinewidth=0.8, capsize=2,
               label="annealing")
    a.set_yscale("log")
    a.set_ylabel("success probability\nper attempt")
    a.legend(frameon=False, loc="lower left")
    a.text(0.03, 0.88, "(b)", transform=a.transAxes)
    a = axs[2]
    a.errorbar(Ns, ttsq, yerr=ettsq, fmt="o", ms=3.5, color=DOWN, ecolor=DOWN, elinewidth=0.8, capsize=2,
               label="random quench")
    a.errorbar(Ns + 1.2, ttss, yerr=ettss, fmt="s", ms=3.2, color=UP, ecolor=UP, elinewidth=0.8, capsize=2,
               label="annealing")
    for key, c in (("tts_quench", DOWN), ("tts_anneal", UP)):
        a.plot(xx, np.exp(fit[key][1] + fit[key][0] * xx), color=c, lw=0.8, ls=(0, (3, 2)))
    a.set_yscale("log")
    a.set_ylabel("time to solution (s)")
    a.set_xlabel("ions $N$")
    a.text(0.03, 0.88, "(c)", transform=a.transAxes)
    for a, lab in zip(axs, "abc"):
        for t in list(a.texts):
            t.remove()
        a.text(0.0, 1.02, f"({lab})", transform=a.transAxes, va="bottom")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.subplots_adjust(hspace=0.28)
    save(fig, "fig_free_landscape")
    summary = {"fit_per_ion": {k: v[0] for k, v in fit.items()},
               "N": Ns.tolist(), "minima": nmin.tolist(), "p_quench": pq.tolist(), "p_anneal": ps.tolist(),
               "tts_quench": ttsq.tolist(), "tts_anneal": ttss.tolist(), "t_quench": tq.tolist(),
               "t_anneal": ts.tolist(), "batches": BATCHES, "quenches_per_batch": QUENCHES_PER_BATCH,
               "anneals_per_batch": SA_PER_BATCH, "sa_sweeps": SA_SWEEPS}
    (HERE / "free_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["fit_per_ion"]))


def schematic(R):
    plt = _style()
    from matplotlib.patches import Circle
    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.18, top=0.88)
    # (a) free crystal: the computed ground state of N = 20 ions in the harmonic trap
    a = fig.add_subplot(gs[0])
    pos = np.array(R["20"]["gs_positions"]).reshape(-1, 2)
    rr = np.linspace(0.5, 3.6, 7)
    for r in rr:
        a.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=0.6))
    a.scatter(pos[:, 0], pos[:, 1], s=46, color=UP, edgecolor="white", lw=0.7, zorder=3)
    a.text(0.0, -4.35, "instance: the ion number $N$\npositions are continuous", ha="center", va="top",
           fontsize=7, color=INK)
    a.set_xlim(-4, 4); a.set_ylim(-5.3, 4.0); a.set_aspect("equal"); a.axis("off")
    # (b) pinned crystal: a prescribed set of wells, k of them occupied
    a = fig.add_subplot(gs[1])
    rng = np.random.default_rng(3)
    tri = np.array([(i + 0.5 * (j % 2), j * np.sqrt(3) / 2) for i in range(-5, 6) for j in range(-5, 6)])
    tri = tri[np.hypot(tri[:, 0], tri[:, 1]) < 3.6] * 1.0
    tri += rng.normal(scale=0.12, size=tri.shape)
    keep = rng.random(len(tri)) > 0.18
    sites = tri[keep]
    k = 20
    occ = pinned_ground(sites, k)
    a.scatter(sites[:, 0], sites[:, 1], s=46, facecolor="white", edgecolor=MUTED, lw=0.7, zorder=2)
    a.scatter(sites[occ, 0], sites[occ, 1], s=46, color=DOWN, edgecolor="white", lw=0.7, zorder=3)
    a.text(0.0, -4.35, f"instance: sites $P$ ({len(sites)}) and ion number $k$ ({k})\nconfiguration = occupied subset",
           ha="center", va="top", fontsize=7, color=INK)
    a.set_xlim(-4, 4); a.set_ylim(-5.3, 4.0); a.set_aspect("equal"); a.axis("off")
    # (c) the two are limits of one potential: smooth trap plus a lattice of depth V0
    a = fig.add_subplot(gs[2])
    x = np.linspace(-3.0, 3.0, 1200)
    period = 0.8
    free_x = np.array([-2.15, -1.3, -0.45, 0.45, 1.3, 2.15])          # Coulomb-spaced, off-lattice
    wells = np.arange(-3, 4) * period                                   # -2.4 ... 2.4
    pinned_x = np.array([-2.4, -1.6, -0.8, 0.8, 1.6, 2.4])              # six ions, one well left empty
    rows = ((4.6, 0.0, free_x, "$V_0=0$: free", UP),
            (2.3, 0.25, 0.55 * free_x + 0.45 * pinned_x, "intermediate $V_0$", MUTED),
            (0.0, 0.9, pinned_x, "$V_0\\gg$ Coulomb scale:\npinned", DOWN))
    for off, V0, ions, lab, col in rows:
        U = lambda z: 0.1 * z ** 2 + V0 * 0.5 * (1 - np.cos(2 * np.pi * z / period))
        a.plot(x, U(x) + off, color=col, lw=1.1)
        a.scatter(ions, U(ions) + off + 0.16, s=26, color=col, edgecolor="white", lw=0.6, zorder=3)
        a.text(3.25, off + 0.35, lab, fontsize=7, color=INK, va="center", linespacing=1.0)
    a.scatter([0.0], [0.0 + 0.16], s=26, facecolor="white", edgecolor=DOWN, lw=0.8, zorder=3)   # empty well
    a.annotate("", xy=(-3.35, -0.1), xytext=(-3.35, 5.3), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    a.text(-3.5, 2.6, "lattice depth $V_0$", rotation=90, fontsize=7, ha="right", va="center")
    a.set_xlim(-3.9, 6.6); a.set_ylim(-0.4, 6.0); a.axis("off")
    for ax in fig.axes:
        ax.set_anchor("N")
    fig.canvas.draw()
    for ax, title in zip(fig.axes, ["(a) free crystal", "(b) pinned crystal", "(c) one trap, two limits"]):
        fig.text(ax.get_position().x0, 0.98, title, ha="left", va="top", fontsize=8, color=INK)
    save(fig, "fig_free_vs_pinned")


def pinned_ground(sites, k, sweeps=3000, seed=0):
    """lowest-energy occupation found by annealing swaps (for the schematic only)"""
    rng = np.random.default_rng(seed)
    d = np.hypot(*(sites[:, None, :] - sites[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    W = 1.0 / d
    best = None
    for rep in range(6):
        occ = np.zeros(len(sites), bool)
        occ[rng.choice(len(sites), k, replace=False)] = True
        E = 0.5 * W[np.ix_(occ, occ)].sum()
        for s in range(sweeps):
            T = 0.5 * (1e-4 / 0.5) ** (s / (sweeps - 1))
            i = rng.choice(np.nonzero(occ)[0]); j = rng.choice(np.nonzero(~occ)[0])
            dE = W[j, occ].sum() - W[j, i] - W[i, occ].sum()
            if dE < 0 or rng.random() < np.exp(-dE / T):
                occ[i], occ[j] = False, True
                E += dE
        if best is None or E < best[0]:
            best = (E, occ.copy())
    return best[1]


if __name__ == "__main__":
    parts = sys.argv[1:] or ["census", "schematic"]
    path = HERE / "free_census.json"
    if "census" in parts:
        R = census()
    R = json.loads(path.read_text())
    plot_census(R)
    if "schematic" in parts:
        schematic(R)
