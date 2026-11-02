"""Figures for the PRA manuscript (single column 3.375 in, double column 7.0 in).

Fig. 1  the reduction: cube graph Q3 -> orthogonal drawing -> selector trees -> objective box
Fig. 2  angular coupling law A_s(theta) for s = 1 (Coulomb) and s = 2
Fig. 3  worst-case consistency budget at s = 1
Palette validated for colour-vision deficiency (dataviz validate_palette.js, all checks pass).
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).resolve().parent
UP, DOWN, AF = "#2f6db5", "#b8661a", "#8a3fb0"     # spin +1, spin -1, antiferro bond
INK, MUTED, GRID = "#1b1f24", "#6b7280", "#d4d8de"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 400,
})


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
V = {"o0": (4, 0), "o1": (8, 4), "o2": (4, 8), "o3": (0, 4),
     "i0": (4, 2), "i1": (6, 4), "i2": (4, 6), "i3": (2, 4)}
SPIN = {v: (+1 if v in ("o0", "o2", "i1", "i3") else -1) for v in V}   # a ground state of H_B
ROUTES = [  # orthogonal drawing of the cube graph: polylines from u to v
    ("o0", "o1", [(4, 0), (8, 0), (8, 4)]), ("o1", "o2", [(8, 4), (8, 8), (4, 8)]),
    ("o2", "o3", [(4, 8), (0, 8), (0, 4)]), ("o3", "o0", [(0, 4), (0, 0), (4, 0)]),
    ("i0", "i1", [(4, 2), (6, 2), (6, 4)]), ("i1", "i2", [(6, 4), (6, 6), (4, 6)]),
    ("i2", "i3", [(4, 6), (2, 6), (2, 4)]), ("i3", "i0", [(2, 4), (2, 2), (4, 2)]),
    ("o0", "i0", [(4, 0), (4, 2)]), ("o1", "i1", [(8, 4), (6, 4)]),
    ("o2", "i2", [(4, 8), (4, 6)]), ("o3", "i3", [(0, 4), (2, 4)]),
]


def col(v):
    return UP if SPIN[v] > 0 else DOWN


def point_at(poly, t):
    P = np.array(poly, float)
    for a, c in zip(P[:-1], P[1:]):
        L = np.linalg.norm(c - a)
        if t <= L + 1e-12:
            return a + (c - a) * t / L
        t -= L
    return P[-1]


def length(poly):
    P = np.array(poly, float)
    return np.linalg.norm(np.diff(P, axis=0), axis=1).sum()


fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.78), gridspec_kw={"width_ratios": [1, 1, 1, 1.55]})
for ax in axs:
    ax.set_aspect("equal")
    ax.axis("off")

# (a) abstract cube graph, coloured by a ground state of H_B
ax = axs[0]
A = {"o0": (0, 0), "o1": (3, 0), "o2": (3, 3), "o3": (0, 3),
     "i0": (1, 1), "i1": (2, 1), "i2": (2, 2), "i3": (1, 2)}
for u, v, _ in ROUTES:
    ax.plot(*zip(A[u], A[v]), color=INK, lw=0.8, zorder=1)
for v, p in A.items():
    ax.scatter(*p, s=38, color=col(v), edgecolor="white", lw=0.6, zorder=3)
ax.set_xlim(-0.4, 3.4)
ax.set_ylim(-0.4, 3.4)

# (b) orthogonal grid drawing
ax = axs[1]
for x in range(9):
    ax.plot([x, x], [-0.3, 8.3], color=GRID, lw=0.3, zorder=0)
    ax.plot([-0.3, 8.3], [x, x], color=GRID, lw=0.3, zorder=0)
for u, v, poly in ROUTES:
    ax.plot(*zip(*poly), color=INK, lw=0.9, zorder=1)
for v, p in V.items():
    ax.scatter(*p, s=30, color=col(v), edgecolor="white", lw=0.6, zorder=3)

# (c) selector trees: cells along each half-edge, an objective box at each edge midpoint
ax = axs[2]
h = 0.42
for u, v, poly in ROUTES:
    L = length(poly)
    mid = L / 2
    for t in np.arange(h, L - h / 2, h):
        if abs(t - mid) < 0.3:
            continue
        ax.scatter(*point_at(poly, t), s=3.2, color=col(u) if t < mid else col(v), lw=0, zorder=2)
    q = point_at(poly, mid)
    ax.add_patch(Rectangle((q[0] - 0.33, q[1] - 0.33), 0.66, 0.66, fill=False, ec=AF, lw=0.8, zorder=3))
for v, p in V.items():
    ax.scatter(*p, s=26, facecolor="white", edgecolor=col(v), lw=1.1, zorder=4)
ax.set_xlim(-0.6, 8.6)
ax.set_ylim(-0.6, 8.6)

# (d) objective-box detail, selectors drawn with exaggerated half-side b
ax = axs[3]
b = 0.17
chain_u = [(-i, 0) for i in range(0, 5)]
chain_v = [(1 + j, 1) for j in range(0, 4)] + [(4, 0), (5, 0), (6, 0)]


def selector(c, spin, color):
    x, y = c
    ax.add_patch(Rectangle((x - b, y - b), 2 * b, 2 * b, fill=False, ec="#aab1ba", lw=0.5, zorder=2))
    d = np.array([(b, b), (-b, -b)]) if spin > 0 else np.array([(b, -b), (-b, b)])
    ax.plot(x + d[:, 0], y + d[:, 1], color=color, lw=1.4, zorder=3, solid_capstyle="round")
    ax.scatter(x + d[:, 0], y + d[:, 1], s=7, color=color, zorder=4)


for chain in (chain_u, chain_v):
    for p, q in zip(chain[:-1], chain[1:]):
        ax.plot(*zip(p, q), color=INK, lw=0.7, zorder=1)
ax.plot([0, 1], [0, 1], color=AF, lw=1.0, ls=(0, (3, 2)), zorder=1)
for c in chain_u:
    selector(c, +1, UP)
for c in chain_v:
    selector(c, -1, DOWN)
ax.annotate("$45^\\circ$ pair: $J\\approx+10\\,b^4$", xy=(0.5, 0.5), xytext=(-4.4, 2.35), fontsize=7,
            color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
ax.annotate("axis pair: $J=-48\\,b^4$", xy=(5.5, -0.1), xytext=(2.4, -1.45), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
ax.text(-4.4, -0.85, "tree $u$, $\\sigma_u=+1$", fontsize=7, color=INK)
ax.text(1.0, 1.65, "tree $v$, $\\sigma_v=-1$", fontsize=7, color=INK)
ax.set_xlim(-4.6, 6.6)
ax.set_ylim(-1.8, 2.8)
fig.subplots_adjust(wspace=0.12, top=0.87, bottom=0.01)
# panel titles on one common baseline, aligned with each panel's left edge
for a in axs:
    a.set_anchor("N")
fig.canvas.draw()
for a, title in zip(axs, ["(a) planar cubic graph", "(b) orthogonal drawing", "(c) selector trees",
                          "(d) objective box"]):
    fig.text(a.get_position().x0, 0.995, title, ha="left", va="top", fontsize=8, color=INK)
save(fig, "fig1_reduction")

# ---------------------------------------------------------------- Fig. 2
th = np.linspace(0, 90, 721)
r = np.radians(th)
A1 = (4.5 - 52.5 * np.cos(4 * r)) / 48
A2 = (32 - 192 * np.cos(4 * r)) / 160
fig, ax = plt.subplots(figsize=(3.375, 2.0))
ax.axhline(0, color=INK, lw=0.6)
ferro_edge = np.degrees(np.arccos(4.5 / 52.5)) / 4
ax.axvspan(0, ferro_edge, color=GRID, alpha=0.45, lw=0)
ax.axvspan(90 - ferro_edge, 90, color=GRID, alpha=0.45, lw=0)
ax.plot(th, A2, color=MUTED, lw=1.2, ls=(0, (4, 2)), label="$s=2$ (Ref. [8])")
ax.plot(th, A1, color=UP, lw=1.6, label="$s=1$ (Coulomb)")
ax.scatter([0, 45, 90], [-1, 57 / 48, -1], s=14, color=UP, zorder=3)
ax.text(45, 57 / 48 - 0.27, "$57/48$", ha="center", fontsize=7, color=INK)
ax.text(45, 1.4 + 0.08, "$224/160$", ha="center", fontsize=7, color=MUTED)
ax.text(ferro_edge / 2, 0.3, "ferro", ha="center", fontsize=7, color=INK)
ax.text(90 - ferro_edge / 2, 0.3, "ferro", ha="center", fontsize=7, color=INK)
ax.text(45, -0.55, "antiferro", ha="center", fontsize=7, color=INK)
ax.set_xlim(0, 90)
ax.set_ylim(-1.25, 1.85)
ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel(r"bond angle $\theta$ (deg)")
ax.set_ylabel(r"$J_R\, r^{s+4}/(b^4\,|A_s(0)|)$")
ax.legend(frameon=False, loc="upper left", handlelength=2.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
save(fig, "fig2_angular_law")

# ---------------------------------------------------------------- Fig. 3
parts = [("", 3.952), ("two perpendicular branches", 23.056), ("", 2.179),
         ("objective box", 13.1)]
shades = ["#9aa3ad", "#c4cad1", "#7d8792", "#aeb6bf"]
fig, ax = plt.subplots(figsize=(3.375, 1.45))
left = 0
for (lab, w), c in zip(parts, shades):
    ax.barh(1, w, left=left, height=0.55, color=c, edgecolor="white", lw=1.0)
    if lab and w > 3:
        ax.text(left + w / 2, 1, lab.replace(" branches", "\nbranches").replace(" box", "\nbox"),
                ha="center", va="center", fontsize=6.3, color=INK, linespacing=0.95)
    left += w
ax.text(left + 0.6, 1, f"{left:.1f}", va="center", fontsize=7, color=INK)
ax.annotate("collinear branch", xy=(3.952 + 23.056 + 1.1, 1.28), xytext=(21, 1.66), fontsize=6.3,
            color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
ax.annotate("same run", xy=(2.0, 1.28), xytext=(0.5, 1.66), fontsize=6.3,
            color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
ax.barh(0, 48, height=0.55, color=UP, edgecolor="white", lw=1.0)
ax.text(48.6, 0, "48.0", va="center", fontsize=7, color=INK)
ax.set_yticks([0, 1])
ax.set_yticklabels(["favourable $|J_{nn}|$", "adverse load"])
ax.set_xlim(0, 54)
ax.set_ylim(-0.5, 1.9)
ax.set_xlabel(r"coupling across the cut ($b^4$)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
save(fig, "fig3_budget")
print("ok")

# ---------------------------------------------------------------- worst cut of a tree (Lemma 7)
fig, ax = plt.subplots(figsize=(3.375, 2.35))
stay = [(i, 0) for i in range(-6, 7)]                      # root (0,0) and the two collinear branches
flip = [(0, j) for j in range(1, 7)]                       # the branch on the far side of the cut
for (x0, y0), (x1, y1) in zip(stay[:-1], stay[1:]):
    ax.plot([x0, x1], [y0, y1], color=INK, lw=0.7, zorder=1)
for (x0, y0), (x1, y1) in zip(flip[:-1], flip[1:]):
    ax.plot([x0, x1], [y0, y1], color=INK, lw=0.7, zorder=1)
# representative cross-cut pairs, one per class
ax.plot([0, 0], [0, 1], color=INK, lw=2.2, zorder=1)                       # the cut edge itself
ax.plot([-0.35, 0.35], [0.38, 0.62], color="#c2364a", lw=1.4, zorder=4)     # cut marker
for (x, y) in [(1, 0), (-1, 0)]:
    ax.plot([x, 0], [y, 1], color=AF, lw=1.0, ls=(0, (3, 2)), zorder=2)     # 45 degree pairs
for (x, y, q) in [(3, 0, 2), (5, 0, 4), (-3, 0, 3), (-4, 0, 5)]:
    ax.plot([x, 0], [y, q], color=AF, lw=0.5, alpha=0.55, zorder=2)         # sector pairs
ax.annotate("", xy=(0.3, 3.0), xytext=(0.3, 1.15), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.scatter(*zip(*stay), s=22, color=UP, edgecolor="white", lw=0.5, zorder=3)
ax.scatter(*zip(*flip), s=22, color=DOWN, edgecolor="white", lw=0.5, zorder=3)
ax.scatter([0], [0], s=46, facecolor="white", edgecolor=UP, lw=1.3, zorder=4)
ax.add_patch(Rectangle((-0.55, 6.25), 1.1, 0.75, fill=False, ec=AF, lw=0.9))
ax.text(0.8, 6.55, "objective box (one, at most)", fontsize=6.6, va="center", color=INK)
ax.text(0.5, 2.0, "same run", fontsize=6.6, color=INK)
ax.text(2.0, 2.6, "perpendicular\nsector pairs", fontsize=6.6, color=INK, linespacing=1.0)
ax.text(-1.05, 0.55, "$45^\circ$ pairs", fontsize=6.6, color=AF, ha="right")
ax.text(0.5, 0.45, "cut edge", fontsize=6.6, color="#c2364a")
ax.text(-6.3, -0.75, "kept side: root and two collinear branches", fontsize=6.6, color=UP)
ax.text(-0.5, 4.3, "flipped\nbranch", fontsize=6.6, color=DOWN, ha="right", linespacing=1.0)
ax.set_xlim(-6.6, 6.6); ax.set_ylim(-1.1, 7.2); ax.set_aspect("equal"); ax.axis("off")
save(fig, "fig_cut")
print("cut figure ok")
