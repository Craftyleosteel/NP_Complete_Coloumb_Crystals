"""Component-flip budget (Emmerich 2026, Sec. 8-9) re-evaluated for Riesz exponent s.
Leading-order selector coupling J(R) = A_s(theta) b^4 r^{-(s+4)} with
  s=2: A = 32 - 192 cos 4θ ;  s=1: A = 9/2 - (105/2) cos 4θ   (see multipole_law.py).
All loads in units of b^4. 'abs' = sum of |J| (what the proof must charge, since spins
across the cut are arbitrary); 'signed' = sum of J when the rest of the tree is uniform.
"""
import numpy as np

LAWS = {2: lambda th: 32 - 192 * np.cos(4 * th), 1: lambda th: 4.5 - 52.5 * np.cos(4 * th)}
N = 1500  # truncation of branch lengths (tails beyond are < 1e-9 here)


def J(s, dx, dy):
    r = np.hypot(dx, dy)
    return LAWS[s](np.arctan2(dy, dx)) * r ** (-(s + 4))


def load(s, A, B):
    """A, B: arrays of cell centres on the two sides of the cut."""
    dx = B[:, 0][None, :] - A[:, 0][:, None]
    dy = B[:, 1][None, :] - A[:, 1][:, None]
    Jm = J(s, dx, dy)
    return np.abs(Jm).sum(), Jm.sum()


k = np.arange(1, N)
ray = lambda d: np.stack([k * d[0], k * d[1]], 1).astype(float)
origin = np.zeros((1, 2))

for s in (2, 1):
    Jnn = J(s, 1, 0)
    # straight run: cut between cell 0 and 1; exclude the nearest pair itself
    left = np.concatenate([origin, ray((-1, 0))]); right = ray((1, 0))
    a, g = load(s, left, right); a -= abs(Jnn); g -= Jnn
    straight = (a, g)
    # root cut, flipped branch +y, remaining branches +x and -x (two perpendicular sectors + root)
    flip = ray((0, 1))
    stay = np.concatenate([origin, ray((1, 0)), ray((-1, 0))])
    a, g = load(s, stay, flip); a -= abs(Jnn); g -= Jnn
    root2perp = (a, g)
    # bend cut: flipped +y branch, remaining = bend cell + +x branch
    stay = np.concatenate([origin, ray((1, 0))])
    a, g = load(s, stay, flip); a -= abs(Jnn); g -= Jnn
    bend = (a, g)
    # objective box (Sec. 8): p_i = (-i,0), q_j = (1+j,1)
    i = np.arange(0, N); P = np.stack([-i, 0 * i], 1).astype(float); Q = np.stack([1 + i, 0 * i + 1], 1).astype(float)
    a, g = load(s, P, Q)
    lam = g; objabs = a
    # U-turn arm-to-arm (Sec. 9): prefix (1+D-i,1), continuation (1+D+j,0), excluding unit step
    ii = np.arange(0, N)
    pre = np.stack([-ii, 0 * ii + 1], 1).astype(float); cont = np.stack([ii, 0 * ii], 1).astype(float)
    Jm = J(s, cont[:, 0][None, :] - pre[:, 0][:, None], cont[:, 1][None, :] - pre[:, 1][:, None])
    Jm[0, 0] = 0.0
    uturn = (np.abs(Jm).sum(), Jm.sum())
    adverse_root = straight[0] + root2perp[0] + objabs
    print(f"--- s={s} ---  |J_nn| = {abs(Jnn):.3f}")
    print(f"  straight tail      abs {straight[0]:7.3f}  signed {straight[1]:7.3f}")
    print(f"  root, 2 perp+root  abs {root2perp[0]:7.3f}  signed {root2perp[1]:7.3f}")
    print(f"  bend               abs {bend[0]:7.3f}  signed {bend[1]:7.3f}")
    print(f"  U-turn arms        abs {uturn[0]:7.3f}  signed {uturn[1]:7.3f}")
    print(f"  objective lambda = {lam:7.3f}   objective abs load = {objabs:7.3f}")
    print(f"  worst root cut: adverse abs load {adverse_root:7.3f}  vs favourable {abs(Jnn):7.3f}"
          f"  -> margin {abs(Jnn) - adverse_root:+7.3f}")
