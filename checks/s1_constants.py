"""Constants needed by the s = 1 proof (write-up Sections 5-8).

1. [t^4] coefficient of the one-spin field h at s = 1.
2. Exact selector Fourier coefficients vs. the leading multipole law (sanity check).
3. Rigorous-style bounds on the objective coefficient lambda (crude |A| <= 57 halo bound).
4. Sensitivity of J and h to a perturbation eta of one selector's + diagonal, in units of
   b^3 r^-5 (for J) and b r^-3 (for h), to fix the constant in the perturbation lemma.
"""
import itertools
import mpmath as mp
import sympy as sp

mp.mp.dps = 40

# ---------- 1. symbolic h4 at s = 1
t, th = sp.symbols("t theta", real=True)
c, sn = sp.cos(th), sp.sin(th)
U = {+1: (1, 1), -1: (1, -1)}


def Fsym(sig, tau):
    tot = 0
    for eps, eta in itertools.product((1, -1), repeat=2):
        v = (eta * U[tau][0] - eps * U[sig][0], eta * U[tau][1] - eps * U[sig][1])
        z = 2 * t * (c * v[0] + sn * v[1]) + t**2 * (v[0] ** 2 + v[1] ** 2)
        tot += sp.series((1 + z) ** sp.Rational(-1, 2), t, 0, 7).removeO()
    return sp.expand(tot)


Fs = {(a, b): Fsym(a, b) for a in (1, -1) for b in (1, -1)}
h = sp.expand((Fs[1, 1] + Fs[1, -1] - Fs[-1, 1] - Fs[-1, -1]) / 4)
J = sp.expand((Fs[1, 1] - Fs[1, -1] - Fs[-1, 1] + Fs[-1, -1]) / 4)
print("h: [t^2] =", sp.simplify(sp.trigsimp(h.coeff(t, 2))), "  [t^4] =", sp.simplify(sp.trigsimp(h.coeff(t, 4))))
print("J: [t^4] =", sp.simplify(sp.trigsimp(J.coeff(t, 4))), "  [t^6] =", sp.simplify(sp.trigsimp(J.coeff(t, 6))))
print("odd orders vanish:", all(sp.simplify(J.coeff(t, k)) == 0 and sp.simplify(h.coeff(t, k)) == 0 for k in (1, 3, 5)))


# ---------- exact numerics
def pts(center, state, b, eta=0.0):
    """two selected points of a selector in a given state; + diagonal has half-side b+eta"""
    cx, cy = center
    if state == +1:
        s = b + eta
        return [(cx + s, cy + s), (cx - s, cy - s)]
    return [(cx + b, cy - b), (cx - b, cy + b)]


def Fexact(R, sig, tau, b, eta_i=0.0, eta_j=0.0):
    A = pts((0, 0), sig, b, eta_i)
    B = pts(R, tau, b, eta_j)
    return mp.fsum(1 / mp.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) for p in A for q in B)


def coeffs(R, b, eta_i=0.0, eta_j=0.0):
    F = {(s1, s2): Fexact(R, s1, s2, b, eta_i, eta_j) for s1 in (1, -1) for s2 in (1, -1)}
    Jv = (F[1, 1] - F[1, -1] - F[-1, 1] + F[-1, -1]) / 4
    h1 = (F[1, 1] + F[1, -1] - F[-1, 1] - F[-1, -1]) / 4
    h2 = (F[1, 1] - F[1, -1] + F[-1, 1] - F[-1, -1]) / 4
    return Jv, h1, h2


A_law = lambda theta: mp.mpf(9) / 2 - mp.mpf(105) / 2 * mp.cos(4 * theta)
b = mp.mpf("1e-4")
print("\n2. exact J / (law b^4 r^-5):")
for R in [(1, 0), (1, 1), (2, 1), (3, 0), (5, 2)]:
    r = mp.sqrt(R[0] ** 2 + R[1] ** 2); theta = mp.atan2(R[1], R[0])
    Jv, h1, _ = coeffs(R, b)
    lawJ = A_law(theta) * b**4 / r**5
    lawh = 6 * mp.sin(2 * theta) * b**2 / r**3
    print(f"   R={R}: J ratio {mp.nstr(Jv / lawJ, 10)}   h ratio {mp.nstr(h1 / lawh, 10) if lawh != 0 else 'h=' + mp.nstr(h1, 3)}")

# ---------- 3. lambda bounds (leading law)
J0 = A_law(mp.pi / 4) / mp.sqrt(2) ** 5
halo_crude = 57 * mp.nsum(lambda k: k * (k**2 + 1) ** mp.mpf(-2.5), [2, mp.inf])
halo_signed = mp.nsum(lambda k: k * A_law(mp.atan2(1, k)) * (k**2 + 1) ** mp.mpf(-2.5), [2, mp.inf])
halo_abs = mp.nsum(lambda k: k * abs(A_law(mp.atan2(1, k))) * (k**2 + 1) ** mp.mpf(-2.5), [2, mp.inf])
print(f"\n3. distinguished J0 = {mp.nstr(J0, 8)} b^4;  halo: signed {mp.nstr(halo_signed, 6)}, |.| {mp.nstr(halo_abs, 6)}, crude {mp.nstr(halo_crude, 6)}")
print(f"   lambda (exact law) = {mp.nstr(J0 + halo_signed, 8)};  crude-rigorous: {mp.nstr(J0 - halo_crude, 6)} < lambda < {mp.nstr(J0 + halo_crude, 6)}")

# ---------- 4. perturbation sensitivity
print("\n4. dJ/deta_i in units of b^3 r^-5, dh1/deta_j and dh1/deta_i in units of b r^-3:")
worstJ = worsth = 0
d = b * mp.mpf("1e-12")
for R in [(1, 0), (1, 1), (2, 1), (1, 2), (3, 1), (4, 3), (7, 0), (5, 5)]:
    r = mp.sqrt(R[0] ** 2 + R[1] ** 2)
    J0_, h10, _ = coeffs(R, b)
    Ji, h1i, _ = coeffs(R, b, eta_i=d)
    Jj, h1j, _ = coeffs(R, b, eta_j=d)
    sJ = max(abs(Ji - J0_), abs(Jj - J0_)) / d / (b**3 / r**5)
    sh = max(abs(h1j - h10), abs(h1i - h10)) / d / (b / r**3)
    worstJ, worsth = max(worstJ, sJ), max(worsth, sh)
    print(f"   R={R}: |dJ| {mp.nstr(sJ, 5)}   |dh1| {mp.nstr(sh, 5)}")
print(f"   max over samples: J {mp.nstr(worstJ, 5)}, h {mp.nstr(worsth, 5)}")
