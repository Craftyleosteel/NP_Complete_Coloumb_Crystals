"""Angular multipole law for the four-point diagonal selector (Emmerich 2026, Lemma 5.1)
at general Riesz exponent s. Reproduces the published s=2 law as a check, then gives s=1 (Coulomb).

F_R(sigma,tau) = sum_{eps,eta} |R + b(eta u_tau - eps u_sigma)|^{-s},  u_+ = (1,1), u_- = (1,-1).
J_R = (F++ - F+- - F-+ + F--)/4,  h_R = (F++ + F+- - F-+ - F--)/4.
"""
import sympy as sp

t, th = sp.symbols("t theta", real=True)
s = sp.Symbol("s", positive=True)
c, sn = sp.cos(th), sp.sin(th)
u = {+1: (1, 1), -1: (1, -1)}


def pair_term(v, order, s_val):
    # |R + b v|^{-s} / r^{-s} = (1+z)^{-s/2},  z = 2 t (Rhat.v) + t^2 |v|^2
    a = c * v[0] + sn * v[1]
    z = 2 * t * a + t**2 * (v[0] ** 2 + v[1] ** 2)
    return sp.series((1 + z) ** (-sp.Rational(s_val) / 2), t, 0, order + 1).removeO()


def F(sig, tau, order, s_val):
    tot = 0
    for eps in (1, -1):
        for eta in (1, -1):
            v = (eta * u[tau][0] - eps * u[sig][0], eta * u[tau][1] - eps * u[sig][1])
            tot += pair_term(v, order, s_val)
    return sp.expand(tot)


def law(s_val, order=4):
    Fs = {(a, b_): F(a, b_, order, s_val) for a in (1, -1) for b_ in (1, -1)}
    J = sp.expand((Fs[1, 1] - Fs[1, -1] - Fs[-1, 1] + Fs[-1, -1]) / 4)
    h = sp.expand((Fs[1, 1] + Fs[1, -1] - Fs[-1, 1] - Fs[-1, -1]) / 4)
    J4 = sp.simplify(sp.trigsimp(J.coeff(t, 4)))
    h2 = sp.simplify(sp.trigsimp(h.coeff(t, 2)))
    J2 = sp.simplify(J.coeff(t, 2))
    return J2, J4, h2


for s_val in (2, 1):
    J2, J4, h2 = law(s_val)
    J4n = sp.simplify(sp.expand_trig(J4))
    print(f"s={s_val}:  [t^2]J = {J2}")
    print(f"        [t^4]J = {sp.simplify(J4)}")
    print(f"        [t^2]h = {sp.simplify(h2)}")
    for deg in (0, 45, 90):
        print(f"        J4(theta={deg}) = {sp.nsimplify(J4.subs(th, sp.pi*deg/180))}")
    print(f"        range of J4 over theta: min={sp.N(sp.minimum(J4, th, sp.Interval(0, sp.pi)))}, "
          f"max={sp.N(sp.maximum(J4, th, sp.Interval(0, sp.pi)))}")
