"""Interval-arithmetic certification of the lattice sums in the cut budget (manuscript Table II,
Lemmas 6-8). Every quantity is an upper bound on a sum of |J| over pairs of selector centres.

Per pair we bound |J_R| <= (|A(theta)| + 6e-4) b^4 r^-5, which combines the angular law of
Lemma 4 with its remainder |R_J| <= 6e4 b^5 r^-6 <= 6e-4 b^4 r^-5 (t = b/r <= 1e-8).
For lattice vectors (i, j) the angular factor is an exact rational:
    cos 4 theta = 1 - 8 i^2 j^2 / (i^2 + j^2)^2,   A = 9/2 - (105/2) cos 4 theta.
Finite parts are summed in mpmath interval arithmetic (outward rounding); tails are bounded
analytically and added as intervals [0, T].

Usage: python certify_budget.py   (about a minute)
"""
from fractions import Fraction

from mpmath import iv

iv.dps = 30
REM = Fraction(6, 10000)          # remainder allowance per pair, in units of b^4 r^-5
AMAX = Fraction(57) + REM         # |A| <= 57 everywhere
N0 = 300                          # truncation of each sector sum


def ivq(q):
    q = Fraction(q)
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)


def weight(i, j):
    """certified upper bound on |A(theta)| + REM for the lattice vector (i, j)"""
    s = i * i + j * j
    c4 = 1 - Fraction(8 * i * i * j * j, s * s)
    return abs(Fraction(9, 2) - Fraction(105, 2) * c4) + REM


def r5(s):
    """interval for s^(-5/2), s an integer"""
    S = iv.mpf(s)
    return 1 / (S * S * iv.sqrt(S))


def zeta_tail(p, n):
    """interval enclosing sum_{k>=n} k^-p (integral bounds)"""
    lo = iv.mpf(n) ** (1 - p) / (p - 1)
    hi = iv.mpf(n - 1) ** (1 - p) / (p - 1)
    return iv.mpf([lo.a, hi.b])


def kpow(p, a, b):
    """interval for sum_{k=a}^{b} k^-p"""
    tot = iv.mpf(0)
    for k in range(a, b + 1):
        tot += 1 / iv.mpf(k) ** p
    return tot


def report(name, x):
    print(f"{name:42s} in [{float(x.a):.6f}, {float(x.b):.6f}]")
    return x


w_axis = ivq(48 + REM)

# same straight run: pairs at path distance k >= 2, multiplicity k, axis weight
same_run = w_axis * (kpow(4, 2, 2000) + zeta_tail(4, 2001))
# opposite collinear branch: pairs (i, j), i, j >= 1 at distance i + j, multiplicity k - 1 at distance k
collinear = w_axis * (kpow(4, 2, 2000) - kpow(5, 2, 2000) + zeta_tail(4, 2001))
# one perpendicular sector: sum_{i, j >= 1} w(i, j) (i^2 + j^2)^-5/2
sector = iv.mpf(0)
for i in range(1, N0 + 1):
    for j in range(1, N0 + 1):
        sector += ivq(weight(i, j)) * r5(i * i + j * j)
# tail: pairs with i > N0 or j > N0;  sum_{j>=1} (i^2+j^2)^-5/2 <= (2/3) i^-4, so the tail is
# at most 2 * AMAX * (2/3) * sum_{i>N0} i^-4
sector += iv.mpf([0, (2 * ivq(AMAX) * ivq(Fraction(2, 3)) * zeta_tail(4, N0 + 1)).b])
two_perp = 2 * sector
# objective box (Lemma 6): distinguished pair 57 * 2^-5/2 plus halo sum_{k>=2} k (k^2+1)^-5/2
halo = iv.mpf(0)
for k in range(2, 2001):
    halo += k * r5(k * k + 1)
halo += iv.mpf([0, zeta_tail(4, 2001).b])                     # k (k^2+1)^-5/2 <= k^-4
box_abs = ivq(57 + REM) * r5(2) + ivq(AMAX) * halo
# far end of a run at path distance >= 2D, D >= 1000: crude sector tail
far_end = iv.mpf([0, (2 * ivq(AMAX) * ivq(Fraction(2, 3)) * zeta_tail(4, 2000)).b])
# U-turn arm-to-arm: sum_{(i,j) != (0,0)} w (( i+j)^2 + 1)^-5/2, multiplicity k+1 at k = i + j
uturn = iv.mpf(0)
for k in range(1, 2001):
    w_max = max(weight(k, 1), weight(1, k))                 # vector (k, 1) up to reflection
    uturn += (k + 1) * ivq(w_max) * r5(k * k + 1)
uturn += iv.mpf([0, (ivq(AMAX) * (zeta_tail(4, 2001) + zeta_tail(5, 2001))).b])

print("Certified bounds (units of b^4):")
report("same straight run", same_run)
report("two perpendicular branches (2 x sector)", two_perp)
report("opposite collinear branch", collinear)
report("far end of the run", far_end)
report("one objective box, absolute load", box_abs)
total = report("TOTAL adverse load, worst cut", same_run + two_perp + collinear + far_end + box_abs)
report("U-turn arm-to-arm", uturn)
report("TOTAL, cut inside a U-turn", same_run + uturn + box_abs + far_end)
jnn = ivq(48) - ivq(REM)
print(f"favourable |J_nn| >= {float(jnn.a):.6f};  certified margin >= {float((jnn - total).a):.4f} b^4")
