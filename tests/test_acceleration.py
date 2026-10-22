"""The weights of the series acceleration (Cohen, Rodriguez Villegas and Zagier's algorithm 1) against exact values
(CORRECTIONS.md, entry 12).

With p_j the coefficients of T_n(1 + 2x), positive integers (p_0 = 1, p_{j+1} = p_j 2 (n+j)(n-j) / ((2j+1)(j+1))), the
weights are w_k = (p_{k+1} + ... + p_n) / (p_0 + ... + p_n), k = 0, ..., n-1; exact as fractions. The smallest weight,
p_n / T_n(3), falls like (4 / 5.83)^n. Gate: every weight within a relative 1e-14 of its exact value, for n = 2 to 96
half-periods (1 to 48 periods; the model uses 12).
"""
from fractions import Fraction

import numpy as np

from common import load_h2, row

N = (2, 4, 8, 16, 24, 32, 48, 64, 96)


def exact(n):
    p = [Fraction(1)]
    for k in range(n):
        p.append(p[-1] * 2 * (n + k) * (n - k) / ((2 * k + 1) * (k + 1)))
    d = sum(p)
    return [sum(p[k + 1:]) / d for k in range(n)]


def run(ctx):
    load_h2()
    from semi_markov.base import lib, ptr_double
    rows = []
    worst, where = 0.0, ""
    for n in N:
        w = exact(n)
        out = np.empty(n)
        lib.acc_coefficients(out.ctypes.data_as(ptr_double), n)
        rel = [float(abs(Fraction(float(o)) / x - 1)) for o, x in zip(out, w)]
        k = int(np.argmax(rel))
        if rel[k] >= worst:
            worst, where = rel[k], f"n = {n}, k = {k}, weight {float(w[k]):.2g}"
        if n == 24:
            rows.append(row("acceleration", "weights, 12 periods (the model's): max relative error", rel[k], "<= 1e-14",
                            rel[k] <= 1e-14, f"k = {k}, weight {float(w[k]):.2g}"))
    rows.append(row("acceleration", "weights, 1 to 48 periods: max relative error", worst, "<= 1e-14", worst <= 1e-14,
                    f"worst at {where}"))
    return rows
