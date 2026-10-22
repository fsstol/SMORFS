"""The Gompertz, log-normal and log-logistic risks (branch speedup, entry 31).

Parameters, as Weibull's, on the log scale where they are positive: Gompertz (shape b, any sign; log rate a), log-normal
(meanlog; log sdlog), log-logistic (log shape; log scale).
1. Each risk's stack (lib.eval_risk) at several parameter sets and times, edge cases included (a defective Gompertz
   with b < 0, Gompertz near b = 0, a log-normal far in both tails, z from -24 to 50), against 50-digit mpmath: the
   closed forms of h and H, and their first and second derivatives with respect to the parameters by mpmath's
   differentiation. h and H: relative 1e-13; each derivative: 1e-12 relative to its largest value over the times.
   log t is rounded to double in the reference too: only the formulas' own rounding is measured. SciPy is not the
   reference: its log-logistic log-survival loses 7 digits in the upper tail (H = 22).
   Also dH/dt against h (central differences in t): 1e-6.
2. Each risk as the second transition of a model (a Weibull onset H -> A, then A -> B with the risk): prevalence of A
   and arrival in B against the convolution of the onset density with the risk's closed forms (1e-4 relative to the
   largest value, the inversion's accuracy at t0 = 10, as the general check), and the model's gradient and Hessian
   against central differences (1e-5, as the general check).
"""
import numpy as np
import mpmath as mp
from scipy.integrate import quad

from common import load_h2, row

TIMES = np.array([0.5, 2.0, 10.0, 30.0, 60.0, 100.0, 150.0])
mp.mp.dps = 50


def mp_h_H(kind, th, t):
    """h and H at log-scale parameters th, in mpmath. log t is taken as rounded to double, as the library computes it:
    near the points where a derivative vanishes (w = 0 for the log-logistic), that rounding alone moves the derivative
    by about 1e-14 (the problem's conditioning), and the comparison is meant to measure the formulas' own rounding."""
    lt = mp.mpf(float(np.log(float(t))))
    t = mp.mpf(t)
    if kind == "Gompertz":
        b, a = th[0], mp.exp(th[1])
        return a * mp.exp(b * t), a * mp.expm1(b * t) / b
    if kind == "LogNormal":
        sigma = mp.exp(th[1])
        z = (lt - th[0]) / sigma
        Q = mp.ncdf(-z)
        H = -mp.log1p(-mp.ncdf(z)) if z < 0 else -mp.log(Q)
        return mp.npdf(z) / (Q * sigma * t), H
    beta, alpha = mp.exp(th[0]), mp.exp(th[1])
    u = mp.exp(beta * (lt - mp.log(alpha)))
    return beta / t * u / (1 + u), mp.log1p(u)


def mp_stack(kind, th, t):
    """[h, dh (2), d2h (2 x 2), H, dH (2), d2H (2 x 2)], as lib.eval_risk."""
    out = []
    a, b = mp.mpf(float(th[0])), mp.mpf(float(th[1]))
    for comp in (0, 1):
        g = lambda x, y: mp_h_H(kind, (x, y), t)[comp]
        d11 = mp.diff(g, (a, b), (1, 1))
        out += [g(a, b), mp.diff(g, (a, b), (1, 0)), mp.diff(g, (a, b), (0, 1)),
                mp.diff(g, (a, b), (2, 0)), d11, d11, mp.diff(g, (a, b), (0, 2))]
    return np.array([float(x) for x in out])


CASES = {  # natural parameters
    "Gompertz": [(0.08, 1e-3), (-0.05, 0.02), (1e-9, 0.1), (0.5, 1e-4)],
    "LogNormal": [(3.0, 0.5), (1.0, 1.5), (4.0, 0.2), (0.0, 0.1)],  # the last reaches z = 50 at t = 150
    "LogLogistic": [(3.0, 60.0), (0.7, 5.0), (8.0, 10.0)],
}


def run(ctx):
    load_h2()
    import semi_markov as S
    from semi_markov.base import lib, ptr_double
    make = {"Gompertz": lambda nat: S.GompertzStable("r", shape=nat[0], rate=nat[1]),
            "LogNormal": lambda nat: S.LogNormalStable("r", meanlog=nat[0], sdlog=nat[1]),
            "LogLogistic": lambda nat: S.LogLogisticStable("r", shape=nat[0], scale=nat[1])}

    def stack(risk, th, t, order=2):
        risk.set_params(np.asarray(th, float))
        out = np.empty(14)
        lib.eval_risk(risk.ptr, float(t), order, out.ctypes.data_as(ptr_double))
        return out

    rows = []
    for kind, cases in CASES.items():
        e_val, e_der, e_dt, where = 0.0, 0.0, 0.0, ""
        for nat in cases:
            risk = make[kind](nat)
            th = risk.get_params()
            st = np.array([stack(risk, th, t) for t in TIMES])  # (times, 14)
            ref = np.array([mp_stack(kind, th, t) for t in TIMES])
            for v in (0, 7):  # h and H
                e_val = max(e_val, float(np.max(np.abs(st[:, v] / ref[:, v] - 1))))
            for j in [1, 2, 3, 4, 6, 8, 9, 10, 11, 13]:  # derivatives (5 and 12 repeat 4 and 11)
                e = float(np.max(np.abs(st[:, j] - ref[:, j])) / np.max(np.abs(ref[:, j])))
                if e > e_der:
                    e_der, where = e, f"parameters {nat}, stack entry {j}"
            for t in TIMES:
                dt = 1e-6 * t
                dH = (stack(risk, th, t + dt, 0)[7] - stack(risk, th, t - dt, 0)[7]) / (2 * dt)
                e_dt = max(e_dt, abs(dH / stack(risk, th, t, 0)[0] - 1))
        note = f"{len(cases)} parameter sets x {len(TIMES)} times"
        rows += [row("risks", f"{kind}: h and H vs mpmath", e_val, "<= 1e-13", e_val <= 1e-13, note),
                 row("risks", f"{kind}: first and second derivatives vs mpmath", e_der, "<= 1e-12", e_der <= 1e-12,
                     f"{note}; worst at {where}"),
                 row("risks", f"{kind}: dH/dt vs h", float(e_dt), "<= 1e-6", e_dt <= 1e-6, note)]

    rows += model_rows(S, make)
    return rows


def model_rows(S, make):
    import test_library_general as G
    model_cases = {"Gompertz": (0.1, 0.05), "LogNormal": (1.5, 0.6), "LogLogistic": (2.5, 5.0)}
    rows = []
    for kind, nat in model_cases.items():
        m = S.SemiMarkov(kind)
        H0, A, B = S.State("H"), S.State("A"), S.State("B")
        onset, risk = S.WeibullStable("onset", k=4.0, s=10.0), make[kind](nat)
        H0.add_transition(A, onset)
        A.add_transition(B, risk)
        B.terminal = True
        for st in (H0, A, B):
            m.add_state(st)
        m.set_initial(H0)
        m.update()
        m.states = {"H": H0, "A": A, "B": B, "risks": (onset, risk)}

        th = risk.get_params()
        h_r = lambda u: float(mp_h_H(kind, th, u)[0])
        S_r = lambda u: float(mp.exp(-mp_h_H(kind, th, u)[1]))
        conv = lambda t, g: quad(lambda u: G.weibull_density(u) * g(t - u), 0, t, epsabs=0, epsrel=1e-12, limit=200)[0]
        ref_prev = np.array([conv(t, S_r) for t in G.T])
        ref_arr = np.array([conv(t, lambda v: h_r(v) * S_r(v) if v > 0 else 0.0) for t in G.T])
        m.eval_order = 0
        arr, prev = m.get_arrival(G.T, G.T0), m.get_prevalence(G.T, G.T0)
        err = max(np.max(np.abs(prev[A].value - ref_prev)) / np.max(ref_prev),
                  np.max(np.abs(arr[B].value - ref_arr)) / np.max(ref_arr))
        g_err, h_err, sym, _ = G.derivative_errors(m)
        rows += [row("risks", f"{kind} in a model: values vs convolution", float(err), "<= 1e-4", err <= 1e-4,
                     "Weibull onset, then the risk; prevalence of A, arrival in B"),
                 row("risks", f"{kind} in a model: gradient and Hessian vs differences", float(max(g_err, h_err)), "<= 1e-5",
                     max(g_err, h_err) <= 1e-5, f"gradient {g_err:.2g}, Hessian {h_err:.2g}, symmetry {sym:.2g}")]
    return rows
