"""The shared adaptive quadrature of the general states' transforms (branch speedup, entries 11 and 19).

H0's stack (value, gradient, Hessian) at the 120 frequencies of the inversion, at the parameter sets of SPEEDUP.md
entry 11: the Table 2 estimate, the MLE, 5 draws within +-1 of the MLE on the log scale, and Weibull shapes x0.14 to
x7.39 (k = 1.0 to 55) and scales x0.5 to x2. Three methods: quad_vec_multi (one subdivision for all frequencies, used
by evaluate), quad_vec per frequency (v2's method), and Gauss-Kronrod 21 on 2,000 equal subintervals of [0.1, 200] as the
reference, graded geometrically towards t = 0 (60 subintervals from 1e-14 to 0.1), where the derivatives with respect
to log shape contain log(t) terms. Errors are relative to each frequency's largest |transform| entry, quad_vec's own tolerance basis.
Gate: the shared quadrature within 1e-8 of the reference everywhere (v2's per-frequency quadrature is up to 1.8e-8).
"""
import re

import numpy as np

from common import ROOT, load_h2, row

N_REF = 2000
MAX_TIME = 200.0


def run(ctx):
    h2 = load_h2()
    import smorfs_benchmark as B
    from semi_markov.base import lib, ptr_complex, ptr_double
    src = open(f"{ROOT}/src/src_lib/quad_vec.cpp").read()
    arr = lambda name: np.array([float(x) for x in re.search(name + r"\[\] = \{(.*?)\};", src, re.S).group(1).split(",")])
    X, V = arr("double X"), arr("double V")
    m = h2.build("model")
    o = m.options
    terms = 2 * o.l * o.periods
    g_re, dg = o.A / (2 * o.l * h2.P.t0), np.pi / (o.l * h2.P.t0)
    gammas = g_re + 1j * dg * np.arange(terms)
    n = 14  # H0's stack at order 2: (1 + 1 transition) x (1 + 2 + 4)
    st, sr = np.empty(n), np.empty(14)

    def reference():
        # graded towards t = 0, where the log-shape derivatives carry log(t) terms (integrable, but singular)
        edges = np.concatenate([[0.0], np.logspace(-14, -1, 60), np.linspace(0.1, MAX_TIME, N_REF + 1)[1:]])
        c, h = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        nodes = (c[:, None] + h[:, None] * X[None, :]).ravel()
        F = np.empty((len(nodes), n))
        for i, t in enumerate(nodes):
            lib.eval_time(m.ptr, m.H0.ptr, st.ctypes.data_as(ptr_double), sr.ctypes.data_as(ptr_double), float(t), 2)
            F[i] = st
        w = (h[:, None] * V[None, :]).ravel()
        return (np.exp(-np.outer(gammas, nodes)) * w) @ F

    mle = np.load(f"{ROOT}/output/urinary bladder mle.npy")
    sets = [("Table 2", np.asarray(B.THETA, float)), ("MLE", mle)]
    rng = np.random.default_rng(0)
    sets += [(f"MLE +- 1, draw {i}", mle + rng.uniform(-1, 1, mle.shape)) for i in range(5)]
    for dk, ds in ((-1.386, 0), (1.386, 0), (0, -0.693), (0, 0.693), (1.386, -0.693), (-1.386, 0.693), (2.0, 0), (-2.0, 0)):
        th = mle.copy()
        th[0] += dk
        th[1] += ds
        sets.append((f"shape x{np.exp(dk):.2f} scale x{np.exp(ds):.2f}", th))

    worst_shared, worst_single, where = 0.0, 0.0, ""
    for name, th in sets:
        m.set_parameters(th)
        m.eval_order = 2
        ref = reference()
        scale = np.abs(ref).max(axis=1, keepdims=True)
        shared = np.empty((terms, n), dtype=complex)
        lib.quad_vec_multi_state(m.ptr, m.H0.ptr, shared.ctypes.data_as(ptr_complex), g_re, dg, terms, MAX_TIME, 2)
        single = np.empty((terms, n), dtype=complex)
        for k, g in enumerate(gammas):
            lib.quad_vec(m.ptr, m.H0.ptr, single[k].ctypes.data_as(ptr_complex), st.ctypes.data_as(ptr_double),
                         sr.ctypes.data_as(ptr_double), g.real, g.imag, MAX_TIME, 2)
        e_sh = float(np.max(np.abs(shared - ref) / scale))
        e_si = float(np.max(np.abs(single - ref) / scale))
        if e_sh > worst_shared:
            worst_shared, where = e_sh, name
        worst_single = max(worst_single, e_si)
    note = f"{len(sets)} parameter sets; worst at {where}; per-frequency quad_vec (v2) worst {worst_single:.2g}"
    return [row("quadrature", "shared subdivision vs fine reference", worst_shared, "<= 1e-8", worst_shared <= 1e-8, note)]
