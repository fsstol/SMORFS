"""Task 6, Weibull onset derivatives in the C++ library, checked below the objective.

Parameters u = log shape and v = log scale, at the Table 2 estimate. H0's only transition is the
Weibull onset, so lib.eval_time on H0 leaves the risk stack in stack_risk and returns the state stack.
  Risk stack: [h, h_u, h_v, h_uu, h_uv, h_vu, h_vv, H, H_u, H_v, H_uu, H_uv, H_vu, H_vv] against the
    closed forms below, at ages 20, 50, 70, 100 and 150.
  State stack: [S, S_u, S_v, S_uu, ..., f, f_u, ...], with f = h S. First derivatives against central
    differences of S and f; second derivatives against central differences of the analytic first
    derivatives (so they are only meaningful once the first derivatives pass). Each error is relative
    to the largest absolute value of that quantity over the five ages, because some entries cross zero.
The transform itself (quad_vec) is not tested here: against a fixed 16,000-node Gauss-Legendre rule
it agrees to <= 4.4e-8 on every entry on the inversion grid, so the errors are all in the integrand.
"""
import numpy as np

from common import TABLE2, load_h2, row

AGES = (20.0, 50.0, 70.0, 100.0, 150.0)
RISK_LABELS = ["h", "h_u", "h_v", "h_uu", "h_uv", "h_vu", "h_vv", "H", "H_u", "H_v", "H_uu", "H_uv", "H_vu", "H_vv"]
EPS = 1e-5


def closed_form(t, k, s):
    L = np.log(t / s)
    H = (t / s) ** k
    h = k / s * (t / s) ** (k - 1)
    return np.array([h, h * (1 + k * L), -k * h,
                     h * (1 + 3 * k * L + k * k * L * L), -k * h * (2 + k * L), -k * h * (2 + k * L), k * k * h,
                     H, k * L * H, -k * H,
                     k * L * H * (1 + k * L), -k * H * (1 + k * L), -k * H * (1 + k * L), k * k * H])


def run(ctx):
    h2 = load_h2()
    from semi_markov.base import lib, ptr_double  # h2 has moved the working directory to src/
    m = h2.build("model")
    th0 = np.load(TABLE2)

    def stacks(t, u, v):
        th = th0.copy()
        th[0], th[1] = u, v
        m.set_parameters(th)
        st, sr = np.zeros(14), np.zeros(14)
        lib.eval_time(m.ptr, m.H0.ptr, st.ctypes.data_as(ptr_double), sr.ctypes.data_as(ptr_double), float(t), 2)
        return st, sr

    u0, v0 = th0[0], th0[1]
    k, s = np.exp(u0), np.exp(v0)

    # Risk stack against closed forms
    risk_err = np.array([np.abs(stacks(t, u0, v0)[1] / closed_form(t, k, s) - 1.0) for t in AGES]).max(axis=0)
    worst = int(np.argmax(risk_err))
    wrong = [RISK_LABELS[i] for i in np.flatnonzero(risk_err > 1e-12)]

    # State stack against central differences. Layout: S at 0, dS at 1:3, d2S at 3:7; f at 7, df at 8:10, d2f at 10:14
    first, second = [], []
    for t in AGES:
        a = stacks(t, u0, v0)[0]
        fd1, fd2 = [], []
        for du, dv in ((EPS, 0.0), (0.0, EPS)):
            p, q = stacks(t, u0 + du, v0 + dv)[0], stacks(t, u0 - du, v0 - dv)[0]
            d = (p - q) / (2 * EPS)
            fd1.append(d[[0, 7]])                # S_x, f_x
            fd2.append(d[[1, 2, 8, 9]])          # (S_ux, S_vx, f_ux, f_vx)
        fd1 = np.array(fd1).T.ravel()            # S_u, S_v, f_u, f_v
        fd2 = np.array(fd2).T.ravel()            # S_uu, S_uv, S_vu, S_vv, f_uu, f_uv, f_vu, f_vv
        first.append((a[[1, 2, 8, 9]], fd1))
        second.append((a[[3, 4, 5, 6, 10, 11, 12, 13]], fd2))

    def rel(pairs):
        an = np.array([p[0] for p in pairs])
        fd = np.array([p[1] for p in pairs])
        return float((np.abs(an - fd) / np.abs(fd).max(axis=0)).max())

    e1, e2 = rel(first), rel(second)
    m.set_parameters(th0)
    return [
        row("weibull", "risk stack max rel error vs closed form, 5 ages", float(risk_err.max()), "<= 1e-12",
            risk_err.max() <= 1e-12, f"worst {RISK_LABELS[worst]}; wrong: {', '.join(wrong) or 'none'}"),
        row("weibull", "state stack dS, df vs central differences", e1, "<= 1e-6", e1 <= 1e-6),
        row("weibull", "state stack d2S, d2f vs differences of dS, df", e2, "<= 1e-6", e2 <= 1e-6),
    ]
