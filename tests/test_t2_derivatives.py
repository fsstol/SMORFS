"""T2, derivatives, with the logic of runA.py and runB.py (verification/h2.py).

Gradient: analytic at theta_Table2 + 0.1 (log scale) against central differences of the objective.
Hessian: analytic at theta_Table2 against central differences of an accurate gradient (analytic for
the 11 rates, finite differences for shape and scale).
Gates: every gradient component <= 1e-5; Hessian Frobenius error <= 1e-4 and positive definite.
Tasks 4 and 5 are gated on the 11x11 rate block, <= 1e-6.
The gradient check is repeated at the MLE + 0.1, and the Hessian check at the MLE against central differences of
the full analytic gradient (supplement numbers, CHANGES.md entry 40).
The Hessian gates are repeated at the optimum: since task 8, the cumulative-path MLE
(tests/cumulative_mle_reference.csv; before task 8, verification/provisional_corrected_mle_reference.csv).
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, TABLE2, load_h2, row

REFERENCE = os.path.join(ROOT, "tests", "cumulative_mle_reference.csv")  # task 8

RATES = slice(2, 13)


def rel_fro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(ctx):
    h2 = load_h2()
    m = h2.build("model")
    th0 = np.load(TABLE2)
    rows = []

    ga = np.array(h2.ev(m, th0 + 0.1, 1)[1])
    gf = h2.fd_grad(m, th0 + 0.1)
    rel = np.abs(ga - gf) / np.abs(gf)
    rows.append(row("T2", "gradient rel error, shape", rel[0], "<= 1e-5", rel[0] <= 1e-5))
    rows.append(row("T2", "gradient rel error, scale", rel[1], "<= 1e-5", rel[1] <= 1e-5))
    rows.append(row("T2", "gradient rel error, worst of 11 rates", rel[RATES].max(), "<= 1e-5", rel[RATES].max() <= 1e-5))

    Ha = np.array(h2.ev(m, th0, 2)[2])
    Hf = h2.hess_ok(m, th0)
    e_rates = rel_fro(Ha[RATES, RATES], Hf[RATES, RATES])
    e_full = rel_fro(Ha, Hf)
    eig_a = np.linalg.eigvalsh(0.5 * (Ha + Ha.T)).min()
    eig_f = np.linalg.eigvalsh(Hf).min()
    rows.append(row("T2", "Hessian rel error, 11x11 rate block", e_rates, "<= 1e-6 (tasks 4, 5)", e_rates <= 1e-6))
    rows.append(row("T2", "Hessian rel error, full 13x13", e_full, "<= 1e-4", e_full <= 1e-4))
    rows.append(row("T2", "analytic Hessian min eigenvalue", eig_a, "> 0", eig_a > 0,
                    f"accurate Hessian min eigenvalue {eig_f:.3g}"))

    th_opt = pd.read_csv(REFERENCE, comment="#").set_index("parameter").loc[h2.NAMES, "log_value"].values
    Ha = np.array(h2.ev(m, th_opt, 2)[2])
    Hf = h2.hess_ok(m, th_opt)
    e_opt = rel_fro(Ha, Hf)
    eig_a = np.linalg.eigvalsh(0.5 * (Ha + Ha.T)).min()
    eig_f = np.linalg.eigvalsh(Hf).min()
    rows.append(row("T2", "Hessian rel error, full 13x13, at the MLE", e_opt, "<= 1e-4", e_opt <= 1e-4))
    rows.append(row("T2", "analytic Hessian min eigenvalue, at the MLE", eig_a, "> 0", eig_a > 0,
                    f"accurate Hessian min eigenvalue {eig_f:.3g}"))

    # the same checks at the current MLE (supplement): gradient at theta_mle + 0.1 against central differences of
    # the objective; Hessian at theta_mle against central differences of the analytic gradient, all 13 components
    # (exact since task 6), eps = h2.eps
    ga = np.array(h2.ev(m, th_opt + 0.1, 1)[1])
    gf = h2.fd_grad(m, th_opt + 0.1)
    rel = np.abs(ga - gf) / np.abs(gf)
    worst = h2.NAMES[int(np.argmax(rel))]
    rows.append(row("T2", "gradient rel error at the MLE + 0.1, worst of 13", rel.max(), "<= 1e-5", rel.max() <= 1e-5,
                    f"worst {worst}; shape {rel[0]:.2e}, scale {rel[1]:.2e}"))
    Hg = np.empty((13, 13))
    for k in range(13):
        e = np.zeros(13)
        e[k] = h2.eps
        Hg[:, k] = (np.array(h2.ev(m, th_opt + e, 1)[1]) - np.array(h2.ev(m, th_opt - e, 1)[1])) / (2 * h2.eps)
    Hg = 0.5 * (Hg + Hg.T)
    e_g = rel_fro(Ha, Hg)
    rows.append(row("T2", "Hessian rel error at the MLE, vs differences of the analytic gradient", e_g, "<= 1e-4",
                    e_g <= 1e-4, f"min eigenvalue {np.linalg.eigvalsh(Hg).min():.3g}"))
    return rows
