"""T3b, accuracy of the package's own interval targets: SingularCancer.eval_targets against the R reference.

T3 runs verification/smorfs_benchmark.py, which computes the targets itself and never calls
eval_targets, so it cannot tell whether eval_targets is accurate. T3b calls eval_targets directly.

Setting, chosen to match verification/bladder_reference_check.R: the rounded Table 2 parameters hard-coded
in the R script (smorfs_benchmark.THETA), background mortality off (survival_function = 1,
survival_grad = 0), ages 35 to 80, the 9 interval series (verification/reference_targets.csv; the
other-cause series is 0 there and is not compared).

eval_targets divides by the integral of S_US over the year, which is 1 year here. The R reference divides
by the model's own person-time alive, which falls below 1 by the cancer deaths (up to 2.9e-3 at age 80).
T3b therefore multiplies eval_targets' rates by 1 / integral(alive), with alive = 1 - F_lifetime from
SMORFS's cumulative inversion (16 Gauss-Legendre nodes per year). The correction is at most 2.9e-3, so a
relative error of 1e-3 in it would add at most 3e-6. It does not use the H0 prevalence, which is
inaccurate at young ages (CORRECTIONS.md, entry 7).
Gate (CORRECTIONS.md, entry 7): max relative error <= 2e-4. The grid summation of v2 (cumulative_targets = False, reported\nfor comparison) gives about 1.05e-2.
"""
import numpy as np
import pandas as pd

from common import VER, load_h2, row

AGES = np.arange(35, 81)
TYPES = {"clinical incidence": "clin_inc", "clinical mortality": "clin_mort", "preclinical mortality": "preclin_mort"}
STAGES = ["Early", "Medium", "Late"]
GATE = 2e-4


def background_off(m, h2):
    n = len(h2.P.t)
    m.survival_function = np.ones(n)
    m.survival_weights = np.cumsum(np.ones(n)) * h2.P.dt
    m.survival_grad = np.zeros(n)


def person_time_alive(m, h2):
    """Integral of the model's alive fraction over each age year, 1 - F_lifetime, Gauss-Legendre."""
    from semi_markov import LIFETIME_STATE
    x, w = np.polynomial.legendre.leggauss(16)
    nodes = (AGES[:, None] + 0.5 + 0.5 * x[None, :]).ravel()
    m.eval_order = 0
    cum = m.get_arrival(nodes, h2.P.t0, cumulative=True)
    alive = (1.0 - cum[LIFETIME_STATE].value).reshape(len(AGES), len(x))
    return 0.5 * (alive * w).sum(axis=1)


def max_error(m, h2, ref, den):
    m.eval_order = 0
    _, y = m.eval_targets(return_targets=True)
    keys = list(m.targets["clinical incidence"]["Early"].keys())
    sel = [keys.index(f"{a:02d} years") for a in AGES]
    worst, where = 0.0, ""
    for tt, short in TYPES.items():
        for st in STAGES:
            r = ref[f"{short}_{st.lower()}"].values
            rel = np.abs(y[tt][st][sel] / den - r) / r
            i = int(np.argmax(rel))
            if rel[i] > worst:
                worst, where = float(rel[i]), f"{short}_{st.lower()} at {AGES[i]}"
    return worst, where


def run(ctx):
    h2 = load_h2()
    import smorfs_benchmark as B
    ref = pd.read_csv(f"{VER}/reference_targets.csv")
    ref = ref[ref["age"].isin(AGES)].reset_index(drop=True)

    m = h2.build("model")
    background_off(m, h2)
    m.set_parameters(B.THETA)
    den = person_time_alive(m, h2)
    err, where = max_error(m, h2, ref, den)
    m.cumulative_targets = False
    err_grid, where_grid = max_error(m, h2, ref, den)
    return [
        row("T3b", "eval_targets vs R reference, max rel error", err, f"<= {GATE:g}", err <= GATE, f"worst {where}"),
        row("T3b", "same, grid path (cumulative_targets = False)", err_grid, "info", None, f"worst {where_grid}"),
        row("T3b", "person-time correction, max 1 - integral(alive)", float((1 - den).max()), "info", None),
    ]
