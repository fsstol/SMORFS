"""T3c, the weighting of interval targets by the US survival step function, in eval_targets.

S_US (parameters.get_survival_function) takes the life-table value at the nearest integer age, so on
the age year [a, a+1] it is L(a) on [a, a+1/2) and L(a+1) on [a+1/2, a+1). The exact interval target
is then
    (L(a) * int_a^{a+1/2} pi + L(a+1) * int_{a+1/2}^{a+1} pi) / ((L(a) + L(a+1)) / 2).
T3c computes this directly, with L read from data/US_life_table_2021.csv and each integral by
16-node Gauss-Legendre quadrature of SMORFS's pointwise arrival density, and compares it with
eval_targets at the Table 2 estimate, for the 9 interval series at ages 35 to 80.
It also checks that the grid S_US is the step function described above on every piece.
Gate: max relative error <= 1e-9. The grid summation of v2 (reported for comparison) gives about 1e-2.
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, TABLE2, load_h2, row
from test_t3b_accuracy import AGES, STAGES, TYPES

GATE = 1e-9


def run(ctx):
    h2 = load_h2()
    P = h2.P
    L = pd.read_csv(os.path.join(ROOT, "data", "US_life_table_2021.csv"), index_col="Age")["Total"] / 1e5

    S, t = np.asarray(P.get_survival_function()), P.t
    step_ok = all((S[(t >= a) & (t < a + 0.5)] == L[a]).all() and (S[(t > a + 0.5) & (t < a + 1)] == L[a + 1]).all()
                  for a in AGES)

    m = h2.build("model")
    m.set_parameters(np.load(TABLE2))
    m.eval_order = 0
    _, y = m.eval_targets(return_targets=True)
    m.cumulative_targets = False
    _, y_grid = m.eval_targets(return_targets=True)
    keys = list(m.targets["clinical incidence"]["Early"].keys())
    sel = [keys.index(f"{a:02d} years") for a in AGES]

    x, w = np.polynomial.legendre.leggauss(16)
    lo = (AGES[:, None] + 0.25 + 0.25 * x[None, :]).ravel()
    hi = lo + 0.5
    arr = m.get_arrival(np.concatenate([lo, hi]), P.t0, cumulative=False)
    La, Lb = L[AGES].values, L[AGES + 1].values
    worst, where, worst_grid = 0.0, "", 0.0
    for tt, short in TYPES.items():
        for st in STAGES:
            dens = arr[m.stage_states[tt][st]].value
            i_lo = 0.25 * (dens[:lo.size].reshape(len(AGES), -1) * w).sum(axis=1)
            i_hi = 0.25 * (dens[lo.size:].reshape(len(AGES), -1) * w).sum(axis=1)
            exact = 1e5 * (La * i_lo + Lb * i_hi) / (0.5 * (La + Lb))
            worst_grid = max(worst_grid, float(np.max(np.abs(y_grid[tt][st][sel] / exact - 1.0))))
            rel = np.abs(y[tt][st][sel] / exact - 1.0)
            i = int(np.argmax(rel))
            if rel[i] > worst:
                worst, where = float(rel[i]), f"{short}_{st.lower()} at {AGES[i]}"
    return [
        row("T3c", "grid S_US is the nearest-age step function", "yes" if step_ok else "no", "yes", step_ok),
        row("T3c", "eval_targets vs exact step-weighted integral", worst, f"<= {GATE:g}", worst <= GATE, f"worst {where}"),
        row("T3c", "same, grid path (cumulative_targets = False)", worst_grid, "info", None),
    ]
