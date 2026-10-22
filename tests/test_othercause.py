"""Task 5, other-cause mortality target (B4): the model value is a rate per living person.

At ages 35, 50, 65 and 80 and for each stage, the target eval_targets returns must equal
lambda_US(t) * p_s(t), with lambda_US = -S_US'/S_US as the code builds it (smoothed derivative over
the life-table survival) and p_s from get_prevalence. Before the fix the ratio is S_US(t).
Consistency of the gradient and Hessian with the value is checked by T2.
"""
import numpy as np

from common import TABLE2, load_h2, row

AGES = (35, 50, 65, 80)
TARGET = "preclinical other mortality"


def run(ctx):
    h2 = load_h2()
    P = h2.P
    m = h2.build("model")
    th0 = np.load(TABLE2)
    m.set_parameters(th0)
    m.eval_order = 0
    _, y = m.eval_targets(return_targets=True)
    prev = m.get_prevalence(P.t, P.t0, prefix="Prevalence")
    lam = m.survival_grad / m.survival_function

    worst = 0.0
    ratios = []
    for stage, state in m.stage_states[TARGET].items():
        ages = list(m.targets[TARGET][stage].keys())
        for a in AGES:
            i = ages.index(f"{a:02d} years")
            g = int(np.argmin(np.abs(P.t - P.age_groups[f"{a:02d} years"][0])))
            ratio = (y[TARGET][stage][i] / 1e5) / (lam[g] * prev[state].value[g])
            worst = max(worst, abs(ratio - 1.0))
            if stage == "Early":
                ratios.append(f"{a}: {ratio:.3f} (S_US {m.survival_function[g]:.3f})")
    return [row("othercause", "max |model / (lambda_US p_s) - 1|, 4 ages x 3 stages", worst, "<= 1e-12",
                worst <= 1e-12, "Early ratios " + ", ".join(ratios))]
