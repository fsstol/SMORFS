"""recovery: on SYNTHETIC data, the MLE recovers the parameters that generated it (tools/synthetic_data.py).

Runs only in a synthetic working copy (SYNTHETIC_DATA.md at its root); elsewhere it is reported as info. main.py's
default optimizer refits from output/urinary bladder.npy (tests/common.run_refit, as tests/table2_mle.py), with
NEWTON_TOL = 1e-8 in place of main.py's newton_tol = 1e-5, so that it does not stop as soon as the gradient norm is
below the gate. It then runs until its own rule stops it: the gradient norm below 1e-8, or no step decreasing the
objective (backtracking), which happens first once the objective's changes are at round-off. On the synthetic data of
tools/synthetic_data.py it stops at iteration 8 with gradient norm 3.5e-6 (the first iterate below 1e-5 is 9.3e-6).
Gates: it converges (analytic gradient norm < 1e-5), and every parameter's z = (log estimate - log generating value) / SE, with
the SE from the observed information at the estimate, has |z| < 4. With 13 parameters and data drawn from the model,
|z| beyond 4 has probability about 1e-3 under correct estimation and SEs.
"""
import os
import shutil

import numpy as np

from common import ROOT, load_h2, row, run_refit

GENERATING = os.path.join(ROOT, "output", "urinary bladder mle generating.npy")
NEWTON_TOL = 1e-8  # the refit's stopping tolerance, far below the gate (main.py's default is 1e-5)


def run(ctx):
    if not os.path.exists(os.path.join(ROOT, "SYNTHETIC_DATA.md")):
        return [row("recovery", "parameter recovery on synthetic data", "not a synthetic working copy", "info", None)]
    d, proc, log, th = run_refit(100, newton_tol=NEWTON_TOL)
    if th is None:
        shutil.rmtree(d, ignore_errors=True)
        return [row("recovery", "refit saved an estimate", f"exit {proc.returncode}", "saved", False, proc.stderr[-200:])]
    path = os.path.join(d, "output", "urinary bladder mle.npy")
    h2 = load_h2()
    m = h2.build("model")
    _, grad, _ = h2.ev(m, th, 1)
    _, _, se = m.limis_bounds(path)
    shutil.rmtree(d, ignore_errors=True)
    gen = np.load(GENERATING)
    z = (th - gen) / se
    worst = int(np.argmax(np.abs(z)))
    gn = float(np.linalg.norm(grad))
    return [row("recovery", "refit converges (analytic gradient norm)", gn, "< 1e-5", gn < 1e-5,
                f"{len(log) - 1} Newton iterations; "
                + next((ln.split(": ", 1)[1] for ln in proc.stdout.splitlines() if ln.startswith("Newton: ")), "")),
            row("recovery", "largest abs(z) of the estimates vs the generating values", float(np.abs(z[worst])), "< 4",
                bool(np.abs(z).max() < 4), f"worst {h2.NAMES[worst]}; abs(z) > 2 for {int(np.sum(np.abs(z) > 2))} of 13")]
