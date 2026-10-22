"""T3, accuracy: verification/smorfs_benchmark.py against the R reference.

The benchmark writes into the repository root, so it runs in a sandbox copy. By default it uses the
committed verification/reference_targets.csv. With --rscript, bladder_reference_check.R first
regenerates reference_targets.csv in the sandbox, and the benchmark uses that copy instead.
No gate before task 8 (cumulative path <= 2e-4). The values in CLAUDE.md are given in the notes.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from common import VER, make_sandbox, row

EXPECTED = {"max_rel_error_vs_reference": 1.05e-2,
            "max_rel_error_vs_reference_cumulative_inversion": 1.02e-4,
            "max_rel_error_pointwise_inversion": 1.7e-8}


def run(ctx):
    d = make_sandbox(hot_start=False, verification=True)
    rows = []
    if ctx.rscript:
        proc = subprocess.run(["Rscript", "verification/bladder_reference_check.R"], cwd=d,
                              capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            return [row("T3", "bladder_reference_check.R", "failed", "runs", False, proc.stderr[-300:])]
        new = pd.read_csv(os.path.join(d, "reference_targets.csv"))
        old = pd.read_csv(os.path.join(VER, "reference_targets.csv"))
        diff = float(np.nanmax(np.abs(new.values - old.values) / np.where(old.values == 0, 1, np.abs(old.values))))
        rows.append(row("T3", "regenerated reference vs committed (max rel diff)", diff, "info", None))
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "verification/smorfs_benchmark.py"], cwd=d, env=env,
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return rows + [row("T3", "smorfs_benchmark.py", "failed", "runs", False, proc.stderr[-300:])]
    res = pd.read_csv(os.path.join(d, "smorfs_benchmark.csv")).set_index("quantity")["value"]
    for q, expected in EXPECTED.items():
        rows.append(row("T3", q, float(res[q]), "info", None, f"CLAUDE.md: {expected:.3g}"))
    for q in ["seconds_objective", "seconds_objective_and_gradient", "seconds_objective_gradient_hessian"]:
        rows.append(row("T3", q, float(res[q]), "info", None))
    return rows
