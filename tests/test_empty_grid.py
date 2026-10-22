"""emptygrid: an empty time grid gives empty results, as v2 did (review of the speedup branch).

In its own process (with a BLAS, an empty product used to stop the process: BLAS's ldc >= max(1, m) rule), the bladder
model's get_arrival and get_prevalence with t of length 0, at orders 0, 1 and 2. Gates: the process exits 0, and every
result has value (0,), gradient (p, 0) and Hessian (p, p, 0), as the shapes of a non-empty call with n_t = 0.
"""
import os
import subprocess
import sys

from common import ROOT, row

CHILD = r'''
import sys
import numpy as np
sys.path.insert(0, "../tests")
from common import load_h2
h2 = load_h2()
m = h2.build("model")
p = len(m.get_parameters())
bad = []
for order in (0, 1, 2):
    m.eval_order = order
    for get in ("get_arrival", "get_prevalence"):
        res = getattr(m, get)(np.zeros(0), 70.0)
        for k, r in res.items():
            want = [(0,), (p, 0), (p, p, 0)][:order + 1]
            got = [np.shape(r.value), np.shape(r.gradient), np.shape(r.hessian)][:order + 1]
            if got != want:
                bad.append(f"{get} order {order} {k}: {got}")
print("RESULTS", len(res), "BAD", len(bad), "; ".join(bad[:3]))
'''


def run(ctx):
    proc = subprocess.run([sys.executable, "-c", CHILD], cwd=os.path.join(ROOT, "src"), capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"), timeout=600)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTS")), "")
    ok = proc.returncode == 0 and " BAD 0 " in line + " "
    note = line if line else (proc.stderr.strip().splitlines() or [""])[-1][:120]
    return [row("emptygrid", "empty time grid: empty results at orders 0-2, process survives",
                "ok" if ok else f"exit {proc.returncode}", "ok", ok, note)]
