"""Shared helpers for the SMORFS regression tests.

The tests import verification/h2.py instead of copying it, so the verification material stays the
single source of the checks. Anything that writes files (main.py writes into output/,
smorfs_benchmark.py into the repository root) runs in a temporary copy of the repository.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = os.path.join(ROOT, "verification")
TABLE2 = os.path.join(ROOT, "output", "urinary bladder.npy")
PUBLISHED = os.path.join(VER, "published_targets.csv")  # a table built from the SEER data; not in this repository
CORRECTED = os.path.join(VER, "corrected_targets.csv")  # a table built from the SEER data; not in this repository
LIBRARY = "semi_markov.dll" if sys.platform == "win32" else "semi_markov.so"  # as src/semi_markov/base.py
DATA_FILES = ["US_life_table_2021.csv",
              "single-multiple/rates_urinary_bladder.csv",
              "single-multiple/mortality_urinary_bladder.csv",
              "single-multiple/doa_urinary bladder.pkl"]


def load_h2():
    """Import verification/h2.py (it changes the working directory to src/ and imports main)."""
    if VER not in sys.path:
        sys.path.insert(0, VER)
    import h2
    return h2


def row(test, quantity, value, gate, passed, note=""):
    """One line of the results table; passed is True, False, or None for information only."""
    return dict(test=test, quantity=quantity, value=value, gate=gate, passed=passed, note=note)


def build(h2, corrected=True):
    """h2.build("model") with the corrected targets (main.py's default since v1.0.1, CORRECTIONS.md entry 16) or, corrected=False, the
    original analysis's (uncorrected denominators). main.corrected_denominators is restored afterwards."""
    saved = h2.M.corrected_denominators
    h2.M.corrected_denominators = corrected
    try:
        return h2.build("model")
    finally:
        h2.M.corrected_denominators = saved


def make_sandbox(hot_start=True, verification=False):
    """A temporary copy of src/, the four data files and the library, with an empty output/."""
    d = tempfile.mkdtemp(prefix="smorfs_test_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"),
                    ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    for f in DATA_FILES:
        os.makedirs(os.path.dirname(os.path.join(d, "data", f)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, "data", f), os.path.join(d, "data", f))
    shutil.copy2(os.path.join(ROOT, LIBRARY), d)
    if sys.platform == "win32":  # a BLAS's DLLs next to the library, where Windows looks for its dependencies
        for f in os.listdir(ROOT):
            if f.lower().endswith(".dll") and f != LIBRARY:
                shutil.copy2(os.path.join(ROOT, f), d)
    os.makedirs(os.path.join(d, "output"))
    if hot_start:
        shutil.copy2(TABLE2, os.path.join(d, "output"))
    if verification:
        shutil.copytree(VER, os.path.join(d, "verification"),
                        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    return d


# Runs main.py's entry point in a sandbox, with LIMIS switched off for the test only (parameters.py is
# not edited). At main_rates' first evaluation it records the starting parameters and objective, the
# number of targets that enter the objective, and (independently of the hot start) the objective of
# main.py's own targets at the Table 2 estimate.
MAIN_DRIVER = r'''
import json
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import sys
import main as M
M.run_IMIS = False
if len(sys.argv) > 1 and sys.argv[1] == "uncorrected":
    M.corrected_denominators = False
record = {}
_eval_targets = M.SingularCancer.eval_targets
def eval_targets(self, *args, **kwargs):
    out = _eval_targets(self, *args, **kwargs)
    if "error" not in record:
        record["params"] = [float(x) for x in self.get_parameters()]
        record["error"] = float(out[0])
        # eval_targets sums target i for i < len(ages) - 1 when its age is in ages_eval
        record["n_targets"] = sum(1 for series in self.targets.values() for target in series.values()
                                  for a in list(target)[:-1] if a in M.ages_eval)
        table2 = "../output/urinary bladder.npy"
        if os.path.exists(table2):
            params, order = self.get_parameters().copy(), self.eval_order
            self.set_parameters(np.load(table2))
            self.eval_order = 0
            record["error_table2"] = float(_eval_targets(self)[0])
            self.set_parameters(params)
            self.eval_order = order
        with open("../record.json", "w") as fh:
            json.dump(record, fh)
    return out
M.SingularCancer.eval_targets = eval_targets
M.main_rates_all(seed=0)
'''


def run_main(hot_start=True, corrected=True):
    """Run main.py in a sandbox, with the corrected targets or (corrected=False) the original analysis's. Returns
    (sandbox dir, completed process, record or None)."""
    d = make_sandbox(hot_start=hot_start)
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "-c", MAIN_DRIVER] + ([] if corrected else ["uncorrected"]),
                          cwd=os.path.join(d, "src"),
                          env=env, capture_output=True, text=True, timeout=1800)
    rec_path = os.path.join(d, "record.json")
    record = None
    if os.path.exists(rec_path):
        with open(rec_path) as fh:
            record = json.load(fh)
    return d, proc, record


# Runs main.py's own optimizer (the saddle-free Newton loop in main_rates) in a sandbox, from the Table 2
# estimate. Overridden here, for the test only (parameters.py is not edited): run_IMIS = False,
# hot_start_save = True (so main_rates saves its best estimate to ../output/urinary bladder mle.npy in the
# sandbox), epochs_hess and newton_damped (v2's 0.25/sqrt(n) schedule when 1; main.py's default, full steps
# with backtracking, when 0). Every order-2 evaluation is logged: objective, gradient norm, parameters.
REFIT_DRIVER = r'''
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import main as M
M.run_IMIS = False
M.hot_start_save = True
M.epochs_hess = int(sys.argv[1])
M.newton_damped = bool(int(sys.argv[2]))
for arg in sys.argv[3:]:
    if arg == "uncorrected":
        M.corrected_denominators = False
    elif arg.startswith("newton_tol="):
        M.newton_tol = float(arg.split("=", 1)[1])
log = open("../refit_log.jsonl", "w")
_eval_targets = M.SingularCancer.eval_targets
def eval_targets(self, *args, **kwargs):
    out = _eval_targets(self, *args, **kwargs)
    if self.eval_order == 2:
        log.write(json.dumps({"f": float(out[0]), "g": float(np.linalg.norm(out[1])),
                              "th": [float(x) for x in self.get_parameters()]}) + "\n")
        log.flush()
    return out
M.SingularCancer.eval_targets = eval_targets
M.main_rates_all(seed=0)
'''


def run_refit(epochs, damped=False, corrected=True, newton_tol=None):
    """Run main.py's refit in a sandbox, with the corrected targets or (corrected=False) the original analysis's, and
    main.newton_tol replaced by newton_tol when it is given. Returns (sandbox dir, completed process, log, saved
    estimate or None)."""
    d = make_sandbox(hot_start=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    extra = ([] if corrected else ["uncorrected"]) + ([] if newton_tol is None else [f"newton_tol={newton_tol!r}"])
    proc = subprocess.run([sys.executable, "-c", REFIT_DRIVER, str(epochs), str(int(damped))] + extra,
                          cwd=os.path.join(d, "src"),
                          env=env, capture_output=True, text=True, timeout=14400)
    log = []
    if os.path.exists(os.path.join(d, "refit_log.jsonl")):
        with open(os.path.join(d, "refit_log.jsonl")) as fh:
            log = [json.loads(line) for line in fh]
    saved = os.path.join(d, "output", "urinary bladder mle.npy")
    return d, proc, log, (np.load(saved) if os.path.exists(saved) else None)


def newton_analytic(h2, m, th, iters=20, tol=1e-5):
    """Saddle-free Newton with backtracking, as h2.newton, but with the analytic gradient and Hessian for
    all 13 parameters (h2.newton uses finite differences for shape and scale). Returns the estimate, the
    objective, the analytic gradient norm and Hessian there, and the objective path."""
    th = np.asarray(th, float)
    fv, g, H = h2.ev(m, th, 2)
    hist = [fv]
    for _ in range(iters):
        g, H = np.asarray(g), np.asarray(H)
        if np.linalg.norm(g) < tol:
            break
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        step = -V @ ((V.T @ g) / np.abs(w))
        a = 1.0
        while h2.f(m, th + a * step) > fv and a > 1e-4:
            a *= 0.5
        th = th + a * step
        fv, g, H = h2.ev(m, th, 2)
        hist.append(fv)
    return th, fv, float(np.linalg.norm(g)), np.asarray(H), hist


def write_mle_reference(path, h2, th, fv, gn, H, header):
    """Write an MLE reference in the format of tests/cumulative_mle_reference.csv."""
    import pandas as pd
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    df = pd.DataFrame({"parameter": h2.NAMES, "log_value": th, "value": np.exp(th), "se_log": se,
                       "ci_low": np.exp(th - 1.96 * se), "ci_high": np.exp(th + 1.96 * se)})
    with open(path, "w") as fh:
        fh.write(f"# {header}\n# objective={fv:.10f} grad_norm={gn:.3e} (analytic gradient)\n")
        df.to_csv(fh, index=False, float_format="%.12g")
