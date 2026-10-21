"""The public test suite: needs no SEER data.

    python tests/run_public_tests.py           # about 3 minutes; exit status 1 if any gated check fails

1. Tests that need no data, run in this repository: general, steps, acceleration, risks, quadws, limis.
2. Tests that build the bladder model, run on SYNTHETIC data (tools/synthetic_data.py) in a temporary synthetic working
   copy: othercause, weibull, T2, T3c, cache, threads, emptygrid, quadrature, errorbars, runbox, recovery; and, when
   Rscript is available, T3 and T3b, after the R reference (verification/bladder_reference_check.R) is regenerated
   there. Their gates are internal (derivatives against finite differences, caches, quadrature, the R reference at
   given parameters, parameter recovery), so they hold on synthetic data.
Tests of the paper's own results need the SEER data and references computed from it, and are not part of this
repository: the targets against the published tables (T1), the hot start's objective, the MLE against its reference
(T4, T4b, T4c, T4d), the LIMIS summary table against the reference SEs (summary), and the golden outputs (golden).
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NO_DATA = ["general", "steps", "acceleration", "risks", "quadws", "limis"]
SYNTHETIC = ["othercause", "weibull", "T2", "T3c", "cache", "threads", "emptygrid", "quadrature", "errorbars", "runbox",
             "recovery"]
WITH_R = ["T3", "T3b"]


def main():
    status = 0
    print("== 1. tests that need no data", flush=True)
    status |= subprocess.run([sys.executable, os.path.join(ROOT, "tests", "run_tests.py")] + NO_DATA, cwd=ROOT).returncode
    work = tempfile.mkdtemp(prefix="smorfs_synthetic_")
    try:
        workdir = os.path.join(work, "synthetic_run")
        print("\n== 2. tests on SYNTHETIC data, in", workdir, flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "synthetic_data.py"), "--workdir", workdir], check=True)
        tests = list(SYNTHETIC)
        if shutil.which("Rscript"):
            r = subprocess.run(["Rscript", "bladder_reference_check.R"], cwd=os.path.join(workdir, "verification"),
                               capture_output=True, text=True)
            if r.returncode == 0:
                tests += WITH_R
            else:
                print("R reference failed; T3 and T3b skipped:", r.stderr[-300:])
        else:
            print("Rscript not found: T3 and T3b skipped")
        status |= subprocess.run([sys.executable, os.path.join(workdir, "tests", "run_tests.py")] + tests,
                                 cwd=workdir).returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("\npublic suite:", "all gated checks passed" if status == 0 else "FAILED (see the tables above)")
    sys.exit(1 if status else 0)


if __name__ == "__main__":
    main()
