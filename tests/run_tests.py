"""Run the SMORFS regression tests and print one table of results.

    python tests/run_public_tests.py           # the public suite: no SEER data needed (synthetic data where needed)
    python tests/run_tests.py                  # every test below, on the data in data/
    python tests/run_tests.py general steps    # a subset: general steps acceleration risks quadws limis othercause
                                               # weibull T2 T3 T3b T3c cache threads emptygrid quadrature errorbars
                                               # runbox recovery
    python tests/run_tests.py --rscript        # T3 also regenerates the R reference in its sandbox

Prerequisites: bash compile.sh. The tests that build the bladder model need the three data files in data/ (SEER data,
data/README.md; or the synthetic files of tools/synthetic_data.py, which tests/run_public_tests.py uses); T3b and
Figure S1 need verification/reference_targets.csv (Rscript verification/bladder_reference_check.R, run in
verification/). Nothing is written into the repository. The exit status is 1 if any gated check fails.
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TESTS = {"general": "test_library_general", "steps": "test_steps", "acceleration": "test_acceleration",
         "risks": "test_risks", "quadws": "test_quadrature_workspace", "limis": "test_limis_toy",
         "othercause": "test_othercause", "weibull": "test_weibull", "T2": "test_t2_derivatives",
         "T3": "test_t3_accuracy", "T3b": "test_t3b_accuracy", "T3c": "test_t3c_weighting", "cache": "test_cache",
         "threads": "test_threads", "emptygrid": "test_empty_grid", "quadrature": "test_quadrature",
         "errorbars": "test_errorbars", "runbox": "test_limis_runbox", "recovery": "test_synthetic_recovery"}
SLOW = []  # run only with --slow, or when named


class Context:
    """Shared state: main.py is run at most once per hot-start setting."""
    def __init__(self, rscript):
        self.rscript = rscript
        self._runs = {}

    def main_run(self, hot_start=True):
        if hot_start not in self._runs:
            self._runs[hot_start] = common.run_main(hot_start=hot_start)
        return self._runs[hot_start]

    def cleanup(self):
        for d, _, _ in self._runs.values():
            shutil.rmtree(d, ignore_errors=True)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if 1e-3 <= abs(v) < 1e5 else f"{v:.2e}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tests", nargs="*")
    parser.add_argument("--rscript", action="store_true")
    parser.add_argument("--slow", action="store_true", help="also run the slow tests: " + ", ".join(SLOW))
    args = parser.parse_args()
    if not args.tests:
        args.tests = [t for t in TESTS if args.slow or t not in SLOW]

    ctx = Context(args.rscript)
    rows = []
    for name in args.tests:
        t_start = time.perf_counter()
        module = __import__(TESTS[name])
        try:
            rows += module.run(ctx)
        except Exception as e:  # report and carry on, so one broken test does not hide the others
            rows.append(common.row(name, "test raised", f"{type(e).__name__}: {e}", "runs", False))
        print(f"[{name} done in {time.perf_counter() - t_start:.1f} s]", file=sys.stderr)
    ctx.cleanup()

    status = {True: "PASS", False: "FAIL", None: "info"}
    widths = [10, 58, 22, 30, 6]
    header = ["test", "quantity", "value", "gate", "status"]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)) + " | note")
    print("-" * 150)
    for r in rows:
        cells = [r["test"], r["quantity"], fmt(r["value"]), r["gate"], status[r["passed"]]]
        print(" | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " | " + r["note"])
    sys.exit(1 if any(r["passed"] is False for r in rows) else 0)


if __name__ == "__main__":
    main()
