"""Table 2's MLE column from the data (README).

    python tests/table2_mle.py      # about 3 s on an M4 Max

main.py's own fit, in a sandbox (common.run_refit): the targets are rebuilt from the four files in data/, and
main.py's default optimizer (saddle-free Newton with backtracking, CORRECTIONS.md entry 9) runs from the v2 estimate
output/urinary bladder.npy until the gradient norm is below newton_tol = 1e-5 (at most 100 iterations). At the
estimate the 95% intervals are exp(theta +- 1.96 SE), with the SEs from the observed information (model.limis_bounds),
in fixed point with four decimal places: the MLE column main_rates writes to CIs.csv.
Writes output/table2_mle/: urinary bladder mle.npy (the refit), table2_mle.csv (the column, the published column of
output/final/urinary bladder CIs.csv, and whether each cell agrees) and refit.log (main.py's output).
The exit status is 1 if the fit does not converge or any cell differs from the published column.
"""
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_h2, run_refit  # noqa: E402

BOUND = 100
OUT = os.path.join(ROOT, "output", "table2_mle")
PUBLISHED = os.path.join(ROOT, "output", "final", "urinary bladder CIs.csv")
COMMITTED = os.path.join(ROOT, "output", "urinary bladder mle.npy")


def fmt_4dp(x):  # as main_rates: fixed point, four digits after the decimal point, trailing zeros kept: 0.1700
    return f"{x:.4f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    d, proc, log, th = run_refit(BOUND)
    seconds = time.perf_counter() - t0
    with open(os.path.join(OUT, "refit.log"), "w") as fh:
        fh.write(proc.stdout + proc.stderr)
    shutil.rmtree(d, ignore_errors=True)
    if th is None:
        sys.exit(f"main.py saved no estimate (exit {proc.returncode}); see {os.path.join(OUT, 'refit.log')}")
    converged = any(ln.startswith("Newton") and "converged" in ln for ln in proc.stdout.splitlines())
    path = os.path.join(OUT, "urinary bladder mle.npy")
    np.save(path, th)

    h2 = load_h2()
    m = h2.build("model")
    fv, grad, _ = h2.ev(m, th, 1)
    _, _, se = m.limis_bounds(path)
    published = pd.read_csv(PUBLISHED, index_col=0)
    est, low, high = np.exp(th), np.exp(th - 1.96 * se), np.exp(th + 1.96 * se)
    table = pd.DataFrame({"mle": est, "low": low, "high": high,
                          "mle_str": [f"{fmt_4dp(a)} ({fmt_4dp(b)}, {fmt_4dp(c)})" for a, b, c in zip(est, low, high)]},
                         index=published.index)
    table["published"] = published["mle_str"]
    table["agrees"] = table["mle_str"] == table["published"]
    table.to_csv(os.path.join(OUT, "table2_mle.csv"))
    rel = np.abs(est / np.exp(np.load(COMMITTED)) - 1.0)

    print(table[["mle_str", "published", "agrees"]].to_string())
    print(f"\n{len(log) - 1} Newton iterations, {seconds:.1f} s; {'converged' if converged else 'NOT converged'}")
    print(f"objective {fv:.6f}, gradient norm {np.linalg.norm(grad):.2e}")
    print(f"largest relative difference from output/urinary bladder mle.npy: {rel.max():.1e}")
    print(f"agrees with the published MLE column: {int(table['agrees'].sum())}/{len(table)}; written to {OUT}")
    if not converged or not table["agrees"].all():
        sys.exit(1)


if __name__ == "__main__":
    main()
