"""Task 10, the benchmark (Table 3): the independent R reference and the SMORFS timings, on one machine.

    python tests/benchmark.py            # about 5 min; run with nothing else on the machine

Writes into output/benchmark/:
  reference_targets.csv, benchmark_table.csv   written by verification/bladder_reference_check.R, run with that
                                               folder as its working directory (verification/ is not touched)
  R_stdout.txt                                 the R script's console output
  R_timings_median10.csv                       the R methods timed REPEATS times each (the script times them once)
  smorfs_benchmark.csv                         SMORFS timings and accuracy, one quantity per row
  table3.csv                                   benchmark_table.csv with the SMORFS row filled in
  machine.txt                                  machine, OS, Python, R and library versions, load average

SMORFS timings: SingularCancer.eval_targets as in estimation (US background survival, h2's exact targets, cumulative
interval targets), at the rounded Table 2 parameters hard-coded in the R script (smorfs_benchmark.THETA), median of
REPEATS runs after one warm-up, for the objective alone, with the gradient, and with the Hessian; the same timing
protocol as verification/smorfs_benchmark.py. Also at the corrected MLE, for information.
SMORFS accuracy: T3b (tests/test_t3b_accuracy.py), eval_targets against the reference written by this run, not
smorfs_benchmark.py's figures (CHANGES.md, entry 15: its denominator uses the H0 prevalence).
"""
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, VER, load_h2  # noqa: E402

OUT = os.path.join(ROOT, "output", "benchmark")
MLE = os.path.join(ROOT, "output", "urinary bladder mle.npy")
REPEATS = 10


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def load():
    return " ".join(f"{x:.2f}" for x in os.getloadavg())


def python_env():
    """The Python environment by name, not by path, so machine.txt holds no local paths."""
    if os.path.isdir(os.path.join(sys.prefix, "conda-meta")):
        # a conda root (base) holds envs/; a named env's prefix ends in its name
        return "conda env base" if os.path.isdir(os.path.join(sys.prefix, "envs")) else \
            f"conda env {os.path.basename(sys.prefix)}"
    return "venv" if sys.prefix != sys.base_prefix else "system Python"


def timings(m, h2, theta):
    m.set_parameters(np.asarray(theta, float))
    out = {}
    for order, label in [(0, "objective"), (1, "objective_and_gradient"), (2, "objective_gradient_hessian")]:
        m.eval_order = order
        m.eval_targets()  # warm-up
        runs = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            m.eval_targets()
            runs.append(time.perf_counter() - t0)
        out[label] = (float(np.median(runs)), float(np.min(runs)), float(np.max(runs)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    machine = {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "cores": f"{sh('sysctl -n hw.ncpu')} ({sh('sysctl -n hw.perflevel0.physicalcpu')} performance + "
                 f"{sh('sysctl -n hw.perflevel1.physicalcpu')} efficiency)",
        "memory_GB": f"{int(sh('sysctl -n hw.memsize') or 0) / 2**30:.0f}",
        "os": f"{sh('sw_vers -productName')} {sh('sw_vers -productVersion')} ({sh('sw_vers -buildVersion')}), "
              f"{platform.platform()}",
        "power": sh("pmset -g batt | head -1"),
        "python": f"{platform.python_version()} ({python_env()})",
        "numpy": np.__version__, "scipy": __import__("scipy").__version__, "pandas": pd.__version__,
        "compiler": sh("g++ --version 2>&1 | head -1"),
        "R": sh("Rscript --version 2>&1"),
        "git_commit": sh(f"git -C '{ROOT}' rev-parse --short HEAD"),
        "load_average_start": load(),
    }

    # 1. R reference and comparison methods (base R, single-threaded)
    t0 = time.perf_counter()
    proc = subprocess.run(["Rscript", os.path.join(VER, "bladder_reference_check.R")], cwd=OUT,
                          capture_output=True, text=True)
    machine["R_script_wall_seconds"] = f"{time.perf_counter() - t0:.1f}"
    with open(os.path.join(OUT, "R_stdout.txt"), "w") as fh:
        fh.write(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
    if proc.returncode != 0:
        raise RuntimeError("bladder_reference_check.R failed:\n" + proc.stderr[-2000:])
    new = pd.read_csv(os.path.join(OUT, "reference_targets.csv"))
    old = pd.read_csv(os.path.join(VER, "reference_targets.csv"))
    ref_diff = float(np.nanmax(np.abs(new.values - old.values) / np.where(old.values == 0, 1, np.abs(old.values))))
    machine["load_average_after_R"] = load()

    # 1b. The R script times each method once (system.time, 1 ms resolution). For a like-for-like comparison with
    # SMORFS's median of REPEATS, source the script (its main block runs only under Rscript) and time each method
    # REPEATS times. verification/ is only read.
    r_code = f"""
source("{os.path.join(VER, 'bladder_reference_check.R')}")
f <- list(reference = function() reference_targets(h = 1 / 256),
          cohort_month = function() cohort_targets(delta = 1 / 12),
          cohort_year = function() cohort_targets(delta = 1),
          microsim = function() micro_targets(N = 2e6))
out <- do.call(rbind, lapply(names(f), function(k) {{
  f[[k]]()
  s <- replicate({REPEATS}, {{ t0 <- proc.time()[["elapsed"]]; f[[k]](); proc.time()[["elapsed"]] - t0 }})
  data.frame(method = k, median_seconds = median(s), min_seconds = min(s), max_seconds = max(s))
}}))
write.csv(out, "R_timings_median{REPEATS}.csv", row.names = FALSE)
"""
    proc = subprocess.run(["Rscript", "-e", r_code], cwd=OUT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("R timing repeats failed:\n" + proc.stderr[-2000:])
    machine["load_average_after_R_repeats"] = load()

    # 2. SMORFS timings
    h2 = load_h2()
    import smorfs_benchmark as B
    m = h2.build("model")
    assert m.cumulative_targets
    t_theta = timings(m, h2, B.THETA)
    t_mle = timings(m, h2, np.load(MLE))
    machine["load_average_after_timings"] = load()

    # 3. SMORFS accuracy, T3b against the reference just written
    import test_t3b_accuracy as T3b
    ref = new[new["age"].isin(T3b.AGES)].reset_index(drop=True)
    ma = h2.build("model")
    T3b.background_off(ma, h2)
    ma.set_parameters(B.THETA)
    den = T3b.person_time_alive(ma, h2)
    acc, where = T3b.max_error(ma, h2, ref, den)
    ma.cumulative_targets = False
    acc_grid, where_grid = T3b.max_error(ma, h2, ref, den)

    rows = [dict(quantity="max_rel_error_vs_reference_T3b", value=acc, note=f"worst {where}"),
            dict(quantity="max_rel_error_vs_reference_T3b_grid_path", value=acc_grid,
                 note=f"v2 grid summation, for comparison; worst {where_grid}"),
            dict(quantity="regenerated_reference_vs_committed_max_rel_diff", value=ref_diff,
                 note="output/benchmark/reference_targets.csv vs verification/reference_targets.csv")]
    for where_, tt in (("table2_rounded", t_theta), ("mle", t_mle)):
        for k, (med, lo, hi) in tt.items():
            rows.append(dict(quantity=f"seconds_{k}_{where_}", value=med,
                             note=f"median of {REPEATS} (min {lo:.4f}, max {hi:.4f})"))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "smorfs_benchmark.csv"), index=False, float_format="%.6g")

    tab = pd.read_csv(os.path.join(OUT, "benchmark_table.csv"))
    i = tab.index[tab["method"].str.startswith("SMORFS")][0]
    tab.loc[i, "max_rel_error"] = acc
    tab.loc[i, "seconds"] = t_theta["objective"][0]
    tab["seconds_with_gradient"] = np.nan
    tab["seconds_with_hessian"] = np.nan
    tab.loc[i, "seconds_with_gradient"] = t_theta["objective_and_gradient"][0]
    tab.loc[i, "seconds_with_hessian"] = t_theta["objective_gradient_hessian"][0]
    tab.to_csv(os.path.join(OUT, "table3.csv"), index=False, float_format="%.6g")

    with open(os.path.join(OUT, "machine.txt"), "w") as fh:
        for k, v in machine.items():
            fh.write(f"{k}: {v}\n")

    print(tab.to_string(index=False))
    print(f"\nSMORFS accuracy (T3b): {acc:.3g} ({where}); grid path {acc_grid:.3g}")
    print(f"regenerated reference vs committed: max rel diff {ref_diff:.3g}")
    print(f"R methods, timed {REPEATS} times each:")
    print(pd.read_csv(os.path.join(OUT, f"R_timings_median{REPEATS}.csv")).to_string(index=False))
    for where_, tt in (("rounded Table 2", t_theta), ("MLE", t_mle)):
        print(f"SMORFS seconds at {where_} (median of {REPEATS}): " +
              ", ".join(f"{k} {v[0]:.4f}" for k, v in tt.items()))
    for k, v in machine.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
