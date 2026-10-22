"""Figures 3 to 7 and the summary table (Table 2's MLE and LIMIS columns) from the
saved LIMIS runs.

    python tests/limis_figures.py                                  # about 7 min on an M4 Max
    python tests/limis_figures.py --limis-dir output/IMIS2_wide    # wide-prior runs
    python tests/limis_figures.py --limis-dir output/IMIS2_wide --steps posterior --axes-to-bounds \
        --dest output/wide_prior      # Figures 6 and 7 of the wide-prior runs, axes at the prior bounds

Reads LIMIS_DIR/samples/ (default output/IMIS2; every run there) and output/urinary bladder mle.npy. Three steps, each main.py's or
model.py's own code, in its own process run from src/ (parameters.py is not edited; overrides are set at run time):
  1. prior coverage: main.py with run_IMIS = load_IMIS = True and IMIS2(..., priors=True), the n0 prior samples of
     every run, unweighted; eval_quantiles' prior layout. Run first, because IMIS2 writes the same histogram and
     correlogram file names for the prior and the posterior.
  2. posterior: main.py with run_IMIS = load_IMIS = True, as in parameters.py otherwise: pooled weighted quantiles
     (CIs.csv), the posterior target bands, histograms and correlogram, from limis_n_draws draws resampled by weight.
  3. MLE fit: the model at output/urinary bladder mle.npy, eval_targets(plot=True) and save_plot, with the targets
     main.py builds (verification/h2.py's build, checked against main.py's objective on the SEER data).
main.py writes into LIMIS_DIR (git-ignored); the target table is not rewritten (write_rates_table = False).
save_plot writes PDF only; here it also writes a PNG at 300 dpi. The correlogram uses DejaVu Sans (see CHILD).
The figures and the table are then copied to the committed names in output/ (for another LIMIS_DIR, to
LIMIS_DIR/figures/, so the committed figures are not overwritten):
  urinary bladder mle.{png,pdf}           Figure 3, fit at the MLE  <- step 3
  urinary bladder quant prior.{png,pdf}   Figure 4, prior coverage  <- IMIS2/urinary bladder_quantiles_prior_N
  urinary bladder quant.{png,pdf}         Figure 5, posterior bands <- IMIS2/urinary bladder_quantiles_N
  urinary bladder hist.{png,pdf}          Figure 6, histograms      <- IMIS2/urinary bladder_hist_N
  urinary bladder corr.{png,pdf}          Figure 7, correlogram     <- IMIS2/urinary bladder_corr_N
  urinary bladder CIs.csv                 summary table             <- IMIS2/CIs.csv
--steps chooses the steps (default prior,posterior,mle); --dest overrides the copy destination. With --axes-to-bounds
(SingularCancer.limis_axes_to_bounds) IMIS2 also draws the histograms and the correlogram with axes at the runs' prior
bounds (..._hist_bounds_N, ..._corr_bounds_N), and the posterior step copies only these four figures:
  urinary bladder hist.{png,pdf}              Figure 6, x-axes at the prior bounds
  urinary bladder hist autoscaled.{png,pdf}   Figure 6, autoscaled x-axes (for comparison)
  urinary bladder corr.{png,pdf}              Figure 7, autoscaled
  urinary bladder corr bounds.{png,pdf}       Figure 7 with axes at the prior bounds (not used)
Logs: LIMIS_DIR/logs/figures_{prior,posterior,mle}.log.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "output")

CHILD = r'''
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
# correlogram1 calls sns.set_theme, whose sans-serif list starts with Arial; on macOS Arial has no subscript digits
# (the labels U1->U2 etc. render as boxes), so DejaVu Sans, matplotlib's default, goes first
_set_theme = sns.set_theme
sns.set_theme = lambda *args, **kwargs: _set_theme(*args, **kwargs, rc={"font.sans-serif": ["DejaVu Sans", "Arial"]})
import model, main as M
step, limis_dir = sys.argv[1], sys.argv[2]
model.SingularCancer.limis_dir = limis_dir
model.SingularCancer.limis_axes_to_bounds = len(sys.argv) > 3 and sys.argv[3] == "1"
M.limis_min_half_width = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "none" else None
_save_plot = model.save_plot
def save_plot(name):  # PDF as parameters.save_plot, plus a PNG
    _save_plot(name)
    plt.savefig(f"../output/{name}.png", dpi=300)
model.save_plot = M.save_plot = save_plot
if step in ("prior", "posterior"):
    for mod in (model, M):
        mod.run_IMIS, mod.load_IMIS, mod.save_IMIS = True, True, False
    M.write_rates_table = False
    if step == "prior":
        _IMIS2 = M.SingularCancer.IMIS2
        M.SingularCancer.IMIS2 = lambda self, seed, priors=False, **kw: _IMIS2(self, seed, priors=True, **kw)
    print("figures:", step, "limis_n_draws =", M.SingularCancer.limis_n_draws, flush=True)
    M.main_rates_all(seed=0)
else:
    sys.path.insert(0, "../verification")
    import h2
    m = h2.build("model")
    m.set_parameters(np.load("../output/urinary bladder mle.npy"))
    m.eval_order = 0
    error, _, _ = m.eval_targets(plot=True)
    print("objective at the MLE:", error, flush=True)
    save_plot(f"{limis_dir}/urinary bladder_mle")
'''


def run(step, limis_dir, axes_to_bounds=False, min_half_width=None):
    log = os.path.join(limis_dir, "logs", f"figures_{step}.log")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        # relative to src/, the form SingularCancer.limis_dir and save_plot (which prefixes ../output/) expect
        proc = subprocess.run([sys.executable, "-c", CHILD, step, os.path.relpath(limis_dir, SRC),
                               str(int(axes_to_bounds)), "none" if min_half_width is None else repr(min_half_width)],
                              cwd=SRC, stdout=fh, stderr=subprocess.STDOUT,
                              env=env)
    print(f"{step}: exit {proc.returncode} in {(time.perf_counter() - t0) / 60:.1f} min, log {log}", flush=True)
    if proc.returncode != 0:
        sys.exit(f"{step} failed; see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limis-dir", default=os.path.join("output", "IMIS2"),
                        help="the LIMIS runs (samples/ inside), relative to the repository root or absolute")
    parser.add_argument("--steps", default="prior,posterior,mle")
    parser.add_argument("--dest", default=None, help="copy destination (default: output/, or LIMIS_DIR/figures/)")
    parser.add_argument("--axes-to-bounds", action="store_true")
    parser.add_argument("--min-half-width", type=float, default=None,
                        help="main.limis_min_half_width for the summary run (only main_rates' printed bounds use it)")
    args = parser.parse_args()
    steps = args.steps.split(",")
    limis_dir = os.path.normpath(os.path.join(ROOT, args.limis_dir))
    dest = OUT if limis_dir == os.path.join(OUT, "IMIS2") else os.path.join(limis_dir, "figures")
    if args.dest is not None:
        dest = os.path.normpath(os.path.join(ROOT, args.dest))
    os.makedirs(os.path.join(limis_dir, "logs"), exist_ok=True)
    os.makedirs(dest, exist_ok=True)

    def copy(src, dst):
        shutil.copy2(os.path.join(limis_dir, src), os.path.join(dest, dst))
        print(f"  {os.path.relpath(os.path.join(limis_dir, src), ROOT)} -> {os.path.relpath(os.path.join(dest, dst), ROOT)}")

    sys.path.insert(0, SRC)
    n = 10000  # SingularCancer.limis_n_draws
    if "prior" in steps:
        run("prior", limis_dir, args.axes_to_bounds, args.min_half_width)
        for ext in ("png", "pdf"):
            copy(f"urinary bladder_quantiles_prior_{n}.{ext}", f"urinary bladder quant prior.{ext}")
    if "posterior" in steps:
        run("posterior", limis_dir, args.axes_to_bounds, args.min_half_width)
        if args.axes_to_bounds:  # Figure 6 with axes at the prior bounds, and the autoscaled versions beside it
            for ext in ("png", "pdf"):
                copy(f"urinary bladder_hist_bounds_{n}.{ext}", f"urinary bladder hist.{ext}")
                copy(f"urinary bladder_hist_{n}.{ext}", f"urinary bladder hist autoscaled.{ext}")
                copy(f"urinary bladder_corr_{n}.{ext}", f"urinary bladder corr.{ext}")
                copy(f"urinary bladder_corr_bounds_{n}.{ext}", f"urinary bladder corr bounds.{ext}")
        else:
            for ext in ("png", "pdf"):
                copy(f"urinary bladder_quantiles_{n}.{ext}", f"urinary bladder quant.{ext}")
                copy(f"urinary bladder_hist_{n}.{ext}", f"urinary bladder hist.{ext}")
                copy(f"urinary bladder_corr_{n}.{ext}", f"urinary bladder corr.{ext}")
            copy("CIs.csv", "urinary bladder CIs.csv")
    if "mle" in steps:
        run("mle", limis_dir)
        for ext in ("png", "pdf"):
            copy(f"urinary bladder_mle.{ext}", f"urinary bladder mle.{ext}")


if __name__ == "__main__":
    main()
