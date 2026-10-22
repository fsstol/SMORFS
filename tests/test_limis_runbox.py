"""runbox: a summary of saved LIMIS runs uses the runs' own prior box, not the box the current settings give.

In a sandbox, src/limis_runs.py makes 2 tiny runs (n0 = 40, b = 20, 2 epochs) with the wide prior, max(0.3, 4 SE),
in ../output/IMIS2_box. main.py then summarises them (load_IMIS = True, limis_n_draws = 200,
limis_axes_to_bounds = True) with main.limis_min_half_width = None, so main_rates' own bounds are the +-4 SE box:
the mismatch behind an earlier error, in which a summary used main_rates' box. The histogram x-limits IMIS2 draws are recorded.
Gates:
  - the runs' box (components files) differs from main_rates' +-4 SE box, so the test can tell them apart
  - every histogram x-limit equals the runs' box, exp(centre +- half-width), relative 1e-12
  - pooling runs saved with different boxes (a third run, +-4 SE, added to the same folder) raises ValueError
"""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

from common import ROOT, make_sandbox, row

MLE = os.path.join(ROOT, "output", "urinary bladder mle.npy")
N0, B, EPOCHS, WIDE = 40, 20, 2, 0.3
BOX_DIR = "../output/IMIS2_box"

DRIVER = r'''
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt
import model, main as M
n0, box_dir = int(sys.argv[1]), sys.argv[2]
for mod in (model, M):
    mod.run_IMIS, mod.load_IMIS, mod.save_IMIS = True, True, False
    mod.n0 = n0
M.write_rates_table = False
M.limis_min_half_width = None  # main_rates' bounds: +-4 SE; the runs were sampled with the wide box
M.SingularCancer.limis_dir = box_dir
M.SingularCancer.limis_n_draws = 200
M.SingularCancer.limis_axes_to_bounds = True
_savefig = plt.savefig
def savefig(path, *args, **kwargs):
    if "_hist_bounds_" in str(path) and str(path).endswith(".png"):
        json.dump([list(ax.get_xlim()) for ax in plt.gcf().get_axes()[:13]], open("../record_xlim.json", "w"))
    return _savefig(path, *args, **kwargs)
plt.savefig = savefig
_limis_bounds = M.SingularCancer.limis_bounds
def limis_bounds(self, *args, **kwargs):
    out = _limis_bounds(self, *args, **kwargs)
    np.save("../record_main_bounds.npy", np.array([out[0], out[1]]))
    return out
M.SingularCancer.limis_bounds = limis_bounds
M.main_rates_all(seed=0)
'''


def limis_runs(src, seeds, min_half_width):
    extra = [] if min_half_width is None else ["--min-half-width", str(min_half_width)]
    return subprocess.run([sys.executable, "limis_runs.py", "--seeds", seeds, "--workers", "2", "--n0", str(N0),
                           "--b", str(B), "--epochs", str(EPOCHS), "--limis-dir", BOX_DIR] + extra,
                          cwd=src, capture_output=True, text=True, timeout=1800)


def summarise(d, src):
    for f in ("record_xlim.json", "record_main_bounds.npy"):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    return subprocess.run([sys.executable, "-c", DRIVER, str(N0), BOX_DIR], cwd=src, capture_output=True, text=True,
                          timeout=1800, env=dict(os.environ, MPLBACKEND="Agg"))


def run(ctx):
    import json
    rows = []
    d = make_sandbox(hot_start=True)
    shutil.copy2(MLE, os.path.join(d, "output"))
    src = os.path.join(d, "src")
    samples = os.path.normpath(os.path.join(src, BOX_DIR, "samples"))
    try:
        proc = limis_runs(src, "0-1", WIDE)
        if proc.returncode != 0:
            return [row("runbox", "tiny wide-prior runs", f"exit {proc.returncode}", "exit 0", False, proc.stdout[-200:])]
        comp = np.load(glob.glob(os.path.join(samples, "*_components_0.npz"))[0])
        runs_box = np.exp(np.column_stack([comp["center"] - comp["half_width"], comp["center"] + comp["half_width"]]))

        proc = summarise(d, src)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
        main_path, xlim_path = os.path.join(d, "record_main_bounds.npy"), os.path.join(d, "record_xlim.json")
        if not (os.path.exists(main_path) and os.path.exists(xlim_path)):
            return [row("runbox", "summary of the wide runs", f"exit {proc.returncode}", "exit 0", False, tail[0][:80])]
        c, hw = np.load(main_path)
        main_box = np.exp(np.column_stack([c - hw, c + hw]))
        gap = float(np.abs(main_box / runs_box - 1).max())
        rows.append(row("runbox", "runs' box differs from main_rates' (+-4 SE) box", gap, "> 0.1", gap > 0.1,
                        "largest relative difference between the two boxes"))
        xlim = np.array(json.load(open(xlim_path)))
        err = float(np.abs(xlim / runs_box - 1).max())
        rows.append(row("runbox", "summary histogram limits = runs' own box", err, "<= 1e-12", err <= 1e-12,
                        f"vs main_rates' box: {float(np.abs(xlim / main_box - 1).max()):.3g}"))

        proc = limis_runs(src, "2", None)  # a +-4 SE run in the same folder
        proc = summarise(d, src)
        raised = proc.returncode != 0 and "ValueError" in proc.stderr and "different prior bounds" in proc.stderr
        rows.append(row("runbox", "pooling runs with different boxes raises", "raises" if raised else
                        f"exit {proc.returncode}", "raises ValueError", raised))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return rows
