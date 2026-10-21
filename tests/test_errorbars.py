"""errorbars: every target error bar in the figures is the 95% interval mu +- 1.96 SE.

The bars are read back from the drawn matplotlib axes (ErrorbarContainer segments) and compared with the targets
in model.targets (rate per 100,000; SE above 1e9 per 100,000, the infinite-SE closing cut point, gets no bar):
  - Figure 3, the fit (eval_targets(plot=True)), at the MLE
  - Figure 4, prior coverage (eval_quantiles(..., priors=True))
  - Figure 5, posterior bands (eval_quantiles(..., priors=False))
eval_quantiles runs in a sandbox output/ folder on 5 draws around the MLE. Gate: for every bar,
|end - (mu -+ 1.96 SE)| <= 1e-9 * max(1, 1.96 SE), at both ends.
"""
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.container import ErrorbarContainer  # noqa: E402
import numpy as np  # noqa: E402

from common import ROOT, load_h2, row  # noqa: E402

MLE = os.path.join(ROOT, "output", "urinary bladder mle.npy")


def bar_errors(m, layout):
    """Largest scaled deviation of the drawn bars from mu +- 1.96 SE, and the number of bars checked."""
    axes = plt.gcf().get_axes()
    types = list(m.targets.keys())
    stages = list(m.stage_groups.keys())
    worst, n = 0.0, 0
    for ax_idx, ax in enumerate(axes):
        for c in ax.containers:
            if not isinstance(c, ErrorbarContainer):
                continue
            if layout == "prior":
                tt, st = types[ax_idx // 3], stages[ax_idx % 3]
            else:
                tt, st = types[ax_idx], c.get_label().replace("Target - ", "")
            target = m.targets[tt][st]
            ages = list(target.keys())[:-1]
            mu = np.array([1e5 * target[a][0] for a in ages])
            se = np.array([1e5 * target[a][1] for a in ages])
            se = np.where(se < 1e9, se, 0.0)
            seg = np.array(c[2][0].get_segments())  # (n bars, 2 ends, x/y)
            lo, hi = seg[:, :, 1].min(axis=1), seg[:, :, 1].max(axis=1)
            scale = np.maximum(1.0, 1.96 * se)
            worst = max(worst, float(np.max(np.abs(lo - (mu - 1.96 * se)) / scale)),
                        float(np.max(np.abs(hi - (mu + 1.96 * se)) / scale)))
            n += len(mu)
    return worst, n


def run(ctx):
    h2 = load_h2()
    m = h2.build("model")
    mle = np.load(MLE)
    rows = []

    m.set_parameters(mle)
    m.eval_order = 0
    m.eval_targets(plot=True)
    err, n = bar_errors(m, "fit")
    rows.append(row("errorbars", "Figure 3 (fit): bars vs mu +- 1.96 SE", err, "<= 1e-9", err <= 1e-9, f"{n} bars"))

    draws = mle + 0.01 * np.random.default_rng(0).standard_normal((5, 13))
    d = tempfile.mkdtemp(prefix="smorfs_errorbars_")
    os.makedirs(os.path.join(d, "src"))
    os.makedirs(os.path.join(d, "output", "IMIS2"))
    cwd = os.getcwd()
    os.chdir(os.path.join(d, "src"))
    try:
        for priors, label, layout in ((True, "Figure 4 (prior coverage)", "prior"),
                                      (False, "Figure 5 (posterior bands)", "posterior")):
            m.eval_quantiles(draws.copy(), priors=priors)
            err, n = bar_errors(m, layout)
            rows.append(row("errorbars", f"{label}: bars vs mu +- 1.96 SE", err, "<= 1e-9", err <= 1e-9,
                            f"{n} bars"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)
    plt.close("all")
    return rows
