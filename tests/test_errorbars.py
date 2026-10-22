"""errorbars: every target error bar in the figures is the 95% interval mu +- 1.96 SE.

The bars are read back from the drawn matplotlib axes (ErrorbarContainer segments) and compared with the targets
in model.targets (rate per 100,000; SE above 1e9 per 100,000, the infinite-SE closing cut point, gets no bar):
  - Figure 3, the fit (eval_targets(plot=True)), at the MLE
  - Figure 4, prior coverage (eval_quantiles(..., priors=True))
  - Figure 5, posterior bands (eval_quantiles(..., priors=False))
eval_quantiles runs in a sandbox output/ folder on 5 draws around the MLE. Gates, in each figure:
  - for every drawn bar, |end - (mu -+ 1.96 SE)| <= 1e-9 * max(1, 1.96 SE), at both ends;
  - the targets based on 1 to 4 events (SingularCancer.plot_suppress_events, SEER's suppression rule; events =
    (mu / SE)^2) are not drawn: point and both bar ends are NaN, which matplotlib leaves out; every other target,
    those with no events included, is drawn.
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
    """Largest scaled deviation of the drawn bars from mu +- 1.96 SE, the number of bars checked, the number of
    targets left undrawn, and whether exactly the targets with 1 to 4 events are the undrawn ones."""
    axes = plt.gcf().get_axes()
    types = list(m.targets.keys())
    stages = list(m.stage_groups.keys())
    lo_ev, hi_ev = m.plot_suppress_events
    worst, n, n_hidden, right = 0.0, 0, 0, True
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
            events = np.array([int(round((target[a][0] / target[a][1]) ** 2)) if 0.0 < target[a][1] < 1e19 else 0
                               for a in ages])
            hidden = (events >= lo_ev) & (events <= hi_ev)
            se = np.where(se < 1e9, se, 0.0)
            segs = c[2][0].get_segments()  # one (2 ends, x/y) array per bar; empty for a bar not drawn (NaN)
            lo = np.array([np.asarray(g)[:, 1].min() if np.size(g) else np.nan for g in segs])
            hi = np.array([np.asarray(g)[:, 1].max() if np.size(g) else np.nan for g in segs])
            point = np.asarray(c[0].get_ydata(), dtype=float)
            drawn = np.isfinite(point) & np.isfinite(lo) & np.isfinite(hi)
            undrawn = np.isnan(point) & np.isnan(lo) & np.isnan(hi)
            right &= bool(np.all(undrawn[hidden]) and np.all(drawn[~hidden]))
            scale = np.maximum(1.0, 1.96 * se)[~hidden]
            worst = max(worst, float(np.max(np.abs(lo[~hidden] - (mu - 1.96 * se)[~hidden]) / scale)),
                        float(np.max(np.abs(hi[~hidden] - (mu + 1.96 * se)[~hidden]) / scale)))
            n += int((~hidden).sum())
            n_hidden += int(hidden.sum())
    return worst, n, n_hidden, right


def run(ctx):
    h2 = load_h2()
    m = h2.build("model")
    mle = np.load(MLE)
    rows = []

    m.set_parameters(mle)
    m.eval_order = 0
    m.eval_targets(plot=True)
    err, n, n_hidden, right = bar_errors(m, "fit")
    rows.append(row("errorbars", "Figure 3 (fit): bars vs mu +- 1.96 SE", err, "<= 1e-9", err <= 1e-9, f"{n} bars"))
    rows.append(row("errorbars", "Figure 3 (fit): targets with 1 to 4 events not drawn, all others drawn", right,
                    "True", right, f"{n_hidden} not drawn"))

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
            err, n, n_hidden, right = bar_errors(m, layout)
            rows.append(row("errorbars", f"{label}: bars vs mu +- 1.96 SE", err, "<= 1e-9", err <= 1e-9,
                            f"{n} bars"))
            rows.append(row("errorbars", f"{label}: targets with 1 to 4 events not drawn, all others drawn", right,
                            "True", right, f"{n_hidden} not drawn"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)
    plt.close("all")
    return rows
