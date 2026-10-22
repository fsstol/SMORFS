"""limis: SingularCancer.IMIS0, unmodified, on a toy target with a known posterior (CORRECTIONS.md, entry 8).

The model is replaced by a 3-parameter stand-in whose objective is 0.5 (x - MU)' A (x - MU), so the posterior
inside the prior box (centre 0, half-width 0.3) is, to within 1e-20, normal with mean MU and SDs SD. IMIS0 and
the Langevin step are the package's own; only eval_targets, get/set_parameters and parameter_step are replaced.
Settings: n0 = 3000, b = 300, 15 epochs, NumPy seed 1 (parameters.py's other LIMIS settings).
Gates:
  - incremental samples: each has its own log target (likelihood evaluated at the sample, not the centre)
  - log mixture densities returned by IMIS0 = an independent recomputation from the proposal components,
    log(n0/NK prior(x) + sum_k b/NK t_k(x)), max abs difference <= 1e-8
  - weighted posterior, weights target/mixture: |mean - MU| <= 0.1 SD and |sd / SD - 1| <= 0.05 per parameter
"""
import contextlib
import io
import os
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from common import ROOT, row

MU = np.array([0.02, -0.03, 0.01])
SD = np.array([0.05, 0.03, 0.08])
N0, B, EPOCHS, SEED = 3000, 300, 15, 1


class Toy:
    n_params = 3
    newton_time = 0.0
    eval_order = 0
    A = np.diag(1.0 / SD ** 2)

    def __init__(self):
        self.x = np.zeros(3)

    def get_parameters(self):
        return self.x.copy()

    def set_parameters(self, p):
        self.x = np.asarray(p, float).copy()

    def parameter_step(self, d):
        self.x = self.x + d

    def eval_targets(self):
        r = self.x - MU
        return 0.5 * r @ self.A @ r, self.A @ r, self.A


def run_toy():
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "src"))
    if os.path.join(ROOT, "src") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        import model
        saved = model.n0, model.b, model.epochs_IMIS
        model.n0, model.b, model.epochs_IMIS = N0, B, EPOCHS
        Toy.langevin_step = model.SingularCancer.langevin_step
        toy = Toy()
        np.random.seed(SEED)
        with contextlib.redirect_stdout(io.StringIO()):
            samples, log_targets, log_denom = model.SingularCancer.IMIS0(toy, np.zeros(3), np.full(3, 0.3))
        comp = getattr(toy, "limis_components", None)
        model.n0, model.b, model.epochs_IMIS = saved
    finally:
        os.chdir(cwd)
    return samples, log_targets, log_denom, comp, model


def moments(samples, w):
    w = w / w.sum()
    m = w @ samples
    return m, np.sqrt(w @ (samples - m) ** 2), 1.0 / (w @ w)


def run(ctx):
    samples, lt, ld, comp, model = run_toy()
    nk = N0 + B * EPOCHS
    inc = lt[N0:].reshape(EPOCHS, B)
    distinct = min(len(np.unique(r[np.isfinite(r)])) for r in inc)
    finite = int(np.isfinite(inc).sum(axis=1).min())

    # independent recomputation of the mixture density from the saved components
    lp = np.where(np.all(np.abs(samples) <= 0.3, axis=1), -np.sum(np.log(2 * np.full(3, 0.3))), -np.inf)
    terms = [lp + np.log(N0 / nk)]
    for loc, shape in zip(comp["loc"], comp["shape"]):
        terms.append(multivariate_t(loc, shape, comp["df"], allow_singular=True).logpdf(samples) + np.log(B / nk))
    ld_ref = logsumexp(np.array(terms), axis=0)
    denom_err = float(np.max(np.abs(ld - ld_ref)))

    lw = np.where(np.isneginf(lt), -np.inf, lt - ld)
    m, s, ess = moments(samples, np.exp(lw - lw.max()))
    mean_err = float(np.max(np.abs(m - MU) / SD))
    sd_err = float(np.max(np.abs(s / SD - 1.0)))
    rows = [
        row("limis", "incremental samples: distinct log targets per component (min)", distinct, f"= {finite} (finite)",
            distinct == finite, f"b = {B}"),
        row("limis", "log mixture density vs recomputation from components", denom_err, "<= 1e-8", denom_err <= 1e-8),
        row("limis", "posterior mean error / SD, weights target/mixture", mean_err, "<= 0.1", mean_err <= 0.1,
            f"mean {np.round(m, 4)} vs {MU}"),
        row("limis", "posterior SD rel error, weights target/mixture", sd_err, "<= 0.05", sd_err <= 0.05,
            f"sd {np.round(s, 4)} vs {SD}; ESS {ess:.0f}"),
    ]
    if hasattr(model.SingularCancer, "limis_weights"):
        m2, s2, ess2 = moments(samples, model.SingularCancer.limis_weights(lt, ld))
        e2 = float(np.max(np.abs(s2 / SD - 1.0)))
        rows.append(row("limis", "posterior SD rel error, IMIS2's weights (limis_weights)", e2, "<= 0.05", e2 <= 0.05,
                        f"sd {np.round(s2, 4)}; ESS {ess2:.0f}"))
    else:
        m2, s2, _ = moments(samples, np.exp(lt - lt.max()))
        e2 = float(np.max(np.abs(s2 / SD - 1.0)))
        rows.append(row("limis", "posterior SD rel error, IMIS2's weights (v2: target alone)", e2, "<= 0.05", e2 <= 0.05,
                        f"sd {np.round(s2, 4)}"))
    return rows
