"""Diagnostics of the task 9 LIMIS runs (src/limis_runs.py), from the saved files only.

    python tests/limis_diagnostics.py [--samples DIR] [--out DIR]
        defaults: output/IMIS2/samples and output/IMIS2/diagnostics

For every run (seed) found in DIR it loads samples, log targets, log mixture densities and the proposal components.
Importance weights are target / mixture density. The mixture density is recomputed from the components and the prior
bounds, log(n0/NK prior(x) + sum_k b/NK t_k(x)), and compared with the saved one (they must agree once LIMIS fixes
B and C are in); the recomputed one is used. Reported:
  - per run: Kish effective sample size, ESS = 1 / sum w^2 (normalised w); the largest weight share; the share of
    samples outside the prior; posterior mean, SD and 2.5/50/97.5% quantiles per parameter (log scale)
  - pooled over runs (raw weights pooled; every run has the same NK): the same quantities
  - agreement across runs, per parameter: the SD of the per-run posterior means divided by the pooled posterior SD,
    and the largest |run median - pooled median| / pooled SD
  - posterior mass near each bound, per parameter and side: the pooled weight within 0.5 SE of the bound
    (with the 4 SE box of task 9, beyond 3.5 SE from the centre). For a normal posterior with SD = SE and that box
    it is 2.3e-4; a value above 1e-2 is flagged, and that bound should be widened and the runs repeated (CLAUDE.md,
    task 9).
The SEs are those saved with the components (key "se", since task 9b); runs saved without them (task 9) have
half-width = 4 SE, so SE = half-width / 4 there.
Writes runs.csv, parameters.csv and bounds.csv to the output folder, and prints a summary.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

NAMES = ["shape", "scale", "U1-U2", "U1-T1", "U1-D", "U2-U3", "U2-T2", "U2-D", "U3-T3", "U3-D", "T1-D", "T2-D", "T3-D"]
PREFIX = "urinary bladder"
N_SE = 4.0  # half-width in SEs, SingularCancer.limis_n_se
NEAR, FLAG = 0.5, 1e-2  # within NEAR SE of a bound; flag above FLAG


def wquantile(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return np.interp(q, (c - 0.5 * w[i]) / c[-1], x[i])


def summarise(x, w):
    w = w / w.sum()
    m = w @ x
    sd = np.sqrt(w @ (x - m) ** 2)
    q = np.array([wquantile(x[:, j], w, [0.025, 0.5, 0.975]) for j in range(x.shape[1])])
    return m, sd, q


def load_run(d, seed):
    s = np.load(os.path.join(d, f"{PREFIX}_samples_{seed}.npy"))
    lt = np.load(os.path.join(d, f"{PREFIX}_logtargets_{seed}.npy"))
    ld = np.load(os.path.join(d, f"{PREFIX}_logweights_{seed}.npy"))
    c = dict(np.load(os.path.join(d, f"{PREFIX}_components_{seed}.npz")))
    nk, n0, b = s.shape[0], int(c["n0"]), int(c["b"])
    inside = np.all(np.abs(s - c["center"]) <= c["half_width"], axis=1)
    lp = np.where(inside, -np.sum(np.log(2.0 * c["half_width"])), -np.inf)
    terms = [lp + np.log(n0 / nk)]
    for loc, shape in zip(c["loc"], c["shape"]):
        terms.append(multivariate_t(loc, shape, int(c["df"]), allow_singular=True).logpdf(s) + np.log(b / nk))
    ld_ref = logsumexp(np.array(terms), axis=0)
    ok = np.isfinite(ld) & np.isfinite(ld_ref)
    saved_vs_recomputed = float(np.max(np.abs(ld[ok] - ld_ref[ok]))) if ok.any() else np.nan
    log_w = np.where(np.isneginf(lt), -np.inf, lt - ld_ref)
    return dict(seed=seed, samples=s, log_w=log_w, comp=c, inside=inside, saved_vs_recomputed=saved_vs_recomputed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.path.join(ROOT, "output", "IMIS2", "samples"))
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "IMIS2", "diagnostics"))
    args = parser.parse_args()
    seeds = sorted(int(re.search(r"_(\d+)\.npy$", f).group(1))
                   for f in glob.glob(os.path.join(args.samples, f"{PREFIX}_samples_*.npy")))
    if not seeds:
        sys.exit(f"no runs in {args.samples}")
    runs = [load_run(args.samples, s) for s in seeds]
    c0 = runs[0]["comp"]
    for r in runs:
        assert np.array_equal(r["comp"]["center"], c0["center"]) and np.array_equal(r["comp"]["half_width"], c0["half_width"]), \
            f"seed {r['seed']}: different prior bounds"
    center, hw = c0["center"], c0["half_width"]
    # the MLE, saved since task 9c, when fixed bounds (main.limis_fixed_bounds) move the box centre away from it;
    # earlier runs were centred at the MLE
    mle = c0["mle"] if "mle" in c0 else center
    se = c0["se"] if "se" in c0 else hw / N_SE  # task 9 runs were saved without the SEs, with half-width = 4 SE

    # per run
    rows, means, medians = [], [], []
    for r in runs:
        w = np.exp(r["log_w"] - r["log_w"].max())
        w /= w.sum()
        m, sd, q = summarise(r["samples"], w)
        means.append(m)
        medians.append(q[:, 1])
        rows.append(dict(seed=r["seed"], n_samples=len(w), ess=1.0 / (w @ w), max_weight_share=w.max(),
                         share_outside_prior=1.0 - r["inside"].mean(),
                         saved_vs_recomputed_log_mixture=r["saved_vs_recomputed"]))
    runs_df = pd.DataFrame(rows)

    # pooled
    log_w = np.concatenate([r["log_w"] for r in runs])
    x = np.concatenate([r["samples"] for r in runs])
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    m, sd, q = summarise(x, w)
    ess_pooled, wmax_pooled = 1.0 / (w @ w), w.max()
    means, medians = np.array(means), np.array(medians)
    between = means.std(axis=0, ddof=1) / sd if len(runs) > 1 else np.full(13, np.nan)
    med_dev = np.max(np.abs(medians - q[:, 1]), axis=0) / sd

    par = pd.DataFrame({"parameter": NAMES, "mle": np.exp(mle), "pooled_mean_log": m, "pooled_sd_log": sd,
                        "se_log_information": se, "half_width_log": hw, "half_width_over_se": hw / se,
                        "sd_over_se": sd / se,
                        "q025": np.exp(q[:, 0]), "q50": np.exp(q[:, 1]), "q975": np.exp(q[:, 2]),
                        "between_run_sd_of_means_over_sd": between, "max_run_median_dev_over_sd": med_dev})
    to_lower, to_upper = (x - (center - hw)) / se, ((center + hw) - x) / se  # distance to each bound, in SEs
    lower = np.array([w[to_lower[:, j] <= NEAR].sum() for j in range(13)])
    upper = np.array([w[to_upper[:, j] <= NEAR].sum() for j in range(13)])
    bnd = pd.DataFrame({"parameter": NAMES, "lower_bound": np.exp(center - hw), "upper_bound": np.exp(center + hw),
                        "mass_near_lower": lower, "mass_near_upper": upper,
                        "flag": np.where((lower > FLAG) | (upper > FLAG), "WIDEN", "")})

    os.makedirs(args.out, exist_ok=True)
    runs_df.to_csv(os.path.join(args.out, "runs.csv"), index=False, float_format="%.6g")
    par.to_csv(os.path.join(args.out, "parameters.csv"), index=False, float_format="%.6g")
    bnd.to_csv(os.path.join(args.out, "bounds.csv"), index=False, float_format="%.6g")
    pd.options.display.width = 200
    print(f"{len(runs)} runs, {len(x):,} samples; pooled ESS {ess_pooled:,.0f}, largest weight share {wmax_pooled:.3g}")
    print(runs_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(par.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(bnd.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("bounds to widen:", ", ".join(bnd.loc[bnd["flag"] != "", "parameter"]) or "none")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
