"""Two sets of LIMIS runs compared: pooled posterior medians and 95% credible limits, in posterior-SD units.

    python tests/compare_limis_posteriors.py --runs output/IMIS2_weibull/samples --reference output/IMIS2_wide/samples \
        --labels weibull,wide --out output/weibull_prior/posterior_weibull_vs_wide.csv

Each set is pooled as tests/limis_diagnostics.py pools it (its load_run: target / recomputed mixture density over every
sample of every run, no resampling) and summarised with its summarise (weighted mean, SD and 2.5%, 50% and 97.5%
quantiles, log scale). Per parameter: both posterior SDs (log scale) and their ratio, and both sets' quantiles (natural
scale) with their difference on the log scale divided by the reference set's posterior SD (column q.._diff_sd; the
layout used for the paper's comparison of the wide and the +-4 SE priors).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402
from limis_diagnostics import NAMES, PREFIX, load_run, summarise  # noqa: E402


def pooled(d):
    seeds = sorted(int(re.search(r"_(\d+)\.npy$", f).group(1))
                   for f in glob.glob(os.path.join(d, f"{PREFIX}_samples_*.npy")))
    if not seeds:
        sys.exit(f"no runs in {d}")
    runs = [load_run(d, s) for s in seeds]
    log_w = np.concatenate([r["log_w"] for r in runs])
    x = np.concatenate([r["samples"] for r in runs])
    w = np.exp(log_w - log_w.max())
    _, sd, q = summarise(x, w / w.sum())
    return len(seeds), sd, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="samples/ folder of the runs compared")
    parser.add_argument("--reference", required=True, help="samples/ folder of the reference runs (the SD unit)")
    parser.add_argument("--labels", default="runs,reference")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    a, b = args.labels.split(",")
    na, sd_a, q_a = pooled(os.path.join(ROOT, args.runs))
    nb, sd_b, q_b = pooled(os.path.join(ROOT, args.reference))
    df = pd.DataFrame({"parameter": NAMES, f"sd_{a}": sd_a, f"sd_{b}": sd_b, "sd_ratio": sd_a / sd_b})
    for k, name in enumerate(["q025", "q50", "q975"]):
        df[f"{name}_{a}"], df[f"{name}_{b}"] = np.exp(q_a[:, k]), np.exp(q_b[:, k])
        df[f"{name}_diff_sd"] = (q_a[:, k] - q_b[:, k]) / sd_b
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    df.to_csv(os.path.join(ROOT, args.out), index=False, float_format="%.6g")
    print(f"{a}: {na} runs; {b}: {nb} runs (reference); differences in units of {b}'s posterior SD, log scale")
    print(df[["parameter", "sd_ratio", "q025_diff_sd", "q50_diff_sd", "q975_diff_sd"]].to_string(index=False))
    for name in ["q025", "q50", "q975"]:
        j = int(np.argmax(np.abs(df[f"{name}_diff_sd"])))
        print(f"largest |{name} difference|: {abs(df[f'{name}_diff_sd'][j]):.4f} SD ({NAMES[j]})")


if __name__ == "__main__":
    main()
