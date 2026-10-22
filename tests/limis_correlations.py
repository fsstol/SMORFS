"""Posterior correlations of a set of LIMIS runs : the exits from each preclinical state,
shape-scale, and the largest correlations outside those blocks.

    python tests/limis_correlations.py --samples output/IMIS2_corrected/samples --out output/final/exit_correlations.csv

The runs are pooled as tests/limis_diagnostics.py pools them (its load_run: target / recomputed mixture density over
every sample of every run, no resampling). Correlations are weighted, on the log scale and on the natural scale.
The correlations of the 10,000 draws resampled for Figure 7, whose values the correlogram prints, are not
recomputed here: it depends on the global random state of the figure run.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limis_diagnostics import NAMES, PREFIX, load_run  # noqa: E402

EXITS = {"U1": ["U1-U2", "U1-T1", "U1-D"], "U2": ["U2-U3", "U2-T2", "U2-D"], "U3": ["U3-T3", "U3-D"]}


def weighted_corr(x, w):
    w = w / w.sum()
    m = w @ x
    c = (x - m).T @ ((x - m) * w[:, None])
    sd = np.sqrt(np.diag(c))
    return c / np.outer(sd, sd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seeds = sorted(int(re.search(r"_(\d+)\.npy$", f).group(1))
                   for f in glob.glob(os.path.join(args.samples, f"{PREFIX}_samples_*.npy")))
    runs = [load_run(args.samples, s) for s in seeds]
    log_w = np.concatenate([r["log_w"] for r in runs])
    x = np.concatenate([r["samples"] for r in runs])
    w = np.exp(log_w - log_w.max())
    ok = w > 0
    r_log = weighted_corr(x[ok], w[ok])
    r_nat = weighted_corr(np.exp(x[ok]), w[ok])
    idx = {n: i for i, n in enumerate(NAMES)}
    rows = []
    for state, exits in EXITS.items():
        for a in range(len(exits)):
            for b in range(a + 1, len(exits)):
                i, j = idx[exits[a]], idx[exits[b]]
                rows.append(dict(state=state, pair=f"{exits[a]} / {exits[b]}", corr_log_weighted=r_log[i, j],
                                 corr_natural_weighted=r_nat[i, j]))
    rows.append(dict(state="onset", pair="shape / scale", corr_log_weighted=r_log[0, 1], corr_natural_weighted=r_nat[0, 1]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(seeds)} runs, {int(ok.sum())} samples with positive weight")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    block = {(idx[a], idx[b]) for ex in EXITS.values() for a in ex for b in ex} | {(0, 1), (1, 0)}
    out = sorted(((abs(r_log[i, j]), NAMES[i], NAMES[j], r_log[i, j]) for i in range(13) for j in range(i + 1, 13)
                  if (i, j) not in block), reverse=True)[:4]
    print("largest outside the exit blocks and shape/scale (log scale):",
          ", ".join(f"{a}/{b} {v:+.3f}" for _, a, b, v in out))


if __name__ == "__main__":
    main()
