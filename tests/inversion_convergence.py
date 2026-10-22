"""Figure S1: accuracy of the Fourier inversion against the number of terms.

    python tests/inversion_convergence.py      # about 2 min; writes output/inversion_convergence/

The number of periods P is varied with l = 5 fixed (base.FourierOptions), so there are 2 l P = 10 P terms (k = 0 to
10 P - 1), and the Pade-type acceleration runs over 2P half-period sums. The paper's setting is P = 12 (120 terms).
The parameters are the rounded Table 2 values of verification/bladder_reference_check.R (smorfs_benchmark.THETA).

(a) Targets: the T3b machinery (tests/test_t3b_accuracy.py; background mortality off, person-time correction)
    against verification/reference_targets.csv, the 9 interval series at ages 35 to 80. Reported: the maximum
    relative error.
(b) Pointwise inversion of the arrival densities (no background mortality) of the 9 states T1-T3, D1-D3, E1-E3,
    against an independent reference: quadrature of the onset density times a matrix exponential, as
    smorfs_benchmark.reference_flows_pointwise (epsabs 0, epsrel 1e-12). Ages 0.25 to 5 by 0.25 (where the
    densities are near 0) and 6 to 80 by 1. Reported: the maximum relative and absolute error over ages 0 to 5,
    5 to 35 and 35 to 80, and the absolute error at every age against the densities themselves: near age 0 the
    truncation error rings (alternating sign) at an absolute size that exceeds the densities below about age 4.
Writes figure_S1.csv (one row per P), figure_S1_pointwise.csv (one row per P, age and state) and
figure_S1.{pdf,png}.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, VER, load_h2  # noqa: E402

PERIODS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24]
PAPER = 12
REFERENCE_ERROR = 5.9e-6  # the R reference's step-halving error, h = 1/256 against 1/512
OUT = os.path.join(ROOT, "output", "inversion_convergence")
POINT_AGES = np.concatenate([np.arange(0.25, 5.0 + 1e-9, 0.25), np.arange(6.0, 81.0)])
STATES = ["T1", "T2", "T3", "D1", "D2", "D3", "E1", "E2", "E3"]
BANDS = {"0-5": (0.0, 5.0), "5-35": (5.0, 35.0), "35-80": (35.0, 80.0)}


def reference_densities(values, ages):
    """Arrival densities at each age: integral of the Weibull onset density times exp(Q (a - u)) e_U1 times the rate."""
    from scipy.integrate import quad_vec
    from scipy.linalg import expm
    k, s = values[0], values[1]
    names = ["U1", "U2", "U3", "T1", "T2", "T3", "E1", "E2", "E3", "D1", "D2", "D3"]
    ix = {n: i for i, n in enumerate(names)}
    pairs = [("U1", "U2"), ("U1", "T1"), ("U1", "E1"), ("U2", "U3"), ("U2", "T2"), ("U2", "E2"),
             ("U3", "T3"), ("U3", "E3"), ("T1", "D1"), ("T2", "D2"), ("T3", "D3")]
    Q = np.zeros((12, 12))
    for (a, b), r in zip(pairs, values[2:]):
        Q[ix[a], ix[b]] = r
    Q -= np.diag(Q.sum(1))
    into = {b: (a, r) for (a, b), r in zip(pairs, values[2:])}  # arrival state -> (source state, rate)
    onset = lambda u: (k / s) * (u / s) ** (k - 1) * np.exp(-(u / s) ** k)
    e1 = np.zeros(12)
    e1[0] = 1.0
    out = np.empty((len(ages), len(STATES)))
    for i, a in enumerate(ages):
        p, _ = quad_vec(lambda u: onset(u) * (expm(Q.T * (a - u)) @ e1), 0.0, a, epsabs=0.0, epsrel=1e-12)
        out[i] = [into[st][1] * p[ix[into[st][0]]] for st in STATES]
    return out


def main():
    h2 = load_h2()
    import smorfs_benchmark as B
    import test_t3b_accuracy as T3b
    ref = pd.read_csv(os.path.join(VER, "reference_targets.csv"))
    ref = ref[ref["age"].isin(T3b.AGES)].reset_index(drop=True)

    t0 = time.perf_counter()
    dens_ref = reference_densities(B.VALUES, POINT_AGES)
    print(f"reference densities at {len(POINT_AGES)} ages: {time.perf_counter() - t0:.0f} s")

    m = h2.build("model")
    T3b.background_off(m, h2)
    m.set_parameters(B.THETA)
    states = {"T1": m.T1, "T2": m.T2, "T3": m.T3, "D1": m.D1, "D2": m.D2, "D3": m.D3,
              "E1": m.E1, "E2": m.E2, "E3": m.E3}
    summary, point_rows = [], []
    for P in PERIODS:
        m.options.periods = P
        den = T3b.person_time_alive(m, h2)
        err, where = T3b.max_error(m, h2, ref, den)
        m.eval_order = 0
        arr = m.get_arrival(POINT_AGES, h2.P.t0, cumulative=False)
        dens = np.column_stack([arr[states[st]].value for st in STATES])
        abs_err = np.abs(dens - dens_ref)
        rel_err = abs_err / dens_ref
        r = dict(periods=P, terms=10 * P, target_max_rel_error=err, target_worst=where)
        for band, (lo, hi) in BANDS.items():
            sel = (POINT_AGES > lo) & (POINT_AGES <= hi) if lo > 0 else (POINT_AGES <= hi)
            r[f"pointwise_max_rel_error_{band}"] = float(rel_err[sel].max())
            r[f"pointwise_max_abs_error_{band}"] = float(abs_err[sel].max())
        summary.append(r)
        for i, a in enumerate(POINT_AGES):
            for j, st in enumerate(STATES):
                point_rows.append(dict(periods=P, terms=10 * P, age=a, state=st, reference=dens_ref[i, j],
                                       smorfs=dens[i, j], abs_error=abs_err[i, j], rel_error=rel_err[i, j]))
        print(f"P = {P:2d} ({10 * P:3d} terms): targets {err:.2e} ({where}); pointwise rel 35-80 "
              f"{r['pointwise_max_rel_error_35-80']:.2e}, 5-35 {r['pointwise_max_rel_error_5-35']:.2e}, "
              f"0-5 {r['pointwise_max_rel_error_0-5']:.2e}", flush=True)
    m.options.periods = PAPER

    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(OUT, "figure_S1.csv"), index=False, float_format="%.6g")
    pts = pd.DataFrame(point_rows)
    pts.to_csv(os.path.join(OUT, "figure_S1_pointwise.csv"), index=False, float_format="%.6g")
    plot(df, pts)
    print("wrote", OUT)


def plot(df, pts):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.semilogy(df["terms"], df["target_max_rel_error"], "o-", color="k", label="Targets, ages 35-80")
    ax.axhline(REFERENCE_ERROR, color="tab:red", lw=0.8, ls=":", label="R reference's own error (step halving)")
    ax.set_title("(a) Interval targets vs R reference")
    ax = axes[1]
    for band, style in (("35-80", "o-"), ("5-35", "s--"), ("0-5", "^:")):
        ax.semilogy(df["terms"], df[f"pointwise_max_rel_error_{band}"], style, label=f"Densities, ages {band}")
    ax.set_title("(b) Pointwise arrival densities vs quadrature")
    for ax in axes[:2]:
        ax.axvline(10 * PAPER, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Number of terms (10 per period)")
        ax.set_ylabel("Maximum relative error")
        ax.grid(which="both", alpha=0.3)
        ax.legend()
    ax = axes[2]
    dens = pts[pts["periods"] == PAPER].groupby("age")["reference"]
    ax.fill_between(dens.min().index, dens.min().values, dens.max().values, color="tab:blue", alpha=0.2,
                    label="Arrival densities (range over 9 states)")
    for P, color in ((4, "tab:orange"), (8, "tab:green"), (PAPER, "k"), (24, "tab:purple")):
        sub = pts[pts["periods"] == P].groupby("age")["abs_error"].max()
        ax.semilogy(sub.index, sub.values, color=color, lw=1.2, label=f"Absolute error, {10 * P} terms")
    ax.set_ylim(1e-17, 1e-2)
    ax.set_title("(c) Absolute error by age, worst state")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density per year / absolute error")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT, f"figure_S1.{ext}"), dpi=300)


if __name__ == "__main__":
    main()
