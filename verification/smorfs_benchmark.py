"""
SMORFS row of Table 3 of the paper: accuracy against the independent R reference, and timings.

Run from the repository root (the folder that contains src/), after
  1. building the library:   bash compile.sh
  2. placing the data:        data/US_life_table_2021.csv
                              data/single-multiple/rates_urinary_bladder.csv
                              data/single-multiple/mortality_urinary_bladder.csv
                              data/single-multiple/doa_urinary bladder.pkl   (note the space)
  3. optionally putting reference_targets.csv (written by bladder_reference_check.R) in the root.

    python smorfs_benchmark.py

Writes smorfs_targets.csv and smorfs_targets_cumulative.csv (same format as reference_targets.csv)
and smorfs_benchmark.csv, and,
if reference_targets.csv is present, prints the maximum relative error against it.

Definitions, chosen to match bladder_reference_check.R exactly:
  * Parameters are the rounded Table 2 values hard-coded in the R script.
  * Background mortality is off; a rate is arrivals over the interval divided by the person-time
    of survivors in the model, integral(pi_s) / integral(S_model), as in Section 2.5 of the paper.
  * Targets use SMORFS's own arrival and prevalence functions and the same summation rule as
    SingularCancer.eval_targets (left sums over the time grid t, interval ends at the nearest grid point).
  * Timings use eval_targets exactly as in estimation (US background survival, exact targets),
    median of REPEATS runs, for the objective alone, with the gradient, and with the Hessian.
The source code is not modified.
"""
import os, sys, time, platform
import numpy as np, pandas as pd

def _repo_root():
    """Find the repository root (the folder containing src/model.py), wherever this script lives."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(d, "src", "model.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot find src/model.py: run this from inside the repository.")
ROOT = _repo_root()
VER = os.path.join(ROOT, "verification")
def vfile(name):
    """A verification input, from verification/ if present, else from the repository root."""
    p = os.path.join(VER, name)
    return p if os.path.exists(p) else os.path.join(ROOT, name)
os.chdir(os.path.join(ROOT, "src")); sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import expm
from scipy.integrate import quad_vec
import main as M                      # star-imports parameters and model
from model import SingularCancer

REPEATS = 10
VALUES = np.array([7.163, 139.1, 0.410, 2.531, 0.012, 1.783, 1.932, 0.049,
                   0.966, 0.148, 0.010, 0.067, 0.142])           # rounded Table 2, R-script order
THETA = np.log(VALUES)
AGES = list(range(35, 81))
TYPES = {"clinical incidence": "clin_inc", "clinical mortality": "clin_mort", "preclinical mortality": "preclin_mort"}


def smorfs_targets(m, t, dt):
    """Interval rates per 100,000 from SMORFS arrival and prevalence functions on grid t."""
    m.set_parameters(THETA); m.eval_order = 0
    arr = m.get_arrival(t, M.t0, prefix="Arrival", cumulative=False)
    prev = m.get_prevalence(t, M.t0, prefix="Prevalence")
    alive = sum(prev[s].value for s in (m.H0, m.U1, m.U2, m.U3, m.T1, m.T2, m.T3))
    idx = [int(np.argmin(np.abs(t - a))) for a in AGES + [AGES[-1] + 1]]
    den = (np.cumsum(alive) * dt)[idx]
    out = pd.DataFrame({"age": AGES})
    for ttype, short in TYPES.items():
        for stage, state in m.stage_states[ttype].items():
            num = (np.cumsum(arr[state].value) * dt)[idx]
            out[f"{short}_{stage.lower()}"] = 1e5 * np.diff(num) / np.diff(den)
    return out, arr, alive


def smorfs_targets_cumulative(m):
    """Same rates, but with numerators from SMORFS's cumulative inversion (Pi_s(omega)/omega) at the exact
    age cut points, as Section 2.5.1 of the paper describes; no summation over a grid. Denominator: model
    survival integrated by the trapezoid rule on a 1/64-year grid aligned with integer ages."""
    m.set_parameters(THETA); m.eval_order = 0
    edges = np.array(AGES + [AGES[-1] + 1], float)
    cum = m.get_arrival(edges, M.t0, prefix="ArrivalCum", cumulative=True)
    fine = np.arange(0.0, edges[-1] + 1 + 1e-12, 1 / 64)
    prev = m.get_prevalence(fine, M.t0, prefix="Prevalence")
    alive = sum(prev[s].value for s in (m.H0, m.U1, m.U2, m.U3, m.T1, m.T2, m.T3))
    C = np.concatenate([[0.0], np.cumsum((alive[1:] + alive[:-1]) / 2) / 64])
    den = np.diff(C[[int(round(e * 64)) for e in edges]])
    out = pd.DataFrame({"age": AGES})
    for ttype, short in TYPES.items():
        for stage, state in m.stage_states[ttype].items():
            out[f"{short}_{stage.lower()}"] = 1e5 * np.diff(cum[state].value) / den
    return out


def exact_targets(stage_groups):
    """Calibration targets from the aggregated SEER files, as in main.main_rates but without halving Population."""
    df = pd.read_csv("../data/single-multiple/rates_urinary_bladder.csv")
    df = df.loc[df[M.diag_col] != "2000-2021"].copy()
    mo = pd.read_csv("../data/single-multiple/mortality_urinary_bladder.csv")
    mo = mo.loc[mo["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)"]
    doa = pd.read_pickle("../data/single-multiple/doa_urinary bladder.pkl").reset_index(drop=False)
    df[M.stage_col] = df[M.stage_col].fillna("Blank(s)")
    df = df.loc[~df[M.stage_col].isin(["Blank(s)", "Unknown/unstaged"])]
    tg = {k: {} for k in M.target_names}
    for name, stages in stage_groups.items():
        a, b, c = df[df[M.stage_col].isin(stages)], mo[mo[M.stage_col].isin(stages)], doa[doa[M.stage_col].isin(stages)]
        tg["clinical incidence"][name] = M.get_target_incidence(a)
        tg["clinical mortality"][name] = M.get_target_mortality(b)
        tg["preclinical mortality"][name] = M.get_target_doa(c.loc[c["outcome"] == 1])
        tg["preclinical other mortality"][name] = M.get_target_doa(c.loc[c["outcome"] == 2])
    return tg


def reference_flows_pointwise(ages):
    """Independent pointwise check of the inversion: post-onset occupancies by quadrature of a matrix exponential."""
    k, s = VALUES[0], VALUES[1]
    names = ["U1", "U2", "U3", "T1", "T2", "T3", "E1", "E2", "E3", "D1", "D2", "D3"]
    ix = {n: i for i, n in enumerate(names)}
    Q = np.zeros((12, 12))
    for (a, b), r in zip([("U1","U2"),("U1","T1"),("U1","E1"),("U2","U3"),("U2","T2"),("U2","E2"),
                          ("U3","T3"),("U3","E3"),("T1","D1"),("T2","D2"),("T3","D3")], VALUES[2:]):
        Q[ix[a], ix[b]] = r
    Q -= np.diag(Q.sum(1))
    onset = lambda u: (k / s) * (u / s) ** (k - 1) * np.exp(-(u / s) ** k)
    e1 = np.zeros(12); e1[0] = 1.0
    rows = []
    for a in ages:
        p, _ = quad_vec(lambda u: onset(u) * (expm(Q.T * (a - u)) @ e1), 0.0, a, epsabs=1e-14, epsrel=1e-12)
        rows.append({"T1": VALUES[3] * p[ix["U1"]], "D1": VALUES[10] * p[ix["T1"]], "E1": VALUES[4] * p[ix["U1"]]})
    return rows


def max_rel(a, b, cols):
    m = 0.0
    for c in cols:
        ok = np.isfinite(a[c]) & np.isfinite(b[c]) & (b[c] > 0)
        m = max(m, float(np.max(np.abs(a.loc[ok, c] - b.loc[ok, c]) / b.loc[ok, c])))
    return m


if __name__ == "__main__":
    m = SingularCancer("Urinary Bladder", 7.0, 140.0)
    tg, arr, _ = smorfs_targets(m, M.t, M.dt)
    tg.to_csv(os.path.join(ROOT, "smorfs_targets.csv"), index=False)
    tgc = smorfs_targets_cumulative(SingularCancer("Urinary Bladder", 7.0, 140.0))
    tgc.to_csv(os.path.join(ROOT, "smorfs_targets_cumulative.csv"), index=False)
    cols = [c for c in tg.columns if c != "age"]

    # step halving of the summation grid (the inversion itself is unchanged)
    t2 = np.arange(0.0, M.t[-1] + M.dt, M.dt / 2)
    tg2, _, _ = smorfs_targets(SingularCancer("Urinary Bladder", 7.0, 140.0), t2, M.dt / 2)
    halving = max_rel(tg, tg2, cols)

    # pointwise inversion accuracy at grid points, against quadrature of a matrix exponential
    gi = [int(np.argmin(np.abs(M.t - a))) for a in (40, 50, 60, 70, 80)]
    refp = reference_flows_pointwise([float(M.t[i]) for i in gi])
    st = {"T1": m.stage_states["clinical incidence"]["Early"], "D1": m.stage_states["clinical mortality"]["Early"],
          "E1": m.stage_states["preclinical mortality"]["Early"]}
    inv_err = max(abs(arr[st[n]].value[i] - r[n]) / r[n] for i, r in zip(gi, refp) for n in st)

    # timings, eval_targets as used in estimation
    mt = SingularCancer("Urinary Bladder", 7.0, 140.0)
    mt.targets = exact_targets(mt.stage_groups)
    S = M.get_survival_function()
    mt.survival_function = S; mt.survival_weights = S.cumsum() * M.dt
    mt.survival_grad = -np.gradient(gaussian_filter1d(S, int(1.0 / M.dt), mode="nearest"), M.dt)
    mt.set_parameters(THETA)
    times = {}
    for order, label in [(0, "objective"), (1, "objective_and_gradient"), (2, "objective_gradient_hessian")]:
        mt.eval_order = order; mt.eval_targets()                     # warm-up
        runs = []
        for _ in range(REPEATS):
            t_start = time.perf_counter(); mt.eval_targets(); runs.append(time.perf_counter() - t_start)
        times[label] = float(np.median(runs))

    ref_path = os.path.join(ROOT, "reference_targets.csv")
    if not os.path.exists(ref_path):
        ref_path = vfile("reference_targets.csv")
    err = err_cum = None
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        ref = ref[ref["age"].isin(AGES)].reset_index(drop=True)
        err = max_rel(tg, ref, cols)
        err_cum = max_rel(tgc, ref, cols)
        by_family = {f: max_rel(tg, ref, [c for c in cols if c.startswith(f)]) for f in ("clin_inc", "clin_mort", "preclin_mort")}

    info = {"machine": platform.processor() or platform.machine(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
            "grid_step_years": M.dt, "grid_points": len(M.t)}
    rows = [dict(quantity="max_rel_error_vs_reference", value=err),
            dict(quantity="max_rel_error_vs_reference_cumulative_inversion", value=err_cum),
            dict(quantity="max_rel_change_on_halving_grid_step", value=halving),
            dict(quantity="max_rel_error_pointwise_inversion", value=inv_err)]
    rows += [dict(quantity=f"seconds_{k}", value=v) for k, v in times.items()]
    rows += [dict(quantity=k, value=v) for k, v in info.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "smorfs_benchmark.csv"), index=False)

    print("SMORFS benchmark, bladder model, rounded Table 2 parameters, background mortality off")
    if err is not None:
        print(f"  max relative error vs R reference (9 series, ages 35-80): {err:.2e}")
        print("  by family: " + ", ".join(f"{k} {v:.2e}" for k, v in by_family.items()))
        print(f"  same, numerators from cumulative inversion at the age cut points: {err_cum:.2e}")
    else:
        print("  (reference_targets.csv not found: run bladder_reference_check.R first for the comparison)")
    print(f"  pointwise inversion error at ages 40-80 (vs quadrature): {inv_err:.2e}")
    print(f"  change on halving the summation step {M.dt} -> {M.dt/2}: {halving:.2e}")
    print("  seconds (median of %d): objective %.3f, with gradient %.3f, with Hessian %.3f"
          % (REPEATS, times["objective"], times["objective_and_gradient"], times["objective_gradient_hessian"]))
    print(f"  machine: {info['machine']}, {info['cpu_count']} cores, {info['platform']}, Python {info['python']}")
    print("Wrote smorfs_targets.csv and smorfs_benchmark.csv")
