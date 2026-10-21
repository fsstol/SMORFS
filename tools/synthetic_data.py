"""SYNTHETIC data for the bladder-cancer model, and the whole analysis run on it. No SEER data is used or needed.

    python tools/synthetic_data.py                        # writes the synthetic working copy synthetic_run/
    python tools/synthetic_data.py --pipeline             # ... and runs MLE, LIMIS (small) and figures on it
    python tools/synthetic_data.py --pipeline --limis full --workers 20   # LIMIS with the paper's settings

The working copy (default synthetic_run/, git-ignored) holds this repository's code (src/, tests/, verification/,
compile.sh) and a data/ folder in which the three SEER files main.py reads are replaced by SYNTHETIC files with
exactly their names, columns and labels. Everything that needs data runs there, so nothing in the repository itself can
be mistaken for SEER data. The working copy is marked SYNTHETIC_DATA.md, and every result collected from it goes to
output/synthetic/ under a "SYNTHETIC" name.

How the data are made. The model is evaluated at the committed MLE (output/urinary bladder mle.npy, the generating
parameters) with main.py's own target machinery: verification/h2.py builds the targets from a template of the three
files (every count 0), and eval_targets(return_targets=True) gives the model's rate for every family, stage group and
single year of age 0 to 84. Each count is then Poisson(rate x person-years), with these stated SYNTHETIC person-years
for every single year of age (the same for every stage, as in SEER's rate sessions):
    clinical incidence        rates_urinary_bladder.csv        PERSON_YEARS["incidence"]
    clinical mortality        mortality_urinary_bladder.csv    PERSON_YEARS["mortality"]
    preclinical mortality     doa_urinary bladder.pkl, outcome 1  PERSON_YEARS["doa"]
    preclinical other mortality  doa_urinary bladder.pkl, outcome 2  PERSON_YEARS["doa"]
The Early group's count is split between SEER's "In situ" (EARLY_IN_SITU_SHARE) and "Localized" stages by drawing each
as an independent Poisson count, so their sum is Poisson(rate x person-years); Medium is "Regional" and Late
"Distant". Age 85+ gets count 0 (main.py does not use it). Crude rates, standard errors and exact Poisson 95% limits
are written as SEER*Stat prints them (per 100,000, six decimals; "~" for the SE of a zero count). The year of
diagnosis is the single label "synthetic"; main.py drops only the "2000-2021" aggregate. The seed is fixed (--seed).

The pipeline (--pipeline), all in the working copy:
  1. MLE: main.py's default optimizer from output/urinary bladder.npy (tests/common.run_refit), the SEs from the
     observed information, and the parameter-recovery table (generating value, estimate, 95% CI, z).
  2. LIMIS: src/limis_runs.py with the wide prior (max(0.3, 4 SE)) around the synthetic MLE; --limis small (2 runs,
     n0 = 2,000, b = 200, 10 epochs; a few minutes) or full (20 runs, parameters.py's settings).
  3. tests/limis_diagnostics.py, and tests/limis_figures.py's two commands (Figures 3 to 7, summary table).
  4. The results are copied to output/synthetic/, each name starting with "SYNTHETIC", with a README; every figure
     also carries a red "SYNTHETIC DATA" line (the working copy's tests/limis_figures.py is patched to stamp it).
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = "Combined Summary Stage (2004+)"
AGE_RATES = "Age recode with single ages and 85+"
AGE_DEATH = "Age at death with single ages and 85+"
CAUSE = "SEER cause-specific death classification"
CANCER_DEATH = "Dead (attributable to this cancer dx)"
YEAR = "Year of diagnosis"
AGES = [f"{a:02d} years" for a in range(85)] + ["85+ years"]
SEER_STAGES = {"Early": ["In situ", "Localized"], "Medium": ["Regional"], "Late": ["Distant"]}
PERSON_YEARS = {"incidence": 20_000_000, "mortality": 20_000_000, "doa": 30_000_000}  # SYNTHETIC, per year of age
EARLY_IN_SITU_SHARE = 0.2
SEED = 20261020
MARKER = "SYNTHETIC_DATA.md"
FILES = {"incidence": "rates_urinary_bladder.csv", "mortality": "mortality_urinary_bladder.csv",
         "doa": "doa_urinary bladder.pkl"}


def seer_numbers(count, pop):
    """Crude rate, SE and exact Poisson 95% limits per 100,000, as SEER*Stat prints them."""
    rate = 1e5 * count / pop
    se = "~" if count == 0 else f"{1e5 * np.sqrt(count) / pop:.6f}"
    low = 0.0 if count == 0 else chi2.ppf(0.025, 2 * count) / 2
    high = chi2.ppf(0.975, 2 * count + 2) / 2
    return f"{rate:.6f}", se, f"{1e5 * low / pop:.6f}", f"{1e5 * high / pop:.6f}"


def write_data(data_dir, counts):
    """The three files main.py reads. counts[kind][(seer_stage, age)] (kind: incidence, mortality, doa1, doa2)."""
    os.makedirs(os.path.join(data_dir, "single-multiple"), exist_ok=True)
    stages = [s for group in SEER_STAGES.values() for s in group]
    rows = []
    for st in stages:
        for a in AGES:
            c = counts["incidence"].get((st, a), 0)
            rows.append([st, "synthetic", a, *seer_numbers(c, PERSON_YEARS["incidence"]), c, PERSON_YEARS["incidence"]])
    pd.DataFrame(rows, columns=[STAGE, YEAR, AGE_RATES, "Crude Rate", "Standard Error", "Lower Confidence Interval",
                                "Upper Confidence Interval", "Count", "Population"]).to_csv(
        os.path.join(data_dir, "single-multiple", FILES["incidence"]), index=False)
    rows = []
    for st in stages:
        for a in AGES:
            c = counts["mortality"].get((st, a), 0)
            rows.append([st, CANCER_DEATH, a, *seer_numbers(c, PERSON_YEARS["mortality"]), c, PERSON_YEARS["mortality"]])
    pd.DataFrame(rows, columns=[STAGE, CAUSE, AGE_DEATH, "Crude Rate", "Standard Error", "Lower Confidence Interval",
                                "Upper Confidence Interval", "Count", "Population"]).to_csv(
        os.path.join(data_dir, "single-multiple", FILES["mortality"]), index=False)
    index, pops, cs = [], [], []
    for outcome in (1, 2):
        for st in sorted(stages):
            for a in AGES:
                index.append((outcome, st, a))
                pops.append(PERSON_YEARS["doa"])
                cs.append(counts[f"doa{outcome}"].get((st, a), 0))
    doa = pd.DataFrame({"Population": np.array(pops, dtype=np.int64), "Count": np.array(cs, dtype=np.int64)},
                       index=pd.MultiIndex.from_tuples(index, names=["outcome", STAGE, AGE_RATES]))
    doa.to_pickle(os.path.join(data_dir, "single-multiple", FILES["doa"]))


RATES_CHILD = r'''
import json, sys
import numpy as np
sys.path.insert(0, "../verification")
import h2
m = h2.build("model")
m.set_parameters(np.load("../output/urinary bladder mle.npy"))
m.eval_order = 0
_, y = m.eval_targets(return_targets=True)
out = {}
for family, series in m.targets.items():
    for stage, target in series.items():
        keys = list(target.keys())
        out[f"{family}|{stage}"] = {a: float(y[family][stage][i]) / 1e5 for i, a in enumerate(keys[:-1])}
json.dump(out, open(sys.argv[1], "w"))
'''


def model_rates(workdir, python):
    """The model's rate per person-year for every family, stage group and age, at the generating parameters."""
    path = os.path.join(workdir, "model_rates.json")
    subprocess.run([python, "-c", RATES_CHILD, path], cwd=os.path.join(workdir, "src"), check=True,
                   env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True)
    with open(path) as fh:
        return json.load(fh)


STAMP = r'''
# SYNTHETIC working copy (tools/synthetic_data.py): every figure is stamped, so that none can pass for a SEER result
_savefig = plt.savefig
def _synthetic_savefig(*args, **kwargs):
    fig = plt.gcf()
    if not getattr(fig, "_synthetic_stamp", False):
        fig.text(0.5, 0.999, "SYNTHETIC DATA (tools/synthetic_data.py): not SEER data, not the paper's results",
                 ha="center", va="top", fontsize=10, color="tab:red")
        fig._synthetic_stamp = True
    return _savefig(*args, **kwargs)
plt.savefig = _synthetic_savefig
'''


def stamp_figures(path):
    """In the working copy only: limis_figures.py's child stamps SYNTHETIC on every figure it saves."""
    s = open(path).read()
    anchor = "import model, main as M\n"
    assert s.count(anchor) == 1, "tests/limis_figures.py changed: cannot stamp the synthetic figures"
    open(path, "w").write(s.replace(anchor, anchor + STAMP))


def make_workdir(workdir, python, seed):
    if os.path.exists(workdir):
        if not os.path.exists(os.path.join(workdir, MARKER)):
            sys.exit(f"{workdir} exists and is not a synthetic working copy (no {MARKER}): not overwritten")
        shutil.rmtree(workdir)
    ignore = shutil.ignore_patterns("__pycache__", ".DS_Store", "*.o", "synthetic_run")
    for d in ("src", "tests", "verification"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(workdir, d), ignore=ignore)
    stamp_figures(os.path.join(workdir, "tests", "limis_figures.py"))
    shutil.copy2(os.path.join(ROOT, "compile.sh"), workdir)
    lib = os.path.join(ROOT, "semi_markov.dll" if platform.system() == "Windows" else "semi_markov.so")
    if os.path.exists(lib):
        shutil.copy2(lib, workdir)
    else:
        subprocess.run(["bash", "compile.sh"], cwd=workdir, check=True)
    os.makedirs(os.path.join(workdir, "output"))
    for f in ("urinary bladder.npy", "urinary bladder mle.npy"):
        shutil.copy2(os.path.join(ROOT, "output", f), os.path.join(workdir, "output", f))
    shutil.copy2(os.path.join(ROOT, "output", "urinary bladder mle.npy"),
                 os.path.join(workdir, "output", "urinary bladder mle generating.npy"))
    os.makedirs(os.path.join(workdir, "data"))
    shutil.copy2(os.path.join(ROOT, "data", "US_life_table_2021.csv"), os.path.join(workdir, "data"))

    # 1. a template with every count 0, so that h2 can build the targets' structure; 2. the model's rates; 3. counts
    write_data(os.path.join(workdir, "data"), {"incidence": {}, "mortality": {}, "doa1": {}, "doa2": {}})
    rates = model_rates(workdir, python)
    rng = np.random.default_rng(seed)
    kinds = {"clinical incidence": ("incidence", "incidence"), "clinical mortality": ("mortality", "mortality"),
             "preclinical mortality": ("doa1", "doa"), "preclinical other mortality": ("doa2", "doa")}
    counts = {k: {} for k in ("incidence", "mortality", "doa1", "doa2")}
    for key, by_age in sorted(rates.items()):
        family, group = key.split("|")
        kind, py = kinds[family]
        for a in AGES[:-1]:
            r = by_age.get(a, 0.0)
            r = r if np.isfinite(r) and r > 0 else 0.0
            stages = SEER_STAGES[group]
            shares = [EARLY_IN_SITU_SHARE, 1 - EARLY_IN_SITU_SHARE] if len(stages) == 2 else [1.0]
            for st, share in zip(stages, shares):
                counts[kind][(st, a)] = int(rng.poisson(r * share * PERSON_YEARS[py]))
    write_data(os.path.join(workdir, "data"), counts)
    os.remove(os.path.join(workdir, "model_rates.json"))
    n_events = {k: sum(v.values()) for k, v in counts.items()}
    note = f"""# SYNTHETIC DATA: not SEER data

This folder is a working copy made by `tools/synthetic_data.py`. Its `data/single-multiple/` files have the names,
columns and labels of the SEER files `main.py` reads, but every count in them is SYNTHETIC: Poisson draws from the
model at the generating parameters `output/urinary bladder mle generating.npy`, with seed {seed}.

- Person-years per single year of age (synthetic): {PERSON_YEARS}.
- Early stage split between "In situ" and "Localized": {EARLY_IN_SITU_SHARE} / {1 - EARLY_IN_SITU_SHARE:g}.
- Events drawn: {n_events}.
- Results computed here describe the synthetic data only.
"""
    for p in (os.path.join(workdir, MARKER), os.path.join(workdir, "data", MARKER)):
        with open(p, "w") as fh:
            fh.write(note)
    return n_events


def refit(workdir, python):
    """Step 1: the MLE on the synthetic data, and the parameter-recovery table."""
    code = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, "tests")
from common import ROOT, load_h2, run_refit
d, proc, log, th = run_refit(100)
open("output/refit.log", "w").write(proc.stdout + proc.stderr)
if th is None:
    sys.exit("main.py saved no estimate")
np.save(os.path.join(ROOT, "output", "urinary bladder mle.npy"), th)
h2 = load_h2()
m = h2.build("model")
fv, g, _ = h2.ev(m, th, 1)
_, _, se = m.limis_bounds(os.path.join(ROOT, "output", "urinary bladder mle.npy"))
gen = np.load(os.path.join(ROOT, "output", "urinary bladder mle generating.npy"))
json.dump({"theta": th.tolist(), "se": se.tolist(), "generating": gen.tolist(), "objective": float(fv),
           "gradient_norm": float(np.linalg.norm(g)), "iterations": len(log) - 1, "names": h2.NAMES},
          open(os.path.join(ROOT, "output", "synthetic_mle.json"), "w"))
'''
    subprocess.run([python, "-c", code], cwd=workdir, check=True, env=dict(os.environ, MPLBACKEND="Agg"))
    with open(os.path.join(workdir, "output", "synthetic_mle.json")) as fh:
        r = json.load(fh)
    th, se, gen = (np.array(r[k]) for k in ("theta", "se", "generating"))
    table = pd.DataFrame({"parameter": r["names"], "generating": np.exp(gen), "estimate": np.exp(th),
                          "ci_low": np.exp(th - 1.96 * se), "ci_high": np.exp(th + 1.96 * se),
                          "z_log": (th - gen) / se})
    table.to_csv(os.path.join(workdir, "output", "synthetic_parameter_recovery.csv"), index=False, float_format="%.6g")
    return r, table


def pipeline(workdir, python, limis, workers):
    r, table = refit(workdir, python)
    print(f"MLE: {r['iterations']} Newton iterations, objective {r['objective']:.4f}, gradient norm "
          f"{r['gradient_norm']:.2e}; |z| <= {np.abs(table.z_log).max():.2f} against the generating parameters", flush=True)
    seeds, size = ("0-19", []) if limis == "full" else ("0-1", ["--n0", "2000", "--b", "200", "--epochs", "10"])
    t0 = time.time()
    subprocess.run([python, "limis_runs.py", "--seeds", seeds, "--workers", str(workers), "--min-half-width", "0.3",
                    "--limis-dir", "../output/IMIS2_synthetic"] + size, cwd=os.path.join(workdir, "src"), check=True)
    print(f"LIMIS ({limis}): {(time.time() - t0) / 60:.1f} min", flush=True)
    subprocess.run([python, "tests/limis_diagnostics.py", "--samples", "output/IMIS2_synthetic/samples", "--out",
                    "output/IMIS2_synthetic/diagnostics"], cwd=workdir, check=True)
    for extra in ([], ["--steps", "posterior", "--axes-to-bounds"]):
        subprocess.run([python, "tests/limis_figures.py", "--limis-dir", "output/IMIS2_synthetic", "--min-half-width",
                        "0.3", "--dest", "output/synthetic"] + extra, cwd=workdir, check=True)
    return r, table


def collect(workdir, limis, n_events, seed, r, table):
    """Step 4: the results, each named SYNTHETIC ..., in output/synthetic/ of the repository."""
    dest = os.path.join(ROOT, "output", "synthetic")
    os.makedirs(dest, exist_ok=True)
    copied = []
    for f in sorted(os.listdir(os.path.join(workdir, "output", "synthetic"))):
        shutil.copy2(os.path.join(workdir, "output", "synthetic", f), os.path.join(dest, f"SYNTHETIC {f}"))
        copied.append(f"SYNTHETIC {f}")
    for f in ("runs.csv", "parameters.csv", "bounds.csv"):
        shutil.copy2(os.path.join(workdir, "output", "IMIS2_synthetic", "diagnostics", f),
                     os.path.join(dest, f"SYNTHETIC diagnostics {f}"))
        copied.append(f"SYNTHETIC diagnostics {f}")
    shutil.copy2(os.path.join(workdir, "output", "synthetic_parameter_recovery.csv"),
                 os.path.join(dest, "SYNTHETIC parameter recovery.csv"))
    copied.append("SYNTHETIC parameter recovery.csv")
    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write(f"""# SYNTHETIC results: not results of the paper

Every file here was computed by `tools/synthetic_data.py --pipeline --limis {limis}` on SYNTHETIC data (seed {seed};
person-years per single year of age {PERSON_YEARS}; events drawn {n_events}), generated from the model at the
committed MLE. They show that the pipeline runs and that the estimation recovers the generating parameters; they
describe no real population.

- MLE: {r['iterations']} Newton iterations, objective {r['objective']:.4f}, gradient norm {r['gradient_norm']:.2e}.
- Parameter recovery (`SYNTHETIC parameter recovery.csv`): largest |z| = {np.abs(table.z_log).max():.2f}, where
  z = (log estimate - log generating value) / SE.

Files: {", ".join(f"`{c}`" for c in copied)}.
""")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default=os.path.join(ROOT, "synthetic_run"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pipeline", action="store_true", help="also run MLE, LIMIS and figures on the data")
    parser.add_argument("--limis", choices=["small", "full"], default="small")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    workdir = os.path.abspath(args.workdir)
    n_events = make_workdir(workdir, sys.executable, args.seed)
    print(f"SYNTHETIC working copy {workdir}; events drawn: {n_events}", flush=True)
    if args.pipeline:
        r, table = pipeline(workdir, sys.executable, args.limis, args.workers)
        print("results:", collect(workdir, args.limis, n_events, args.seed, r, table))


if __name__ == "__main__":
    main()
