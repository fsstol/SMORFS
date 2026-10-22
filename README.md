# SMORFS: Semi-Markov Optimization and Regression by Fourier Series

SMORFS evaluates semi-Markov disease natural history models in the frequency domain. It computes the Laplace
transforms of the transition densities and inverts them with an accelerated Fourier series. This gives model outputs
(state prevalences, arrival rates, incidence and mortality targets) and their exact gradients and Hessians with
respect to the parameters. It also includes estimation by maximum likelihood (saddle-free Newton) and Bayesian
calibration by Langevin incremental mixture importance sampling (LIMIS).

This repository accompanies the paper "Frequency-Domain Evaluation and Estimation of Semi-Markov Disease Natural
History Models", whose application is a natural history model of urinary bladder cancer fitted to SEER data.

**SEER data are not included.** SEER's data-use agreement does not allow SEER data, or files derived from it, in a
public repository. `data/README.md` describes the files the model expects and how to build them from SEER. Everything
here also runs on **synthetic data** drawn from the fitted model (`tools/synthetic_data.py`), which has the same files
and formats.

## Contents

| Path | What it holds |
|---|---|
| `src/semi_markov/`, `src/src_lib/` | The method: the Python package and its C++ library (the Fourier inversion, transforms, derivatives, optimizer and sampler steps), with Eigen 3.4.0 vendored in `src/src_lib/eigen/` (MPL 2.0) |
| `src/model.py`, `src/main.py`, `src/parameters.py` | The bladder-cancer model: its structure, its targets and its estimation |
| `src/limis_runs.py` | Independent LIMIS runs in parallel |
| `verification/` | The independent R reference and the comparison methods (cohort and microsimulation; `bladder_reference_check.R`), and helpers the tests use |
| `tests/` | The tests (`run_public_tests.py`) and the scripts that make the tables and figures |
| `tools/synthetic_data.py` | Synthetic data, and the whole analysis run on them |
| `data/` | The 2021 period life table of the US Social Security Administration (Table 4.C6; public data), and `data/README.md` for the SEER files |
| `output/` | The paper's estimates and tables: the MLE (`urinary bladder mle.npy`), Table 2 (`final/urinary bladder CIs.csv`), Table 3 (`benchmark/table3.csv`) |
| `paper_figures/` | The paper's figures that show no SEER data: Figures 6 and 7 (posterior histograms and correlogram) and Figure S1 (accuracy of the inversion) |

## Setup

Requirements: Python 3.12, a C++20 compiler, and R for the comparison methods (base R only). `requirements.txt` pins
the versions that produced the paper's results; `environment.yml` is the same environment as a conda export
(macOS arm64).

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash compile.sh                  # builds semi_markov.so here
                                 # (Windows, or anywhere with CMake: cmake -S . -B build && cmake --build build --config
                                 # Release, which builds semi_markov.dll or semi_markov.so here)
python tests/run_public_tests.py # the public tests (below), about 1 minute
```

## The analysis on synthetic data

```
python tools/synthetic_data.py --pipeline                          # small LIMIS: about 3 minutes
python tools/synthetic_data.py --pipeline --limis full --workers 20 # LIMIS with the paper's settings
```

The script writes a working copy, `synthetic_run/` (git-ignored), of this repository's code. Its `data/` holds
synthetic files with exactly the names, columns and labels of the SEER files. Each count is a Poisson draw from the
model at the committed MLE, on stated synthetic person-years (the script's docstring and `synthetic_run/SYNTHETIC_DATA.md`
give the details and the seed). In that working copy it then runs:
1. the MLE, from the earlier estimate `output/urinary bladder.npy`, with `main.py`'s optimizer;
2. LIMIS, with the wide prior, max(0.3, 4 SE) on the log scale, around that MLE;
3. the diagnostics;
4. the figures and the summary table.

The results go to `output/synthetic/` (git-ignored). Every name starts with "SYNTHETIC", every figure carries a red
"SYNTHETIC DATA" line, and `README.md` there states what they are. They include a parameter-recovery table comparing the estimates with the generating values.

## Tests

`python tests/run_public_tests.py` needs no SEER data.
- **Without data:** `general` (other model structures: cycles, shared risks, several risks per state, structure errors),
  `steps`, `acceleration`, `risks` (Gompertz, log-normal and log-logistic risks against 50-digit references), `quadws`
  (an AddressSanitizer build) and `limis` (LIMIS on a toy target with a known posterior).
- **On synthetic data, in a temporary working copy:** `othercause`, `weibull`, `T2` (gradients and Hessians against
  finite differences), `T3c`, `cache`, `threads`, `emptygrid`, `quadrature`, `errorbars`, `runbox`, `recovery` (the MLE
  recovers the generating parameters) and `denominators` (every target's person-years are the rates file's,
  `CORRECTIONS.md` entry 16). With R: `T3` and `T3b` (SMORFS against the independent R reference).

The tests of the paper's own numbers need the SEER data and references computed from it, and are not included.

## Reproducing the paper (needs the SEER data)

With the three SEER files in `data/single-multiple/` (`data/README.md`), run from this folder:

| Result | Command | Time (Apple M4 Max) |
|---|---|---|
| Table 2, MLE column | `python tests/table2_mle.py` | 3 s |
| LIMIS runs | `cd src && python limis_runs.py --seeds 0-19 --workers 20 --min-half-width 0.3 --limis-dir ../output/IMIS2_corrected` | 7 min |
| Table 2, LIMIS column; Figures 3 to 7 | `python tests/limis_figures.py --limis-dir output/IMIS2_corrected --min-half-width 0.3 --dest output/final`, then the same with `--steps posterior --axes-to-bounds` | 3 min |
| Table 3 | `python tests/benchmark.py` (needs `Rscript`) | 20 s |
| Figure S1 | `cd verification && Rscript bladder_reference_check.R && cd ..`, then `python tests/inversion_convergence.py` | 3 s |

- **Table 2.** `tests/table2_mle.py` rebuilds the 552 targets with `main.py`'s functions and refits the model. It then
  compares the MLE column, at four decimal places, with `output/final/urinary bladder CIs.csv`, and exits with
  status 1 if any cell differs.
- **The targets** (`data/README.md`). Every target is count / person-years, with SE √count / person-years. The
  person-years of all four families are SEER 17's for 2004 to 2021, by single year of age, from the rates file
  (`main.corrected_denominators = True`, the default since v1.0.1; `CORRECTIONS.md` entry 16).
- **LIMIS.** The prior is uniform on the log scale, centred at `output/urinary bladder mle.npy`, with half-width
  max(0.3, 4 SE). There are 20 runs with seeds 0 to 19, each with 10,000 initial samples and 100 steps of 1,000.
- **Figures.** The second `limis_figures.py` command draws Figure 6 with its x-axes at the prior bounds. Figures 3 to 5
  show SEER rates, so only the outputs of these commands contain them; keep those out of public repositories.
- **The correlogram's font.** `tests/limis_figures.py` puts DejaVu Sans first in seaborn's font list. With seaborn's
  default (Arial first), macOS renders the subscripts of the labels (U₁→U₂) as boxes.

## Licence

MIT; see `LICENSE`. Eigen (`src/src_lib/eigen/`) is under the MPL 2.0.
