# Corrections since v2

This release corrects the code of SMORFS v2, the version behind the first submission. Each entry gives the code
location, what was wrong in v2, what changed, and the test that checks it. A test named here is part of the public
suite (`python tests/run_public_tests.py`). There it runs without data, or on synthetic data drawn from the model. An
entry whose check needs the SEER data says "verified on the SEER data; not in the public suite".

## Estimation of the bladder-cancer model

### 1. Targets: no halving of the population

- **Where:** `src/main.py`, `main_rates`.
- **v2:** the `Population` column was divided by 2 in all three data frames: incidence, cancer mortality, and
  deaths at diagnosis. Every target was therefore twice SEER's rate.
- **Now:** the halving is removed, and each target is count / population, as SEER*Stat computes its crude rates.
- **Test:** verified on the SEER data (the targets against the published tables); not in the public suite.

### 2. Targets: the age-80 targets enter the objective

- **Where:** `src/main.py`, `main_rates` (the closing cut point).
- **v2:** every age in `parameters.ages_remove` was deleted, including "81 years". Each interval target uses two
  consecutive cut points, so the 9 interval targets and 3 pointwise targets at age 80 were silently dropped from the
  objective.
- **Now:** "81 years" is kept as the closing cut point, with an infinite standard error, so that it is never fitted
  itself.
- **Test:** verified on the SEER data (`main.py`'s objective equals one over all the published targets); not in the
  public suite.

### 3. Hot start: the right file, and no silent fallback

- **Where:** `src/main.py`, `main_rates` (`filename_hot_start_c`).
- **v2:** the hot start loaded `output/urinary bladder mle.npy`, which did not exist. On failure it printed a
  traceback and carried on from the default parameters.
- **Now:** it loads `output/urinary bladder.npy`, and a missing file raises an error. Saving (`hot_start_save`)
  still writes `output/urinary bladder mle.npy`, so the starting estimate is never overwritten.
- **Test:** verified on the SEER data (the hot start's objective); not in the public suite.

### 4. Hessian: the Gauss-Newton term of each target

- **Where:** `src/model.py`, `eval_targets`, the interval targets and the pointwise other-cause targets.
- **v2:** an interval target's model value is a difference of cumulative values at its two cut points, weighted by
  c_i. v2 formed its Gauss-Newton term as G_{i+1}G_{i+1}^T − G_iG_i^T, which is not the outer product of the
  difference. The pointwise other-cause block had the analogous error. The Hessian was wrong and could be
  indefinite at the optimum.
- **Now:** each term is `np.outer(g_i, g_i)`, with g_i = (G_{i+1} − G_i) c_i, and likewise for the pointwise block.
- **Test:** `T2` (the Hessian against central differences of the analytic gradient, and positive definite at the MLE).

### 5. The other-cause mortality target is a rate per living person

- **Where:** `src/model.py`, `eval_targets`, the "preclinical other mortality" block (value, gradient and Hessian).
- **v2:** the model value was the all-cause death density times the preclinical prevalence, a death density per
  member of the birth cohort. The SEER target is a rate per living person.
- **Now:** the value and its derivatives are multiplied by S'_US / S_US, so the target is λ_US(t) p_s(t).
- **Test:** `othercause` (the model value equals λ_US(t) p_s(t)), and `T2`.

### 6. Weibull onset: the derivatives with respect to shape and scale

- **Where:**
  - `src/src_lib/risk.cpp`, `Weibull::eval_time`;
  - `src/src_lib/semi_markov.cpp`, `eval_time_state`.
- **v2:** three coding errors.
  - The derivatives of H = (t/s)^k with respect to log k and log s were swapped.
  - The derivative of the survival, dS, had the wrong sign: it was +dH S, not −dH S.
  - ∂²h/∂s² had the wrong denominator.

  The shape and scale gradients and Hessian were wrong. The exponential rates were not affected.
- **Now:** dH and its second derivatives are in the right order, dS = −dH S, and ∂²h/∂s² = h k(k+1)/s².
- **Test:** `weibull` (the risk's stack against the closed form, the state's derivatives against central
  differences), and `T2`.

### 7. Interval targets by cumulative inversion, not grid summation

- **Where:** `src/model.py`, `eval_targets` (`SingularCancer.cumulative_targets = True`).
- **v2:** each interval target's numerator was a left Riemann sum of S_US × the arrival density over the model's
  time grid. That was the largest numerical error in the targets.
- **Now:** the numerators come from the cumulative arrival, from the Fourier inversion with `cumulative=True`,
  evaluated at break points.
  - The break points are every integer age and every half-year step of the nearest-age S_US.
  - The integral of the step function S_US times the arrival density is exact on each piece.
  - The denominators, the integrals of S_US, are exact for the same step function.
  - Gradients and Hessians follow the same path.
  - The grid path remains behind `cumulative_targets = False`, for comparison.
- **Test:** `T3b` (against the independent R reference, with R) and `T3c` (against the exact step-weighted integral).

### 8. LIMIS: four errors in the importance sampler (fixes A to D)

- **Where:** `src/model.py`, `IMIS0`, `IMIS2` and `SingularCancer.limis_weights`.
- **v2 and now:**
  - **A.** The likelihood of each incremental sample was evaluated at the component's centre; now it is evaluated at
    the sample.
  - **B.** The mixture's prior row held the likelihood; now it holds the prior density.
  - **C.** The mixture weights were applied cumulatively, to a view of the array, at every epoch; now they are
    applied once, to a copy.
  - **D.** The final weights were the target density alone; now they are target / mixture density.
- **Test:** `limis` (LIMIS on a toy target with a known posterior: each sample's own log target, the mixture density
  against an independent recomputation, and the weighted posterior mean and SD against the known ones).

### 9. Optimizer: full saddle-free Newton steps with backtracking by default

- **Where:**
  - `src/main.py`, `newton_backtracking_step`;
  - `src/parameters.py` (`newton_damped = False`, `newton_tol = 1e-5`, `newton_max_step = 1.0`,
    `newton_max_halvings = 20`).
- **v2:** every Newton step was the saddle-free step scaled by `lr_newton` / √n, with no line search. That damping
  compensated for the derivative errors above, and the optimizer converged only linearly.
- **Now:** each iteration takes the saddle-free direction −|H|⁻¹g.
  - The step is capped so that no parameter moves by more than `newton_max_step` on the log scale.
  - It backtracks by halving until the Armijo condition holds.
  - The optimizer stops when the gradient norm is below `newton_tol`.
  - v2's schedule remains behind `newton_damped = True`.
- **Test:** `recovery` (on synthetic data, `main.py`'s default optimizer converges, with gradient norm below 1e-5, and
  recovers the generating parameters).

## The library

### 10. Hessian: the cross terms between two risks out of one quadrature state

- **Where:** `src/src_lib/semi_markov.cpp`, `laplace_general_state`.
- **v2:** a quadrature state has a non-exponential risk out of it. For such a state, v2 kept the second derivatives
  only within each risk's own parameters, and dropped ∂²S/∂a∂b = S ∂H₁/∂a ∂H₂/∂b between two risks. The Hessian
  was wrong for any model with two or more risks out of a quadrature state. The bladder model has one such state, H0,
  with a single risk, so it was not affected.
- **Now:** the state's whole local block of second derivatives is kept.
- **Test:** `general` (model G1: a cycle through a quadrature state with two risks; Hessian against central
  differences).

### 11. A second transition between the same two states

- **Where:**
  - `src/src_lib/model.cpp`, `State::add_transition`;
  - `src/semi_markov/model.py`, `State.add_transition`.
- **v2:** the Python side kept a dict keyed by destination, so a second transition to the same state replaced the
  first. The C++ side added both, and the two disagreed.
- **Now:** a second transition to the same destination raises `ValueError`. So does a transition to a state not in
  the model.
- **Test:** `general`.

### 12. Accuracy of the series acceleration's weights

- **Where:** `src/src_lib/fourier.cpp`, `acc_alg1`.
- **v2:** the weights of the Cohen–Rodriguez Villegas–Zagier acceleration came from a forward recursion that
  subtracts. The smallest weights lost relative precision, and at many periods the last weights were lost
  entirely.
- **Now:** the weights are backward tail sums of positive terms, accurate to a few units of machine precision at any
  number of periods. The change is far below the inversion's error at the model's 12 periods.
- **Test:** `acceleration` (the weights against exact rational values, for 1 to 48 periods).

### 13. The shared quadrature's error buffer (performance work)

- **Where:** `src/src_lib/quad_vec.cpp`, `QuadMultiWorkspace::claim`.
- **Before:** the performance work computes all frequencies' transforms with one shared adaptive quadrature. Before
  this release, its workspace enlarged its error buffer only when its integral buffer grew. After a call with a long
  stack, a call with a short stack that needed more intervals wrote past the end of the error buffer.
- **Now:** each buffer is resized on its own condition.
- **Test:** `quadws` (an AddressSanitizer build of the sequence that overflowed: a second-order evaluation at the
  MLE, then an objective at a much smaller Weibull shape).

### 14. A change of structure after `update()`

- **Where:**
  - `src/src_lib/semi_markov.cpp`, `structure_current` and `evaluate`;
  - `src/src_lib/exports.cpp`;
  - `src/semi_markov/semi_markov.py`.
- **Before:** after `update()`, adding a state or a transition went undetected. The C++ library could then return the
  old structure's results from its cache, or write past a workspace buffer.
- **Now:** `update()` records the structure. An evaluation after a change raises `RuntimeError`, and asks for
  `update()` again.
- **Test:** `general` (a transition or a state added after `update()` must raise, and after `update()` the results
  must equal a freshly built model's, bit for bit).

### 15. An empty time grid

- **Where:** `src/src_lib/fourier.cpp`, `series_product`.
- **Before:** in builds that use a BLAS for the Fourier series' matrix product, an empty time grid passed a leading
  dimension of 0 to the BLAS. The BLAS stopped the process. v2 returned empty arrays.
- **Now:** an empty product returns at once, so the results are empty arrays of the right shapes, as in v2.
- **Test:** `emptygrid`.

## Since v1.0.0

### 16. Targets: the person-years of clinical mortality and of the deaths at diagnosis

- **Where:** `src/main.py`: `corrected_denominators`, `seer17_person_years`, `apply_corrected_denominators`,
  called from `main_rates`. Also `verification/h2.py`, `exact_targets`.
- **v1.0.0 and v2:** each family was divided by the population of its own data file, and two of those populations
  did not belong to the counts.
  - Clinical mortality used the incidence-based mortality file's person-years of the years of death 2000 to 2021.
    But only cases diagnosed from 2004 have a stage, so the deaths in the targets come from those cases.
  - The deaths at diagnosis, SEER 17's cases diagnosed in 2004 to 2021, used the populations stored in the pickle.
    Those cover a larger area: by single year of age they match the SEER 22 area (`data/README.md`).
  - Both families' denominators were therefore too large, and their targets too low.
- **Now:** all four families share one denominator, SEER 17's person-years for 2004 to 2021 by single year of age,
  from the rates file. These are the clinical incidence targets' own person-years, which are unchanged.
  - `main.corrected_denominators = False` gives the old targets.
  - Refitted on the corrected targets, the rates of progression and diagnosis out of the preclinical states are 36%
    to 45% lower, and the death rates after diagnosis 19% to 66% higher (Table 2).
- **Test:** `denominators`, on synthetic data. Its checks:
  - the incidence targets are unchanged;
  - every mortality and deaths-at-diagnosis target is count / the rates file's 2004 to 2021 person-years, computed
    from the files directly;
  - `False` reproduces the old denominators;
  - a missing year of diagnosis raises.

  On the SEER data the targets were also checked against a table built independently of `main.py`, and the
  original targets against the published tables. That check is verified on the SEER data and is not in the public
  suite.
