from itertools import product
from multiprocessing import Pool
import argparse

import progressbar
from scipy.ndimage import gaussian_filter1d

from model import *

np.set_printoptions(linewidth=300, edgeitems=10, precision=4, threshold=30)
pd.options.display.width = 0
pd.options.display.min_rows = 20
pd.options.display.max_rows = 20

# main_rates writes the target table ../output/<cancer>_rates.csv; parallel runs (limis_runs.py) set this False in all
# but one process, because concurrent writes of the same file can corrupt it even when the content is identical
write_rates_table = True
# LIMIS prior half-width floor on the log scale: None gives limis_n_se SE throughout (task 9); 0.3, v2's half-width,
# gives max(0.3, limis_n_se SE) per parameter (task 9b, the wide prior; limis_runs.py --min-half-width)
limis_min_half_width = None
# LIMIS prior: fixed natural-scale bounds for chosen parameters, {name as in the bounds table: (lower, upper)}, e.g.
# {"Incidence_1": (7.0, 8.0), "Incidence_2": (130.0, 140.0)} for the Weibull shape and scale (task 9c;
# limis_runs.py --fixed-bounds); the other parameters keep the rule above. None (the default): none fixed
limis_fixed_bounds = None
# Denominators of the clinical mortality and deaths-at-diagnosis targets (entries 79 and 81). True (the default since
# entry 81, the paper's targets): both are SEER 17's person-years of 2004 to 2021 by single year of age, from the rates
# file, the incidence targets' own denominators. False: the original analysis's targets, each file's own population: the
# incidence-based mortality file's person-years of 2000 to 2021, and the pickle's of 2004 to 2021, which equal the
# SEER 22 area's (SEER 17 plus Idaho, Illinois, Massachusetts, New York and Texas), about 1.8 times SEER 17's
corrected_denominators = True
corrected_years = (2004, 2021)


def seer17_person_years(df_rates: pd.DataFrame) -> pd.Series:
    """SEER 17's person-years of corrected_years by single year of age (the rates file's age labels), from the rates
    file's single-year rows. In a rate session every stage row of a year and age carries the same population, so each
    (year, age) is counted once."""
    years = pd.to_numeric(df_rates[diag_col], errors="coerce")
    rows = df_rates.loc[years.between(*corrected_years)]
    pops = rows.groupby([diag_col, age_col_rates])["Population"].agg(["first", "nunique"])
    if (pops["nunique"] != 1).any():
        raise ValueError("rates file: the stage rows of a year and age carry different populations")
    if pops.index.get_level_values(0).nunique() != corrected_years[1] - corrected_years[0] + 1:
        raise ValueError(f"rates file: not every year of {corrected_years} is present")
    return pops["first"].groupby(level=age_col_rates).sum()


def apply_corrected_denominators(df_rates, df_mortality_c, df_doa_c):
    """With corrected_denominators, replace the Population columns of the mortality rows and the deaths-at-diagnosis
    rows (modified in place) with seer17_person_years(df_rates), matched on the age label. The counts, and so the
    targets' Poisson SE sqrt(count) / population, are otherwise computed as before."""
    if not corrected_denominators:
        return
    py = seer17_person_years(df_rates)
    for df, age_col in [(df_mortality_c, "Age at death with single ages and 85+"), (df_doa_c, age_col_rates)]:
        new = df[age_col].map(py)
        if new.isna().any():
            raise ValueError(f"no SEER 17 person-years for ages {sorted(df.loc[new.isna(), age_col].unique())}")
        df["Population"] = new.astype(np.int64)


def main_pickle():
    t_start = time.perf_counter()
    single = pd.read_csv("../data/single-multiple/single.csv")
    t_end = time.perf_counter()
    print(single)
    single.to_pickle("../data/single-multiple/single.pkl")
    print(t_end - t_start)


def main_doa():
    site = "Urinary Bladder"
    year_col = "Year of death recode"
    # cols_group = [stage_col, year_col, age_col_rates]
    # cols_group = [stage_col, age_col_rates]
    cols_group = ["outcome", stage_col, age_col_rates]

    t_start = time.perf_counter()
    df_single: pd.DataFrame = pd.read_pickle("../data/single-multiple/single.pkl")
    t_end = time.perf_counter()
    print(t_end - t_start)

    # Outcome classifications:
    # 0: cancer diagnosis, still alive
    # 1: cancer diagnosis, dead from this cancer
    # 2: cancer diagnosis, dead from other causes
    df_single["outcome"] = -1
    df_single.loc[df_single["Vital status recode (study cutoff used)"] == "Alive", "outcome"] = 0
    df_single.loc[df_single["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)", "outcome"] = 1
    df_single.loc[df_single["SEER other cause of death classification"] == "Dead (attributable to causes other than this cancer dx)", "outcome"] = 2
    df_single = df_single.loc[df_single["outcome"] >= 0]
    df_single.reset_index(inplace=True, drop=True)

    df_single = df_single.loc[df_single[site_col] == site]
    df_single = df_single.loc[df_single["Survival months"] != "Unknown"]
    df_single["Survival months"] = df_single["Survival months"].astype(int)
    df_single = df_single.loc[df_single[stage_col] != "Unknown/unstaged"]
    df_single = df_single.loc[df_single["outcome"] > 0]
    df_single = df_single.loc[df_single["Survival months"] <= months_cutoff]
    df_single[year_col] = df_single[year_col].astype(int)
    df_single.reset_index(inplace=True, drop=True)
    counts_doa: pd.Series = df_single.groupby(cols_group).size().sort_index()

    df_pops = pd.read_csv("../data/single-multiple/populations.csv", low_memory=False)
    df_pops = df_pops.loc[df_pops["Year"] != "1990-2023"]
    df_pops["Year"] = df_pops["Year"].astype(int)
    df_pops = df_pops.loc[df_pops["Year"] >= 2004]
    df_pops = df_pops.loc[df_pops["Year"] <= 2021]
    df_pops.rename({"Year": year_col}, axis=1, inplace=True)

    age_col_90 = "Age recode with single ages and 90+"
    df_pops[age_col_rates] = df_pops[age_col_90]
    for y in range(85, 90):
        df_pops.loc[df_pops[age_col_90] == f"{y:02d} years", age_col_rates] = "85+ years"
    df_pops.loc[df_pops[age_col_90] == f"90+ years", age_col_rates] = "85+ years"
    dfs = []
    print(df_pops)
    for outcome, stage in product(df_single["outcome"].unique(), df_single[stage_col].unique()):
        df = df_pops.copy()
        df["outcome"] = outcome
        df[stage_col] = stage
        dfs.append(df)
    df_pops = pd.concat(dfs, axis=0)
    pops = df_pops.groupby(cols_group)["Count"].sum().sort_index()

    df = pd.DataFrame(pops)
    df.rename({"Count": "Population"}, axis=1, inplace=True)
    df["Count"] = counts_doa
    df["Count"] = df["Count"].fillna(0).astype(int)

    df.to_pickle(f"../data/single-multiple/doa_{site.lower()}.pkl")
    df.to_csv(f"../data/single-multiple/doa_{site.lower()}.csv")
    print(df)


def get_target_incidence(df_single_c: pd.DataFrame):
    stage = df_single_c[stage_col].unique()[0]
    totals = df_single_c.loc[df_single_c[stage_col] == stage].groupby(age_col_rates)["Population"].sum()
    totals = totals.to_dict()

    target = dict()
    for a in age_groups:
        mask = df_single_c[age_col_rates] == a
        count = df_single_c.loc[mask, "Count"].sum()
        total = totals.get(a, 0)
        if total == 0:
            mu_i = 0.0
            sigma_i = infinity
        else:
            mu_i = count/total
            sigma_i = np.sqrt(count)/total if count > 0 else infinity
        target[a] = (mu_i, sigma_i)
    return target


def get_target_mortality(df_single_c: pd.DataFrame):
    stage = df_single_c[stage_col].unique()[0]
    age_col = "Age at death with single ages and 85+"
    totals = df_single_c.loc[df_single_c[stage_col] == stage].groupby(age_col)["Population"].sum()
    totals = totals.to_dict()

    target = dict()
    for a in age_groups:
        mask = df_single_c[age_col] == a
        count = df_single_c.loc[mask, "Count"].sum()
        total = totals.get(a, 0)
        if total == 0:
            mu_i = 0.0
            sigma_i = infinity
        else:
            mu_i = count/total
            sigma_i = np.sqrt(count)/total if count > 0 else infinity
        target[a] = (mu_i, sigma_i)
    return target


def get_target_doa(df_doa_c: pd.DataFrame):
    stage = df_doa_c[stage_col].unique()[0]
    age_col = "Age recode with single ages and 85+"
    totals = df_doa_c.loc[df_doa_c[stage_col] == stage].groupby(age_col)["Population"].sum()
    totals = totals.to_dict()

    target = dict()
    for a in age_groups:
        mask = df_doa_c[age_col] == a
        count = df_doa_c.loc[mask, "Count"].sum()
        total = totals.get(a, 0)
        if total == 0:
            mu_i = 0.0
            sigma_i = infinity
        else:
            mu_i = count/total
            sigma_i = np.sqrt(count)/total if count > 0 else infinity
        target[a] = (mu_i, sigma_i)
    return target


def newton_backtracking_step(model_c, f0, grad, hess):
    """Full saddle-free Newton step with backtracking (newton_damped = False). The direction is the package's
    saddle-free step d = -|H|^-1 g. If a parameter would move by more than newton_max_step (log scale), the whole
    step is scaled down to that, keeping its direction. Then a = 1, 1/2, 1/4, ... until the Armijo condition
    f(theta + a d) <= f(theta) + 1e-4 a g.d holds, at most newton_max_halvings halvings. Returns the eigenvalues of
    H and a, or a = None if no step length decreased the objective enough (the parameters are then unchanged)."""
    theta = model_c.get_parameters().copy()
    eig = model_c.saddle_free_newton_step(hess, grad)
    d = model_c.get_parameters() - theta
    if newton_max_step is not None and np.abs(d).max() > newton_max_step:
        d *= newton_max_step / np.abs(d).max()
    slope = grad @ d
    order = model_c.eval_order
    model_c.eval_order = 0
    a = 1.0
    for _ in range(newton_max_halvings + 1):
        model_c.set_parameters(theta + a * d)
        f, _, _ = model_c.eval_targets()
        if np.isfinite(f) and f <= f0 + 1e-4 * a * slope:
            break
        a *= 0.5
    else:
        model_c.set_parameters(theta)
        a = None
    model_c.eval_order = order
    return eig, a


def main_rates(arguments):
    seed, cancer, df_single_c, survival_function = arguments
    survival_weights = survival_function.cumsum() * dt
    window = int(1.0/dt)
    # survival_filter = savgol_filter(survival_function, window, 2)
    survival_filter = gaussian_filter1d(survival_function, window, mode='nearest')
    survival_grad = -np.gradient(survival_filter, dt)

    print("survival:", survival_function)
    print("window:", window)
    print("smoothed:", survival_filter)
    print("gradient:", survival_grad)
    print()

    np.random.seed(seed)

    df_mortality_c = pd.read_csv(f"../data/single-multiple/mortality_{cancer.replace(' ', '_').lower()}.csv")
    df_mortality_c = df_mortality_c.loc[df_mortality_c["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)"].copy()
    df_doa_c: pd.DataFrame = pd.read_pickle(f"../data/single-multiple/doa_{cancer.lower()}.pkl")
    df_doa_c.reset_index(drop=False, inplace=True)
    apply_corrected_denominators(df_single_c, df_mortality_c, df_doa_c)

    filename_c = f"../output/{cancer.lower()} mle.npy"
    filename_hot_start_c = f"../output/{cancer.lower()}.npy"  # Table 2 estimate; never overwritten

    df_single_c[stage_col] = df_single_c[stage_col].fillna("Blank(s)")
    df_single_c = df_single_c.loc[~df_single_c[stage_col].isin(['Blank(s)', 'Unknown/unstaged'])]
    n_total = df_single_c["Count"].sum()

    print()
    print("#" * 120)
    print(cancer)
    print()
    for stage in sorted(df_single_c[stage_col].unique()):
        print("\t", stage, df_single_c.loc[df_single_c[stage_col] == stage, "Count"].sum() / n_total)
    print()

    print("Incidence Count:", df_single_c["Count"].sum(), "Population:", df_single_c["Population"].sum())
    print("Mortality Count:", df_mortality_c["Count"].sum(), " Population:", df_mortality_c["Population"].sum())
    print("Incidence per 100,000:", df_single_c["Count"].sum()/df_single_c["Population"].sum() * 1e5)
    print("Mortality per 100,000:", df_mortality_c["Count"].sum()/df_mortality_c["Population"].sum() * 1e5)
    print()

    weibull_params = Weibull_map.get(cancer, None)
    if weibull_params is None:
        shape, scale = Weibull_map["Default"]
    else:
        shape, scale = weibull_params
    model_c = SingularCancer(cancer, shape, scale)

    targets = {k: dict() for k in target_names}
    dfs = {k: [] for k in target_names}
    fmt = '{:,.3g}'
    # closing cut point of the last fitted interval (80-81): kept, with infinite sigma, so the age-80
    # targets stay in the objective; it is not fitted itself and not written to the target table
    age_close = "81 years"
    for name, stages in model_c.stage_groups.items():
        df_single_c_stage = df_single_c.loc[df_single_c[stage_col].isin(stages)]
        df_mortality_c_stage = df_mortality_c.loc[df_mortality_c[stage_col].isin(stages)]
        df_doa_c_stage = df_doa_c.loc[df_doa_c[stage_col].isin(stages)]
        targets["clinical incidence"][name] = get_target_incidence(df_single_c_stage)
        targets["clinical mortality"][name] = get_target_mortality(df_mortality_c_stage)
        targets["preclinical mortality"][name] = get_target_doa(df_doa_c_stage.loc[df_doa_c_stage["outcome"] == 1])
        targets["preclinical other mortality"][name] = get_target_doa(df_doa_c_stage.loc[df_doa_c_stage["outcome"] == 2])

        print(name, "<-", stages)
        print(df_single_c_stage)

        for k in ages_remove:
            for target in targets:
                if k == age_close:
                    targets[target][name][k] = (targets[target][name][k][0], infinity)
                else:
                    del targets[target][name][k]
        for t in target_names:
            df_t = pd.DataFrame.from_dict(targets[t][name]).T.drop(index=age_close) * 1e5
            df_t: pd.Series = df_t[0].map(fmt.format) + " (" + df_t[1].map(fmt.format) + ")"
            df_t = df_t.to_frame(name=name)
            df_t.columns = pd.MultiIndex.from_product([[t], df_t.columns])
            dfs[t].append(df_t)
    dfs_list = []
    for dfs_t in dfs.values():
        dfs_list.extend(dfs_t)
    df = pd.concat(dfs_list, axis=1)
    print(df)
    rates_path = f"../output/{cancer.lower()}_rates.csv"
    if write_rates_table:
        df.to_csv(rates_path)
        print("target table written:", rates_path)
    else:
        print("target table not written (write_rates_table = False):", rates_path)
    
    model_c.targets = targets
    model_c.survival_function = survival_function
    model_c.survival_grad = survival_grad
    model_c.survival_weights = survival_weights

    if hot_start_load:
        params = model_c.get_parameters().copy()
        # no fallback: a missing or unreadable file raises instead of silently starting from the defaults
        new_params: np.ndarray = np.load(filename_hot_start_c)
        # new_params = np.array(new_params.tolist() + [0.0])
        # new_params = np.array(new_params.tolist() + [0.0, np.log(2.2)])
        # new_params[0] += np.log(1.0)
        # new_params[1] += np.log(0.6)
        # new_params[2:7] += np.log(1e-2)
        # new_params[3] += np.log(0.8)
        # new_params[5] += np.log(3.0)
        # new_params[6] += np.log(2.5)
        # new_params[8] += np.log(1.0)
        model_c.set_parameters(new_params)

    # r_H_U1_shape  r_H_U1_scale  r_U1_T1       r_U1_U2       r_U2_T2       r_U2_U3       r_U3_T3
    # 6.595584e+00  5.386969e-15  6.739468e-01  1.000000e-01  2.958004e-01  2.000000e-01  8.000000e-01
    # shape = 6.595584e+00
    # scale_factor = 5.386969e-15
    # scale = np.power(scale_factor, -1.0/shape)
    # model_c.set_parameters(np.log([shape, scale, 1.000000e-01, 6.739468e-01, 0.1, 2.000000e-01, 2.958004e-01, 0.1, 8.000000e-01, 0.1, 0.01107465, 0.06453399, 0.1599436]))

    print("initial model:")
    model_c.print_parameters()
    model_c.eval_order = 0
    model_c.options.verbose = verbose
    error_val, _, _ = model_c.eval_targets(plot=True)
    print("error:", error_val)

    widgets = [
        ' [', progressbar.Timer(), '] ',
        progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        # progressbar.GranularBar(), ' ',
        # progressbar.Percentage(), ' ',
        progressbar.Variable("params"), ' ',
        progressbar.Variable("error", width=12, precision=12),
    ]

    t_start = time.perf_counter()
    model_c.eval_order = 1
    with progressbar.ProgressBar(max_value=epochs_grad, enable_colors=False, widgets=widgets, redirect_stdout=True) as bar:
        for epoch in range(epochs_grad):
            if verbose:
                print()
                print("-" * 120)
                print("grad epoch", epoch + 1)
                model_c.print_parameters()

            error_val, error_grad, _ = model_c.eval_targets()
            params = model_c.get_parameters()

            if verbose:
                print("\nerror:", error_val)
                print("gradient:", error_grad)
            bar.update(epoch, params=str(np.exp(model_c.get_parameters())), error=error_val)

            factor = np.power(epoch+1, 0.00)
            error_grad[0:2] *= lr_weibull * factor
            error_grad[2:] *= lr_exp * factor
            model_c.parameter_step(-1.0 / np.sqrt(epoch + 1) * error_grad)

    widgets += [' ', progressbar.Variable("min_eig", width=12, precision=12)]
    model_c.eval_order = 2
    with progressbar.ProgressBar(max_value=epochs_hess, enable_colors=False, widgets=widgets, redirect_stdout=True) as bar:
        for epoch in range(epochs_hess):
            if verbose:
                print()
                print("-" * 120)
                print("hess epoch", epoch + 1)
                model_c.print_parameters()

            error_val, error_grad, error_hess = model_c.eval_targets()
            params = model_c.get_parameters()

            if epoch == epochs_hess-1:
                model_c.options.verbose = True

            if model_c.options.verbose:
                print("\nerror:", error_val)
                print("gradient:", error_grad)

            if newton_damped:  # v2's schedule
                factor = np.power(epoch+1, 0.5)
                eig = model_c.saddle_free_newton_step(error_hess, lr_newton/factor * error_grad)
            else:
                if np.linalg.norm(error_grad) < newton_tol:
                    print(f"Newton: converged at iteration {epoch + 1}, gradient norm {np.linalg.norm(error_grad):.3g}"
                          f" < {newton_tol:g}, objective {error_val:.10f}")
                    break
                eig, step_length = newton_backtracking_step(model_c, error_val, error_grad, error_hess)
                print(f"Newton iteration {epoch + 1}: objective {error_val:.10f}, gradient norm "
                      f"{np.linalg.norm(error_grad):.3g}, step length {step_length}")
                if step_length is None:
                    print(f"Newton: stopped at iteration {epoch + 1}, no step decreased the objective after "
                          f"{newton_max_halvings} halvings")
                    break
            # eig = model_c.newton_raphson_step(error_hess, lr_newton * factor * error_grad)
            # eig = model_c.newton_flow_step(error_hess, error_grad, t=1e-15 * factor)
            bar.update(epoch, params=str(np.exp(params)), error=error_val, min_eig=eig[0])

    # save the last epoch's parameters (where the last gradient was evaluated), not the epoch with the lowest
    # objective: near the optimum quadrature noise in the objective decides that epoch, and it can be unconverged
    if hot_start_save and epochs_grad + epochs_hess > 0:
        np.save(filename_c, params)
        print("saved the last epoch's parameters; gradient norm there:", np.linalg.norm(error_grad))

    model_c.options.verbose = True
    print()
    print("-" * 120)
    print("final model:")
    model_c.print_parameters()
    model_c.eval_order = 0
    model_c.options.verbose = False
    error_val, _, _ = model_c.eval_targets(plot=True)
    print("error:", error_val)
    print()

    if run_IMIS:
        bounds_names = [risk.name if risk.get_n_params() == 1 else f"{risk.name}_{i+1}"
                        for risk in model_c.risk_indices.keys() for i in range(risk.get_n_params())]
        unknown = set(limis_fixed_bounds or {}) - set(bounds_names)
        if unknown:
            raise ValueError(f"limis_fixed_bounds: unknown parameters {sorted(unknown)}; names: {bounds_names}")
        fixed = {bounds_names.index(k): v for k, v in (limis_fixed_bounds or {}).items()}
        # prior bounds from the observed information at the MLE file; raises if it is missing or not positive definite
        limis_center, limis_half_width, limis_se = model_c.limis_bounds(filename_c,
                                                                        min_half_width=limis_min_half_width,
                                                                        fixed_bounds=fixed)
        limis_mle = np.load(filename_c)  # the box centre, except for parameters with fixed bounds
        floor = "" if limis_min_half_width is None else f", at least {limis_min_half_width:g}"
        fixed_text = "".join(f"; {k} fixed at [{lo:g}, {hi:g}]" for k, (lo, hi) in (limis_fixed_bounds or {}).items())
        print(f"LIMIS prior: uniform, centre {filename_c}, half-width {model_c.limis_n_se:g} SE{floor} (log scale)"
              f"{fixed_text}; runs in {model_c.limis_dir}")
        if load_IMIS:
            print("(these bounds follow the current settings and are used only when sampling; a summary of saved runs "
                  "uses the runs' own prior box, which IMIS2 reads from their components files and prints)")
        print("centre:", np.exp(limis_center))
        print("bounds:", np.exp(limis_center - limis_half_width), np.exp(limis_center + limis_half_width))
        bounds_table = pd.DataFrame({"MLE": np.exp(limis_mle), "half-width (log)": limis_half_width,
                                     "lower": np.exp(limis_center - limis_half_width),
                                     "upper": np.exp(limis_center + limis_half_width),
                                     "half-width (SE)": limis_half_width / limis_se}, index=bounds_names)
        if fixed:  # each bound's distance from the MLE, in SEs (log scale); the box is not centred at the MLE
            bounds_table["MLE - lower (SE)"] = (limis_mle - (limis_center - limis_half_width)) / limis_se
            bounds_table["upper - MLE (SE)"] = (limis_center + limis_half_width - limis_mle) / limis_se
            bounds_table["rule"] = ["fixed" if j in fixed else "" for j in range(len(bounds_names))]
        print(bounds_table.to_string())
        quantiles_params, quantiles_targets = model_c.IMIS2(seed, priors=False, center=limis_center,
                                                            half_width=limis_half_width, se=limis_se, mle=limis_mle)

    if epochs_hess > 0:
        cov = np.linalg.pinv(error_hess)
        sigmas = np.sqrt(np.diag(cov))
        print(np.exp(params - 1.96*sigmas))
        print(np.exp(params))
        print(np.exp(params + 1.96*sigmas))
        print()

    if run_IMIS:
        print(np.exp(quantiles_params))

    names = []
    for risk in model_c.risk_indices.keys():
        name = risk.name
        n_params = risk.get_n_params()
        if n_params == 1:
            names.append(name)
        else:
            names.extend(f"{name}_{i+1}" for i in range(n_params))
    if run_IMIS:
        # MLE column: the MLE file and its information-matrix SEs (limis_bounds), the centre of the LIMIS prior unless
        # limis_fixed_bounds moves it; not params and sigmas, which with epochs_hess = 0 are the constructor's defaults
        # and undefined
        df = pd.DataFrame({
            "mle_med": limis_mle, "mle_low": limis_mle - 1.96*limis_se, "mle_high": limis_mle + 1.96*limis_se,
            "imis_med": quantiles_params[1, :], "imis_low": quantiles_params[0, :], "imis_high": quantiles_params[2, :]},
            index=names)
        df = np.exp(df)

        def fmt_4dp(x):  # fixed point, four digits after the decimal point, trailing zeros kept: 7.4014, 0.1700
            return f"{x:.4f}"
        for col in ["mle", "imis"]:
            cols = [df[f"{col}_{c}"].map(fmt_4dp) for c in ["med", "low", "high"]]
            df[f"{col}_str"] = cols[0] + " (" + cols[1] + ", " + cols[2] + ")"
        df = df[["mle_str", "imis_str"]]
        print(df)
        df.to_csv(f"{model_c.limis_dir}/CIs.csv")

    t_end = time.perf_counter()

    if True:
        model_c.print_times(n_digits=3)

    # the current figure is the final model's fit only without LIMIS; IMIS2 draws and saves its own figures, and
    # the last of them (the correlogram) must not be saved as ../output/<cancer>.pdf
    if not run_IMIS:
        save_plot(cancer.lower())
    exit()


def main_rates_all(seed):
    # df_rates: pd.DataFrame = pd.read_csv("../data/single-multiple/rates.csv")
    df_rates: pd.DataFrame = pd.read_csv("../data/single-multiple/rates_urinary_bladder.csv")
    df_rates[site_col_rates] = "Urinary Bladder"
    print(df_rates)

    temp = df_rates[site_col_rates].map(map_groups)
    mask = temp.isna()
    temp.loc[mask] = df_rates.loc[mask, site_col_rates]
    df_rates[site_col_rates] = temp
    # df_rates = df_rates.loc[df_rates[diag_col] == "2016-2020"]
    df_rates = df_rates.loc[df_rates[diag_col] != "2000-2021"]
    # df_rates = df_rates.loc[df_rates[diag_col] == "2000-2021"]

    single_sorted = df_rates.groupby(site_col_rates)["Count"].sum()
    single_sorted.sort_values(ascending=False, inplace=True)
    # cancers = list(single_sorted.index[:n_cancers]) + ["Other cancer"]
    cancers = list(single_sorted.index[:n_cancers])
    df_rates.loc[~df_rates[site_col_rates].isin(cancers), site_col_rates] = "Other cancer"
    n_cases = df_rates["Count"].sum()
    print(single_sorted.cumsum()/n_cases)
    print("Cancers:", single_sorted[:n_cancers].sum()/n_cases, list(cancers))

    df_single_grouped = df_rates.groupby(site_col_rates)
    survival_function = get_survival_function()

    for cancer in cancers:
        df_single_c: pd.DataFrame = df_single_grouped.get_group(cancer).copy()
        main_rates((seed, cancer, df_single_c, survival_function))
    # threads = 10
    # arguments = [(cancer, df_single_grouped.get_group(cancer).copy(), survival_function.copy()) for cancer in cancers]
    # with Pool(processes=threads) as p:
    #     p.map(main_single, arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args, _ = parser.parse_known_args(None)
    params = vars(args)

    # main_pickle()
    # main_doa()
    main_rates_all(seed=params["seed"])
