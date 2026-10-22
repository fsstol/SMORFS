import glob
import os
import pickle
import re

from scipy.special import logsumexp
from scipy.stats import multivariate_t, multivariate_normal

from parameters import *
from semi_markov import *
from correlogram import *


class SingularCancer(SemiMarkov):
    # Interval targets: True integrates S_US * arrival density exactly, from the cumulative arrival at the age
    # cut points and the half years where the nearest-age S_US steps; False is v2's left sum over the grid t
    cumulative_targets = True
    # LIMIS prior: uniform on the log scale, centred at the MLE, half-width limis_n_se information-matrix SEs
    # (limis_bounds: optionally at least min_half_width, or fixed natural-scale bounds for chosen parameters)
    limis_n_se = 4.0
    # posterior (or prior) draws, resampled by weight, for the figures and the target bands; the parameter
    # quantiles use every sample with its weight instead (limis_quantiles)
    limis_n_draws = 10000
    # folder of the LIMIS runs: samples/ and the summary figures and tables; the wide-prior runs use
    # ../output/IMIS2_wide, so the +-4 SE runs in ../output/IMIS2 stay for the sensitivity comparison
    limis_dir = "../output/IMIS2"
    # True: IMIS2 also draws the histograms and the correlogram with axes set to the prior bounds (exp(center +-
    # half_width)) instead of matplotlib's autoscale of the draws, saved as ..._hist_bounds_N, ..._corr_bounds_N
    limis_axes_to_bounds = False
    # figures only: targets based on this many events (inclusive range; events = (target / SE)^2 = the count) are not
    # drawn, neither point nor error bar, as SEER's data-use agreement requires for statistics based on counts of 1 to
    # 4 (README). Targets with no events are drawn as before. The objective, the targets and every model curve and
    # band are unchanged. None draws every target
    plot_suppress_events = (1, 4)

    def __init__(self, name: str, shape, scale):
        super().__init__(name)
        self.options.verbose = verbose

        self.H0 = State(f"H0")
        self.U1 = State(f"U1")
        self.U2 = State(f"U2")
        self.U3 = State(f"U3")
        self.T1 = State(f"T1")
        self.T2 = State(f"T2")
        self.T3 = State(f"T3")

        self.D1 = State(f"D1")
        self.D2 = State(f"D2")
        self.D3 = State(f"D3")

        self.E1 = State(f"E1")
        self.E2 = State(f"E2")
        self.E3 = State(f"E3")

        for s in [self.D1, self.D2, self.D3, self.E1, self.E2, self.E3]:
            s.terminal = True
        for s in [self.H0, self.U1, self.U2, self.U3, self.T1, self.T2, self.T3, self.D1, self.D2, self.D3, self.E1, self.E2, self.E3]:
            self.add_state(s)
        self.set_initial(self.H0)

        # self.R_inc = ExponentialStable("Incidence", rate=lambda0)
        self.R_inc = WeibullStable("Incidence", k=shape, s=scale)
        # self.R_inc = Weibull("Incidence", k=shape, s=scale)
        # self.R_inc = PiecewiseExponentialStable("Incidence", 5, np.ones(5), np.arange(0, 100.0, 20.0))

        self.R_U1_U2 = ExponentialStable("U1->U2", rate=6e2*lambda0)
        self.R_U1_T1 = ExponentialStable("U1->T1", rate=1e3*lambda0)

        self.R_U2_U3 = ExponentialStable("U2->U3", rate=3e5*lambda0)
        self.R_U2_T2 = ExponentialStable("U2->T2", rate=3e5*lambda0)

        self.R_U3_T3 = ExponentialStable("U3->T3", rate=1e6*lambda0)

        # Exponential cancer mortality by stage
        r_T1_D1 = 0.03
        r_T2_D2 = 0.10
        r_T3_D3 = 0.25

        self.R_T1_D1 = ExponentialStable("T1->D1", rate=r_T1_D1)
        self.R_T2_D2 = ExponentialStable("T2->D2", rate=r_T2_D2)
        self.R_T3_D3 = ExponentialStable("T3->D3", rate=r_T3_D3)

        self.R_U1_E1 = ExponentialStable("U1->D1", rate=1e5*lambda0)
        self.R_U2_E2 = ExponentialStable("U2->D2", rate=1e5*lambda0)
        self.R_U3_E3 = ExponentialStable("U3->D3", rate=1e2*lambda0)

        self.H0.add_transition(self.U1, self.R_inc)

        self.U1.add_transition(self.U2, self.R_U1_U2)
        self.U1.add_transition(self.T1, self.R_U1_T1)

        self.U2.add_transition(self.U3, self.R_U2_U3)
        self.U2.add_transition(self.T2, self.R_U2_T2)

        self.U3.add_transition(self.T3, self.R_U3_T3)

        self.T1.add_transition(self.D1, self.R_T1_D1)
        self.T2.add_transition(self.D2, self.R_T2_D2)
        self.T3.add_transition(self.D3, self.R_T3_D3)

        self.U1.add_transition(self.E1, self.R_U1_E1)
        self.U2.add_transition(self.E2, self.R_U2_E2)
        self.U3.add_transition(self.E3, self.R_U3_E3)

        self.update()

        self.stage_groups = {
            "Early": ["In situ", "Localized"],
            "Medium": ["Regional"],
            "Late": ["Distant"],
        }
        self.stage_states = {
            "clinical incidence": {
                "Early": self.T1,
                "Medium": self.T2,
                "Late": self.T3,
            },
            "clinical mortality": {
                "Early": self.D1,
                "Medium": self.D2,
                "Late": self.D3,
            },
            "preclinical mortality": {
                "Early": self.E1,
                "Medium": self.E2,
                "Late": self.E3,
            },
            "preclinical other mortality": {
                "Early": self.U1,
                "Medium": self.U2,
                "Late": self.U3,
            },
        }
        self.stage_colors = {
            "Early": "g",
            "Medium": "y",
            "Late": "r",
        }

    def calc_CIs(self, mus, sigmas):
        # error bar lengths below and above each target: the 95% interval mu +- 1.96 SE, which matches the objective's
        # normal likelihood (v2: mu exp(+-1.96 SE/mu), an interval on the log scale)
        sigmas = 1.96 * np.array(sigmas)
        return np.array([sigmas, sigmas])

    def suppress_low_counts(self, target_inc, ages, mus, sigmas):
        """mus (n) and error-bar lengths sigmas (2 x n) for drawing, with NaN, which matplotlib leaves undrawn, at the
        targets whose event count is in plot_suppress_events."""
        mus, sigmas = np.array(mus, dtype=float), np.array(sigmas, dtype=float)
        if self.plot_suppress_events is not None:
            lo, hi = self.plot_suppress_events
            for i, a in enumerate(ages):
                mu, se = target_inc[a]
                events = int(round((mu / se) ** 2)) if 0.0 < se < 1e19 else 0
                if lo <= events <= hi:
                    mus[i] = np.nan
                    sigmas[:, i] = np.nan
        return mus, sigmas

    def plot_target(self, ax_idx, target_type, stage, target_inc, target_inc_ages, y_vals):
        fig = plt.gcf()
        ax = fig.get_axes()[ax_idx]
        plt.sca(ax)
        plt.title(f"{self.name} {target_type}")
        mus = np.array([1e5 * target_inc[a][0] for a in target_inc_ages[:-1]])
        sigmas = [target_inc[a][1] for a in target_inc_ages[:-1]]
        sigmas = np.array([1e5 * s if s < 1e9 else 0.0 for s in sigmas])
        sigmas = self.calc_CIs(mus, sigmas)
        mus, sigmas = self.suppress_low_counts(target_inc, target_inc_ages[:-1], mus, sigmas)

        x = [(age_groups[a][0]+age_groups[a][1])/2 for a in target_inc_ages[:-1]]
        plt.plot(x, 1e5 * y_vals, self.stage_colors[stage], linewidth=1.0, label="Model - " + stage)
        plt.errorbar(x=x, y=mus, yerr=sigmas, capsize=3, fmt=".--", color=self.stage_colors[stage], linewidth=1.0, markersize=3.0, ecolor=self.stage_colors[stage], label="Target - " + stage)
        plt.xlabel("Age")
        plt.ylabel("Rate per 100,000")

    def eval_targets(self, plot=False, return_targets=False):
        age_idxs = {k: np.argmin(np.abs(t-t_i[0])) for k, t_i in age_groups.items()}
        p = self.n_params
        errors = [0.0]*3
        y = {target_type: dict() for target_type in self.targets.keys()}

        if plot:
            restart_plot()

        if self.cumulative_targets:
            # break points: the interval cut points, and the half years in between where S_US (life table at the
            # nearest integer age) steps; S_US is constant on each piece, so its integrals below are exact
            cuts = sorted({age_groups[k][0] for target_type in ["clinical incidence", "clinical mortality", "preclinical mortality"]
                           for target_inc in self.targets[target_type].values() for k in target_inc})
            t_break = np.union1d(cuts, np.arange(np.floor(cuts[0]) + 0.5, cuts[-1], 1.0))
            mids = 0.5 * (t_break[1:] + t_break[:-1])
            S_piece = self.survival_function[np.abs(t.reshape((-1, 1)) - mids).argmin(axis=0)]
            idxs = {k: np.searchsorted(t_break, t_i[0]) for k, t_i in age_groups.items()}
            cum_weights = np.concatenate([[0.0], np.cumsum(S_piece * np.diff(t_break))])

            def weighted_cumsum(x):  # integral of S_US * density from t_break[0] to each break point, x cumulative
                return np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(S_piece * np.diff(x, axis=-1), axis=-1)], axis=-1)
        else:
            idxs = age_idxs
            cum_weights = self.survival_weights

            def weighted_cumsum(x):  # left sum of S_US * density over the grid t
                return np.cumsum(self.survival_function * x, axis=-1) * dt

        try:
            if self.cumulative_targets:
                arr = self.get_arrival(t_break, t0, prefix="Arrival", cumulative=True)
            else:
                arr = self.get_arrival(t, t0, prefix="Arrival", cumulative=False)
            prev = self.get_prevalence(t, t0, prefix="Prevalence")  # TODO: merge with arrival
        except:  # handle numerical errors
            errors[0] = infinity
            if self.eval_order >= 1:
                errors[1] = np.zeros(self.n_params, dtype=float)
            if self.eval_order >= 2:
                errors[2] = np.zeros((self.n_params, self.n_params), dtype=float)

            if return_targets:
                for stage in self.stage_groups.keys():
                    for k, target_type in enumerate(self.stage_states.keys()):
                        # target_inc = self.targets[target_type][stage]
                        # n_ages = len(target_inc.keys())
                        n_ages = len(ages_eval)
                        y[target_type][stage] = np.full(n_ages, fill_value=infinity, dtype=float)
                return errors, y
            else:
                return errors

        # no background: y_arr(t) density
        # with background: y_arr(t) * S(t)/S_i(t) density
        # total alive: S(t)
        # proportion: y_arr(t) * S(t)/S_i(t) / S(t) = y_arr(t)/S_i(t)

        for stage in self.stage_groups.keys():
            for k, target_type in enumerate(["clinical incidence", "clinical mortality", "preclinical mortality"]):
                state = self.stage_states[target_type][stage]
                target_inc = self.targets[target_type][stage]
                target_inc_ages = list(target_inc.keys())
                age_idxs_stage = [idxs[k] for k in target_inc_ages]
                n_ages = len(age_idxs_stage)-1
                y_arr: EvalResult = arr[state]

                div = y_arr
                # div_inc = y_arr_inc
                # survival_function = 1.0

                # LHS values
                target_model_val = weighted_cumsum(div.value)
                target_model_val = target_model_val[age_idxs_stage]
                # target_model_val_inc = np.cumsum(survival_function * div_inc.value) * dt
                # target_model_val_inc = target_model_val_inc[age_idxs_stage]
                survival_function_factors = cum_weights[age_idxs_stage]
                survival_function_factors = [1.0/(survival_function_factors[i+1] - survival_function_factors[i]) for i in range(n_ages)]
                y_vals = np.empty(n_ages, dtype=float)
                diffs = np.empty(n_ages, dtype=float)
                for i in range(n_ages):
                    mu_i, sigma_i = target_inc[target_inc_ages[i]]
                    y_val = (target_model_val[i+1] - target_model_val[i]) * survival_function_factors[i]
                    diff = y_val - mu_i
                    y_vals[i] = y_val
                    diffs[i] = diff
                    if target_inc_ages[i] in ages_eval:
                        # errors[0] += 0.5 * diff * diff / (sigma_i*sigma_i) + np.log(np.sqrt(2*np.pi*sigma_i*sigma_i))
                        errors[0] += 0.5 * diff * diff / (sigma_i*sigma_i)

                # LHS gradients
                if self.eval_order >= 1:
                    target_model_grad = weighted_cumsum(div.gradient)
                    target_model_grad = target_model_grad[:, age_idxs_stage]
                    for i in range(n_ages):
                        if target_inc_ages[i] in ages_eval:
                            mu_i, sigma_i = target_inc[target_inc_ages[i]]
                            target_model_grad_i = (target_model_grad[:, i+1] - target_model_grad[:, i]) * survival_function_factors[i]
                            errors[1] += (diffs[i] * target_model_grad_i) / (sigma_i*sigma_i)

                # LHS hessians
                if self.eval_order >= 2:
                    target_model_hess = weighted_cumsum(div.hessian)
                    target_model_hess = target_model_hess[:, :, age_idxs_stage]
                    # FIX: Gauss-Newton term uses the gradient of each interval target, not cumulative gradients
                    g_int = (target_model_grad[:, 1:] - target_model_grad[:, :-1]) * np.asarray(survival_function_factors)
                    for i in range(n_ages):
                        if target_inc_ages[i] in ages_eval:
                            mu_i, sigma_i = target_inc[target_inc_ages[i]]
                            target_model_hess_i = (target_model_hess[:, :, i+1] - target_model_hess[:, :, i]) * survival_function_factors[i]
                            term_i = np.outer(g_int[:, i], g_int[:, i])
                            errors[2] += (diffs[i] * target_model_hess_i + term_i) / (sigma_i*sigma_i)

                if return_targets:
                    y[target_type][stage] = 1e5*y_vals
                if plot:
                    self.plot_target(k, target_type, stage, target_inc, target_inc_ages, y_vals)

            # r_i rate preclinical other mortality stage i:
            # approximation
            # S_-i(t) ~ S(t) => S_-i'(t) ~ S'(t)
            # r_i(t) = p_i(t) * -S_-i'(t)
            #        ~ p_i(t) * -S'(t)

            # exact:
            # S_-i(t) = S(t)/S_i(t)
            # r_i(t) = p_i(t) * -S_-i'(t)
            #        = p_i(t) * -d/dt (S(t)/S_i(t))
            #        = p_i(t) * (S(t)*S_i'(t) - S'(t)*S_i(t))/S_i(t)^2
            # TODO: complete

            target_type = "preclinical other mortality"
            state = self.stage_states[target_type][stage]
            target_inc = self.targets[target_type][stage]
            target_inc_ages = list(target_inc.keys())
            age_idxs_stage = [age_idxs[k] for k in target_inc_ages]
            n_ages = len(age_idxs_stage)-1
            y_prev: EvalResult = prev[state]

            # LHS values
            target_model_val = (self.survival_grad / self.survival_function) * y_prev.value
            target_model_val = target_model_val[age_idxs_stage]
            y_vals = np.empty(n_ages, dtype=float)
            diffs = np.empty(n_ages, dtype=float)
            for i in range(n_ages):
                mu_i, sigma_i = target_inc[target_inc_ages[i]]
                y_val = target_model_val[i]
                diff = y_val - mu_i
                y_vals[i] = y_val
                diffs[i] = diff
                if target_inc_ages[i] in ages_eval:
                    # errors[0] += 0.5 * diff * diff / (sigma_i*sigma_i) + np.log(np.sqrt(2*np.pi*sigma_i*sigma_i))
                    errors[0] += 0.5 * diff * diff / (sigma_i*sigma_i)

            # LHS gradients
            if self.eval_order >= 1:
                target_model_grad = (self.survival_grad / self.survival_function) * y_prev.gradient
                target_model_grad = target_model_grad[:, age_idxs_stage]
                for i in range(n_ages):
                    if target_inc_ages[i] in ages_eval:
                        mu_i, sigma_i = target_inc[target_inc_ages[i]]
                        target_model_grad_i = target_model_grad[:, i]
                        errors[1] += (diffs[i] * target_model_grad_i) / (sigma_i*sigma_i)

            # LHS hessians
            if self.eval_order >= 2:
                target_model_hess = (self.survival_grad / self.survival_function) * y_prev.hessian
                target_model_hess = target_model_hess[:, :, age_idxs_stage]
                # FIX: Gauss-Newton term for a pointwise target
                for i in range(n_ages):
                    if target_inc_ages[i] in ages_eval:
                        mu_i, sigma_i = target_inc[target_inc_ages[i]]
                        target_model_hess_i = target_model_hess[:, :, i]
                        term_i = np.outer(target_model_grad[:, i], target_model_grad[:, i])
                        errors[2] += (diffs[i] * target_model_hess_i + term_i) / (sigma_i*sigma_i)

            k = 3
            if return_targets:
                y[target_type][stage] = 1e5*y_vals
            if plot:
                self.plot_target(k, target_type, stage, target_inc, target_inc_ages, y_vals)

        if return_targets:
            return errors, y
        else:
            return errors

    def limis_bounds(self, mle_path, n_se=None, min_half_width=None, fixed_bounds=None):
        """LIMIS prior bounds: centre at the MLE saved in mle_path, half-width n_se (default limis_n_se) standard
        errors from the observed information there (the Hessian of the objective 0.5 sum z^2, log scale), or
        min_half_width (log scale) where that is wider: half-width = max(min_half_width, n_se SE) per parameter.
        min_half_width None (the default) gives n_se SE throughout.
        fixed_bounds {parameter index: (lower, upper)}, natural scale, replaces that rule for those parameters: their
        box is [log lower, log upper], whose centre is its midpoint, not the MLE; the MLE must lie strictly inside.
        None (the default) fixes none, and the centre is the MLE throughout.
        Raises FileNotFoundError if the file is missing, ValueError if the information matrix is not positive
        definite or a fixed box does not contain the MLE. The model's parameters and eval_order are restored.
        Returns (centre of the box, half_width, se)."""
        n_se = self.limis_n_se if n_se is None else n_se
        mle = np.load(mle_path)
        params, order = self.get_parameters().copy(), self.eval_order
        self.set_parameters(mle)
        self.eval_order = 2
        _, _, hess = self.eval_targets()
        self.set_parameters(params)
        self.eval_order = order
        hess = 0.5 * (hess + hess.T)
        if not np.all(np.isfinite(hess)) or np.linalg.eigvalsh(hess)[0] <= 0.0:
            raise ValueError(f"information matrix at {mle_path} is not positive definite: no LIMIS bounds")
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
        half_width = n_se * se
        if min_half_width is not None:
            half_width = np.maximum(min_half_width, half_width)
        center = mle.copy()
        for j, (lower, upper) in (fixed_bounds or {}).items():
            lo, hi = np.log(lower), np.log(upper)
            if not lo < mle[j] < hi:
                raise ValueError(f"fixed LIMIS bounds ({lower}, {upper}) for parameter {j} do not contain its MLE "
                                 f"{np.exp(mle[j]):.6g}")
            center[j], half_width[j] = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return center, half_width, se

    def IMIS0(self, center=None, half_width=None):
        # prior: uniform on the box |x - center| <= half_width (log scale), from limis_bounds
        if center is None or half_width is None:
            raise ValueError("LIMIS needs its prior bounds (center, half_width): call limis_bounds first")
        x0 = np.asarray(center, dtype=float)
        half_width = np.asarray(half_width, dtype=float)
        assert x0.shape == half_width.shape == (self.n_params,) and np.all(half_width > 0)
        log_prior = -np.sum(np.log(2.0 * half_width))

        Id = np.eye(self.n_params, dtype=float)
        NK = n0 + b*epochs_IMIS
        log_targets = np.full(NK, fill_value=-np.inf, dtype=float)
        log_weights = np.full((1+epochs_IMIS, NK), fill_value=-np.inf, dtype=float)
        log_importance = np.full(NK, fill_value=-np.inf, dtype=float)
        samples = np.full((NK, self.n_params), fill_value=np.nan, dtype=float)

        def eval_log_prior(x):
            if np.all(np.abs(x-x0) <= half_width):
                return log_prior
            else:
                return -np.inf

        # Initial samples
        self.eval_order = 0
        iterator = tqdm(range(n0), smoothing=0.0) if verbose else range(n0)
        for i in iterator:
            xi = x0 + 2.0*half_width*(np.random.random(self.n_params) - 0.5)
            samples[i, :] = xi
            self.set_parameters(xi)
            ll_i, _, _ = self.eval_targets()
            lp_i = eval_log_prior(xi)
            lt_i = -ll_i + lp_i
            log_targets[i] = lt_i
            log_weights[0, i] = lp_i  # row 0: the prior, the first mixture component
        log_importance[:n0] = log_targets[:n0] - log_weights[0, :n0]  # prior samples: weight = likelihood
        log_denom = log_weights[0, :n0]  # mixture density before any epoch: the prior
        weights = np.exp(log_importance[:n0] - log_importance[:n0].max())
        weights /= weights.sum()

        mvts = []
        nk = n0
        for epoch in range(epochs_IMIS):
            # Incremental samples
            # i0 = np.argmax(log_importance[:nk])
            i0 = np.random.choice(range(nk), p=weights)
            mu = samples[i0, :]
            self.set_parameters(mu)
            if verbose:
                print("epoch", epoch, "/ nk =", nk, "/ i0 =", i0, mu)

            sigma = delta*Id
            self.eval_order = 2
            for k in range(steps):
                ll_val, ll_grad, ll_hess = self.eval_targets()
                sigma_t = self.langevin_step(sigma, -ll_grad, -ll_hess, ptime/steps)
                mu_t = self.get_parameters()
                if verbose:
                    print(f"mu {k}:", ll_val, mu, "->", mu_t, "sigma:")
                    print(sigma_t)
                mu = mu_t
                sigma = sigma_t

            mvt = multivariate_t(mu, sigma, deg, allow_singular=True)
            # mvt = multivariate_normal(mu, sigma, allow_singular=True)
            mvts.append(mvt)

            samples_k = mvt.rvs(b).reshape(b, self.n_params)
            samples[nk:nk+b, :] = samples_k

            self.eval_order = 0
            for i in range(b):
                xi = samples_k[i, :]
                self.set_parameters(xi)
                ll_i, _, _ = self.eval_targets()
                lp_i = eval_log_prior(xi)
                lt_i = -ll_i + lp_i
                log_targets[nk + i] = lt_i
                log_weights[0, nk+i] = lp_i

                # old components, new samples
                for j in range(epoch):
                    log_weights[j+1, nk+i] = mvts[j].logpdf(xi)

            nk += b
            # new component, all samples
            log_weights[1+epoch, :nk] = mvts[epoch].logpdf(samples[:nk, :])

            log_weights_red = log_weights[:2+epoch, :nk].copy()  # a copy: the mixture weights below apply once
            log_weights_red[0, :] += np.log(n0/nk)
            log_weights_red[1:2+epoch, :] += np.log(b/nk)
            log_denom = logsumexp(log_weights_red, axis=0)
            log_num = log_targets[:nk]
            log_importance[:nk] = log_num - log_denom
            weights = np.exp(log_importance[:nk] - log_importance[:nk].max())
            weights /= weights.sum()
            # print(weights)
            print("ESS =", 1/(weights*weights).sum())
            print("weight std =", np.std(weights))
            print()
        
        self.limis_components = dict(loc=np.array([m.loc for m in mvts]), shape=np.array([m.shape for m in mvts]),
                                     df=deg, center=x0, half_width=half_width, n0=n0, b=b)
        # return samples, log_targets, log_weights
        return samples, log_targets, log_denom
    
    def eval_quantiles(self, samples, priors):
        n_samples = samples.shape[0]
        filename = f'quantiles_prior_{n_samples}' if priors else f'quantiles_{n_samples}'
        filepath = f'{self.limis_dir}/{filename}.pkl'
        # the bands are computed from these samples every time; the pickle is written as a record and never read,
        # so a file left by an earlier run cannot stand in for them
        params, order = self.get_parameters().copy(), self.eval_order
        self.eval_order = 0
        ys = dict()
        for target_type in self.targets.keys():
            ys[target_type] = dict()
            for stage in self.targets[target_type].keys():
                ys[target_type][stage] = dict()
                ages = self.targets[target_type][stage].keys()
                ys[target_type][stage] = np.empty((n_samples, len(ages)-1))
        iterator = tqdm(range(n_samples), smoothing=0.0, desc='Quantiles') if verbose else range(n_samples)
        for i in iterator:
            self.set_parameters(samples[i, :])
            (error_val, _, _), y = self.eval_targets(plot=False, return_targets=True)
            for target_type in self.targets.keys():
                for stage in self.targets[target_type].keys():
                    ys[target_type][stage][i, :] = y[target_type][stage]
        for target_type in self.targets.keys():
                for stage in self.targets[target_type].keys():
                    ys[target_type][stage] = np.quantile(ys[target_type][stage], [0.025, 0.975], axis=0)
                    print(target_type, stage)
                    print(ys[target_type][stage])
        with open(filepath, 'wb') as handle:
            pickle.dump(ys, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.set_parameters(params)
        self.eval_order = order

        # plot quantiles
        if not priors:
            restart_plot()
            
            def plot_target(prefix, ax_idx, target_type, stage, target_inc, target_inc_ages, y_vals1, y_vals2):
                fig = plt.gcf()
                ax = fig.get_axes()[ax_idx]
                plt.sca(ax)
                plt.title(f"{prefix} {target_type}")
                mus = [1e5 * target_inc[a][0] for a in target_inc_ages[:-1]]
                sigmas = [target_inc[a][1] for a in target_inc_ages[:-1]]
                sigmas = [1e5 * s if s < 1e9 else 0.0 for s in sigmas]
                sigmas = self.calc_CIs(mus, sigmas)
                mus, sigmas = self.suppress_low_counts(target_inc, target_inc_ages[:-1], mus, sigmas)

                x = [(age_groups[a][0]+age_groups[a][1])/2 for a in target_inc_ages[:-1]]
                plt.fill_between(x, y_vals1, y_vals2, label="Model - " + stage, color=self.stage_colors[stage], alpha=0.5)
                plt.errorbar(x, mus, sigmas, capsize=3, fmt=".--", color=self.stage_colors[stage], linewidth=1.0, markersize=3.0, ecolor=self.stage_colors[stage], label="Target - " + stage)
                plt.xlabel("Age")
                plt.ylabel("Rate per 100,000")

            for k, target_type in enumerate(self.targets.keys()):
                for stage in self.targets[target_type].keys():
                    y_vals1 = ys[target_type][stage][0, :]
                    y_vals2 = ys[target_type][stage][1, :]
                    target_inc = self.targets[target_type][stage]
                    target_inc_ages = list(target_inc.keys())
                    plot_target(self.name, k, target_type, stage, target_inc, target_inc_ages, y_vals1, y_vals2)
        else:
            plt.clf()
            fig, axes = plt.subplots(4, 3, squeeze=False)
            fig.set_size_inches(12, 10)

            major_ticks = np.arange(0, 121, 10)
            minor_ticks = np.arange(0, 121, 5)
            axes = axes.flatten()
            for ax in axes:
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.grid(which='minor', alpha=0.2)
                ax.grid(which='major', alpha=0.5)
            
            def plot_target(ax_idx, target_type, stage, target_inc, target_inc_ages, y_vals1, y_vals2):
                fig = plt.gcf()
                ax = fig.get_axes()[ax_idx]
                plt.sca(ax)
                plt.title(f"{target_type.title()} - {stage}")
                y = [1e5 * target_inc[a][0] for a in target_inc_ages[:-1]]
                sigmas = [target_inc[a][1] for a in target_inc_ages[:-1]]
                sigmas = [1e5 * s if s < 1e9 else 0.0 for s in sigmas]
                sigmas = self.calc_CIs(y, sigmas)  # 95% interval, as in the other figures (v2: +-1 SE)
                y, sigmas = self.suppress_low_counts(target_inc, target_inc_ages[:-1], y, sigmas)
                x = [(age_groups[a][0]+age_groups[a][1])/2 for a in target_inc_ages[:-1]]
                plt.fill_between(x, y_vals1, y_vals2, label="Prior Coverage", color=self.stage_colors[stage], alpha=0.5)
                plt.errorbar(x, y, sigmas, capsize=3, fmt=".--", color=self.stage_colors[stage], linewidth=1.0, markersize=3.0, ecolor=self.stage_colors[stage], label="Target")
                plt.xlabel("Age")
                plt.ylabel("Rate per 100,000")

            for idx_row, target_type in enumerate(self.targets.keys()):
                for idx_col, stage in enumerate(self.targets[target_type].keys()):
                    k = idx_row * 3 + idx_col
                    y_vals1 = ys[target_type][stage][0, :]
                    y_vals2 = ys[target_type][stage][1, :]
                    target_inc = self.targets[target_type][stage]
                    target_inc_ages = list(target_inc.keys())
                    plot_target(k, target_type, stage, target_inc, target_inc_ages, y_vals1, y_vals2)
                    if idx_col != 0:
                        plt.ylabel(None)
        save_plot(f"{self.limis_dir}/{self.name.lower()}_{filename}")
        return ys

    @staticmethod
    def limis_quantiles(samples, weights, q):
        """Weighted quantiles q of each column of samples (weights None: equal), at the midpoints of the cumulative
        weights, as tests/limis_diagnostics.py; shape (len(q), n_params)."""
        if weights is None:
            weights = np.ones(samples.shape[0])
        out = np.empty((len(q), samples.shape[1]))
        for j in range(samples.shape[1]):
            i = np.argsort(samples[:, j])
            c = np.cumsum(weights[i])
            out[:, j] = np.interp(q, (c - 0.5 * weights[i]) / c[-1], samples[i, j])
        return out

    @staticmethod
    def limis_weights(log_targets, log_denom):
        """Normalised importance weights, target / mixture density; samples outside the prior get weight 0."""
        log_w = np.where(np.isneginf(log_targets), -np.inf, log_targets - log_denom)
        weights = np.exp(log_w - log_w.max())
        return weights / weights.sum()

    def IMIS2(self, seed, priors=False, center=None, half_width=None, se=None, mle=None):
        # se: the information-matrix SEs behind the bounds, saved with the components (half_width need not be a
        # fixed multiple of them; limis_bounds' min_half_width); mle: the MLE, saved with them too, since with
        # limis_bounds' fixed_bounds the box centre is not the MLE
        prefix = self.name.lower()
        samples_dir = f"{self.limis_dir}/samples"
        if load_IMIS:
            samples = []
            log_targets = []
            log_weights = []
            # every run saved in {limis_dir}/samples/, not a fixed number of them
            pattern = f"{samples_dir}/{prefix}_samples_*.npy"
            seeds = sorted(int(re.search(r"_(\d+)\.npy$", f).group(1)) for f in glob.glob(pattern))
            if not seeds:
                raise FileNotFoundError(f"no LIMIS runs to pool: {pattern}")
            print(f"pooling {len(seeds)} LIMIS runs, seeds", seeds)
            run_bounds = []  # each run's own prior box, from its components file
            for i in seeds:
                components_i = f"{samples_dir}/{prefix}_components_{i}.npz"
                if os.path.exists(components_i):
                    with np.load(components_i) as c:
                        run_bounds.append((c["center"], c["half_width"]))
                samples_i = np.load(f"{samples_dir}/{prefix}_samples_{i}.npy")
                log_targets_i = np.load(f"{samples_dir}/{prefix}_logtargets_{i}.npy")
                log_weights_i = np.load(f"{samples_dir}/{prefix}_logweights_{i}.npy")

                if priors:
                    samples_i = samples_i[:n0, :]
                    log_targets_i = log_targets_i[:n0]
                    log_weights_i = log_weights_i[:n0]

                samples.append(samples_i)
                log_targets.append(log_targets_i)
                log_weights.append(log_weights_i)
            
            samples = np.concatenate(samples, axis=0)
            log_targets = np.concatenate(log_targets, axis=0)
            log_weights = np.concatenate(log_weights, axis=0)
            # the pooled runs' prior, not the bounds main_rates recomputes (those follow the current settings, which
            # need not be the ones the runs were sampled with); the runs must share it
            if len(run_bounds) == len(seeds):
                center, half_width = run_bounds[0]
                if not all(np.array_equal(c, center) and np.array_equal(h, half_width) for c, h in run_bounds):
                    raise ValueError(f"the runs in {samples_dir} have different prior bounds")
                print("prior bounds of the pooled runs:", np.exp(center - half_width), np.exp(center + half_width))
        else:
            if save_IMIS:  # fail now, not after sampling
                os.makedirs(samples_dir, exist_ok=True)
            samples, log_targets, log_weights = self.IMIS0(center, half_width)
            if se is not None:
                self.limis_components["se"] = np.asarray(se, dtype=float)
            if mle is not None:
                self.limis_components["mle"] = np.asarray(mle, dtype=float)
            if save_IMIS:
                np.savez(f"{samples_dir}/{prefix}_components_{seed}.npz", **self.limis_components)
                np.save(f"{samples_dir}/{prefix}_samples_{seed}.npy", samples)
                np.save(f"{samples_dir}/{prefix}_logweights_{seed}.npy", log_weights)
                np.save(f"{samples_dir}/{prefix}_logtargets_{seed}.npy", log_targets)
                exit()

        print(samples.shape, log_targets.shape, log_weights.shape)
        print("priors:", priors)

        if priors:
            weights_final = None
        else:
            weights_final = self.limis_weights(log_targets, log_weights)
            
            print(f"Seed {seed} effective sample size:", 1.0/(weights_final*weights_final).sum())
        
        nk = samples.shape[0]

        # n_samples = nk
        n_samples = self.limis_n_draws
        samples_all, weights_all = samples, weights_final
        idxs = np.random.choice(range(nk), size=n_samples, p=weights_final)
        samples = samples[idxs, :]
        if weights_final is not None:
            weights_final = weights_final[idxs]
            weights_final /= weights_final.sum()
            print(f"Seed {seed} effective sample size:", 1.0/(weights_final*weights_final).sum())

        # from all the samples and their weights, before resampling, so the quantiles carry no resampling noise
        quantiles_params = self.limis_quantiles(samples_all, weights_all, [0.025, 0.5, 0.975])
        quantiles_targets = self.eval_quantiles(samples, priors)

        risk_renames = {}
        subscripts = "₀₁₂₃₄₅₆₇₈₉"
        for risk, i in self.risk_indices.items():
            p = risk.get_n_params()
            for j in range(p):
                suffix = f"{j+1}" if p > 1 else ""
                name = risk.name + suffix
                rename = name.replace("->", "→")
                for i in range(len(subscripts)):
                    rename = rename.replace(str(i), subscripts[i])
                risk_renames[name] = rename

        bounds = None
        if self.limis_axes_to_bounds and center is not None and half_width is not None:
            bounds = np.exp(np.column_stack([center - half_width, center + half_width]))
        # always the autoscaled histograms and correlogram; with limis_axes_to_bounds also versions with axes at the
        # prior bounds (file names ..._bounds_N). Both histograms come before the correlograms, because correlogram1
        # sets a seaborn theme
        variants = [(None, "")] + ([(bounds, "_bounds")] if bounds is not None else [])

        df = pd.DataFrame()
        for axis_bounds, tag in variants:
            plt.clf()
            w, h = 4, 4
            fig, axes = plt.subplots(w, h)
            fig.set_size_inches(15, 10)
            axes = axes.ravel()
            idx = 0
            for risk, i in self.risk_indices.items():
                p = risk.get_n_params()
                for j in range(p):
                    ax = axes[idx]
                    x = np.exp(samples[:, idx])
                    ax.hist(x, bins=100)
                    suffix = f"{j+1}" if p > 1 else ""
                    name = risk.name + suffix
                    ax.set_title(risk_renames[name])
                    df[name] = x
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    if axis_bounds is not None:
                        ax.set_xlim(axis_bounds[idx])
                    idx += 1
            while idx < w*h:
                axes[idx].axis('off')
                idx += 1
            plt.tight_layout()
            plt.savefig(f"{self.limis_dir}/{prefix}_hist{tag}_{n_samples}.png", dpi=300)
            plt.savefig(f"{self.limis_dir}/{prefix}_hist{tag}_{n_samples}.pdf", dpi=300)

        for axis_bounds, tag in variants:
            correlogram1(df, risk_renames, bounds=axis_bounds)
            # correlogram2(df, risk_renames)
            plt.savefig(f"{self.limis_dir}/{prefix}_corr{tag}_{n_samples}.png", dpi=150)
            plt.savefig(f"{self.limis_dir}/{prefix}_corr{tag}_{n_samples}.pdf", dpi=150)

        return quantiles_params, quantiles_targets
