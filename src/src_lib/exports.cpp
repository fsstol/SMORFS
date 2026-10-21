#pragma once

extern "C" {
    SMORFS_EXPORT State* create_state() {return new State;}
    SMORFS_EXPORT void destroy_state(State* s) {delete s;}

    SMORFS_EXPORT Exponential* create_exponential(double rate) {return new Exponential(rate);}
    SMORFS_EXPORT void destroy_exponential(Exponential* r) {delete r;}

    SMORFS_EXPORT Weibull* create_weibull(double shape, double scale) {return new Weibull(shape, scale);}
    SMORFS_EXPORT void destroy_weibull(Weibull* r) {delete r;}

    SMORFS_EXPORT Gompertz* create_gompertz(double shape, double rate) {return new Gompertz(shape, rate);}
    SMORFS_EXPORT LogNormal* create_lognormal(double meanlog, double sdlog) {return new LogNormal(meanlog, sdlog);}
    SMORFS_EXPORT LogLogistic* create_loglogistic(double shape, double scale) {return new LogLogistic(shape, scale);}
    SMORFS_EXPORT void destroy_risk(Risk* r) {delete r;}
    // a risk's stack at time t, for the tests: [h, dh, d2h, H, dH, d2H]
    SMORFS_EXPORT void eval_risk(Risk* r, double t, int_t order, double* stack) {r->eval_time(t, order, stack);}

    SMORFS_EXPORT SemiMarkov* create_semi_markov() {return new SemiMarkov;}
    SMORFS_EXPORT void destroy_semi_markov(SemiMarkov* sm) {delete sm;}

    SMORFS_EXPORT int_t add_transition(State* state, State* other, Risk* risk) {return state->add_transition(other, risk) ? 0 : 1;}
    SMORFS_EXPORT void set_terminal(State* state, int_t terminal) {state->terminal = terminal;}

    SMORFS_EXPORT void get_params(Risk* risk, double* target) {risk->get_params(target);}
    SMORFS_EXPORT void set_params(Risk* risk, double* params) {risk->set_params(params);}

    SMORFS_EXPORT void add_state(SemiMarkov* sm, State* state) {sm->add_state(state);}
    SMORFS_EXPORT void set_initial(SemiMarkov* sm, State* state) {sm->set_initial(state);}
    SMORFS_EXPORT int_t update(SemiMarkov* sm) {return sm->update() ? 0 : 1;}
    // cache statistics: transform computations, transform reuses, basis computations, basis reuses
    SMORFS_EXPORT void get_cache_stats(SemiMarkov* sm, int_t* out) {std::copy(sm->stats, sm->stats + 4, out);}
    SMORFS_EXPORT void sm_set_parameters(SemiMarkov* sm, const double* params) {sm->set_parameters(params);}
    SMORFS_EXPORT void sm_get_parameters(SemiMarkov* sm, double* params) {sm->get_parameters(params);}
    // sizes: n_states, n_general, n_risks, n_params
    SMORFS_EXPORT void get_sizes(SemiMarkov* sm, int_t* out) {
        out[0] = sm->get_n_states();
        out[1] = sm->get_n_general();
        out[2] = sm->get_n_risks();
        out[3] = sm->n_params;
    }
    // states in the model's order, risks in parameter order, and each risk's parameter offset
    SMORFS_EXPORT void get_layout(SemiMarkov* sm, State** states, Risk** risks, int_t* offsets) {
        std::copy(sm->get_order().begin(), sm->get_order().end(), states);
        for (size_t r=0; r<sm->get_risks().size(); r++) {
            risks[r] = sm->get_risks()[r];
            offsets[r] = sm->risk_indices.at(risks[r]);
        }
    }
    SMORFS_EXPORT int_t evaluate(SemiMarkov* sm, int_t prevalence, const int_t* cumulative, const int_t* cutoff, const double* t, int_t n_t, double t0, double A, int_t l, int_t periods, double max_time, int_t eval_order,
                   double* out, double* timings) {
        sm->eval_order = eval_order;
        if (!sm->structure_current()) return 2;  // a state or transition added after update
        return sm->evaluate(prevalence, cumulative, cutoff, t, n_t, t0, A, l, periods, max_time, out, timings) ? 0 : 1;
    }
    SMORFS_EXPORT void eval_time(SemiMarkov* sm, State* state, double* stack, double* stack_risk, double t, int_t eval_order) {
        sm->eval_order = eval_order;
        sm->eval_time_state(state, stack, stack_risk, t);
    }

    // the transforms of a state's stack at n_freq frequencies gamma_re + i k dgamma by quad_vec_multi; res: n_freq x stack
    SMORFS_EXPORT void quad_vec_multi_state(SemiMarkov* sm, const State* state, complex_t* res, double gamma_re, double dgamma, int_t n_freq,
                              const double b, int_t eval_order) {
        sm->eval_order = eval_order;
        std::vector<double> stack(sm->get_stack_size(state)), stack_risk(2*(1 + state->p_sum + state->p_sum*state->p_sum));
        function_t f = [&](double t) {
            sm->eval_time_state(state, stack.data(), stack_risk.data(), t);
            return stack.data();
        };
        QuadMultiWorkspace ws;
        quad_vec_multi(f, stack.size(), n_freq, res, complex_t(gamma_re, 0.0), dgamma, b, ws);
    }
    SMORFS_EXPORT void quad_vec(SemiMarkov* sm, const State* state, complex_t* res, double* stack, double* stack_risk, double gamma_re, double gamma_im, const double b, int_t eval_order) {
        sm->eval_order = eval_order;
        int_t len = 1;
        if (eval_order >= 1) len += state->p_sum;
        if (eval_order >= 2) len += state->p_sum*state->p_sum;
        len *= 1+state->transitions.size();

        function_t f = [&](double t) {
            sm->eval_time_state(state, stack, stack_risk, t);
            return stack;
        };
        QuadWorkspace ws;
        quad_vec(f, len, res, gamma_re + I*gamma_im, b, ws);
    }
}
