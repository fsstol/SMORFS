#pragma once

#include <cfloat>
#include <queue>
#include <algorithm>
#include <vector>

static const double sys_eps = DBL_EPSILON;
static const double sys_min = DBL_MIN;

/*
Gauss-Kronrod 21 quadrature with error estimate
*/
#define NUM_EVAL 21
#define NUM_W 10
// Gauss-Kronrod points
double X[] = {0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0,
        -0.148874338981631210884826001129720,
        -0.294392862701460198131126603103866,
        -0.433395394129247190799265943165784,
        -0.562757134668604683339000099272694,
        -0.679409568299024406234327365114874,
        -0.780817726586416897063717578345042,
        -0.865063366688984510732096688423493,
        -0.930157491355708226001207180059508,
        -0.973906528517171720077964012084452,
        -0.995657163025808080735527280689003};

// 10-point weights
double W[] = {0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338,
        0.295524224714752870173892994651338,
        0.269266719309996355091226921569469,
        0.219086362515982043995534934228163,
        0.149451349150580593145776339657697,
        0.066671344308688137593568809893332};

// 21-point weights
double V[] = {0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
        0.147739104901338491374841515972068,
        0.142775938577060080797094273138717,
        0.134709217311473325928054001771707,
        0.123491976262065851077958109831074,
        0.109387158802297641899210590325805,
        0.093125454583697605535065465083366,
        0.075039674810919952767043140916190,
        0.054755896574351996031381300244580,
        0.032558162307964727478818972459390,
        0.011694638867371874278064396062192};

static inline double max_norm(const complex_t* x, const int_t len) {
    double m = -INFINITY;
    for (int_t i=0; i<len; i++) m = std::max(m, std::abs(x[i]));
    return m;
}

struct Interval {
    double a;
    double b;
    double err;
    double round_err;
    int_t slot;  // integral stored at QuadWorkspace::integrals[slot*len]

    friend bool operator <(const Interval& lhs, const Interval& rhs) {return lhs.err < rhs.err;}
};

// Scratch memory for quad_vec, reused across calls: interval integrals in slots, the interval heap, and the
// Gauss-Kronrod sums
struct QuadWorkspace {
    int_t len = 0;
    int_t n_slots = 0;
    std::vector<complex_t> integrals;
    std::vector<int_t> free_slots;
    std::vector<Interval> heap;
    std::vector<complex_t> fv, s_k, s_g;
    std::vector<double> s_k_dabs;

    void reset(const int_t len_) {
        len = len_;
        n_slots = 0;
        free_slots.clear();
        heap.clear();
        if ((int_t)fv.size() < len*NUM_EVAL) fv.resize(len*NUM_EVAL);
        if ((int_t)s_k.size() < len) {
            s_k.resize(len);
            s_g.resize(len);
            s_k_dabs.resize(len);
        }
    }
    int_t claim() {
        if (!free_slots.empty()) {
            const int_t slot = free_slots.back();
            free_slots.pop_back();
            return slot;
        }
        if ((int_t)integrals.size() < (n_slots+1)*len) integrals.resize(2*(n_slots+1)*len);
        return n_slots++;
    }
    complex_t* integral(const Interval& iv) {return integrals.data() + iv.slot*len;}
};

// Gauss-Kronrod 21 quadrature on [a, b] with error estimate
static Interval eval_interval(const double a, const double b, const function_t& f, const complex_t gamma, QuadWorkspace& ws) {
    const int_t len = ws.len;
    Interval iv{a, b, 0.0, 0.0, ws.claim()};
    complex_t* integral = ws.integral(iv);
    complex_t* fv = ws.fv.data();
    complex_t* s_k = ws.s_k.data();
    complex_t* s_g = ws.s_g.data();
    double* s_k_dabs = ws.s_k_dabs.data();
    std::fill(s_k, s_k + len, complex_t(0.0));
    std::fill(s_g, s_g + len, complex_t(0.0));
    std::fill(s_k_dabs, s_k_dabs + len, 0.0);

    const double c = (a+b)*0.5;
    const double h = (b-a)*0.5;

    for (int_t i=0; i<NUM_EVAL; i++) {
        const double t_i = c + h*X[i];
        const complex_t factor = std::exp(-gamma * t_i);

        const int_t offset = i*len;
        const double vv = V[i];
        const double* ff_i = f(t_i);

        for (int_t j=0; j<len; j++) {
            const complex_t ff_ij = ff_i[j] * factor;
            fv[offset+j] = ff_ij;
            s_k[j] += vv * ff_ij;
        }
    }

    for (int_t i=0; i<NUM_W; i++) {
        const int_t offset = (2*i + 1)*len;
        const double ww = W[i];
        for (int_t j=0; j<len; j++) s_g[j] += ww * fv[offset+j];
    }

    for (int_t i=0; i<NUM_EVAL; i++) {
        const int_t offset = i*len;
        const double vv = V[i];
        for (int_t j=0; j<len; j++) s_k_dabs[j] += vv * std::abs(fv[offset+j] - s_k[j]*0.5);
    }

    double err = -INFINITY;
    double round_err = -INFINITY;
    double dabs = -INFINITY;
    for (int_t j=0; j<len; j++) {
        err = std::max(err, h * std::abs(s_k[j] - s_g[j]));
        dabs = std::max(dabs, h * s_k_dabs[j]);
        round_err = std::max(round_err, 50 * sys_eps * h * s_k_dabs[j]);
        integral[j] = h * s_k[j];
    }
    if ((dabs != 0) && (err != 0)) {
        err = dabs * std::min(1.0, std::pow(200 * err / dabs, 1.5));
    }
    if (round_err > sys_min) {
        err = std::max(err, round_err);
    }
    iv.err = err;
    iv.round_err = round_err;
    return iv;
}

static void quad_vec(
        const function_t& f, const int_t len, complex_t* res, const complex_t gamma, const double b, QuadWorkspace& ws,
        const double epsabs=1e-200, const double epsrel=1e-6, const int_t limit=10000
) {
    const int_t min_intervals = 2;
    ws.reset(len);
    std::vector<Interval>& heap = ws.heap;

    // Initial interval set
    Interval interval0 = eval_interval(0.0, b, f, gamma, ws);
    std::copy(ws.integral(interval0), ws.integral(interval0) + len, res);
    double global_err = interval0.err;
    double round_err = interval0.round_err;
    heap.push_back(interval0);
    std::push_heap(heap.begin(), heap.end());

    // Process intervals
    while ((heap.size() > 0) && ((int_t)heap.size() < limit)) {
        // Select interval with the largest errors for subdivision
        const Interval interval = heap.front();

        // Subdivide and evaluate
        const double c = (interval.a+interval.b)*0.5;
        Interval interval1 = eval_interval(interval.a, c, f, gamma, ws);
        Interval interval2 = eval_interval(c, interval.b, f, gamma, ws);

        const complex_t* i0 = ws.integral(interval);
        const complex_t* i1 = ws.integral(interval1);
        const complex_t* i2 = ws.integral(interval2);
        for (int_t i=0; i<len; i++) res[i] += i1[i] + i2[i] - i0[i];
        global_err += interval1.err + interval2.err - interval.err;
        round_err += interval1.round_err + interval2.round_err;

        ws.free_slots.push_back(interval.slot);
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        heap.push_back(interval1);
        std::push_heap(heap.begin(), heap.end());
        heap.push_back(interval2);
        std::push_heap(heap.begin(), heap.end());

        // Termination check
        if ((int_t)heap.size() >= min_intervals) {
            const double tol = std::max(epsabs, epsrel*max_norm(res, len));
            if (global_err < tol/8) break;
            if (global_err < round_err) break;
        }

        if (not (std::isfinite(global_err) && std::isfinite(round_err))) break;
    }
}

// One adaptive Gauss-Kronrod 21 subdivision for the transforms at several frequencies gamma_k = gamma0 + i k dgamma,
// k = 0, ..., n_freq-1: the integrand is evaluated once per node, and each frequency's integral and error estimate
// are those of quad_vec. The interval refined next is the one with the largest error relative to its frequency's
// tolerance, over the frequencies not yet converged; the subdivision stops when every frequency meets quad_vec's
// criterion. res: n_freq x len.
struct MultiInterval {
    double a;
    double b;
    int_t slot;  // integrals at QuadMultiWorkspace::integrals[slot*n_freq*len], errors at errors[slot*2*n_freq]
    double priority;

    friend bool operator <(const MultiInterval& lhs, const MultiInterval& rhs) {return lhs.priority < rhs.priority;}
};

struct QuadMultiWorkspace {
    int_t len = 0, n_freq = 0, n_slots = 0;
    std::vector<complex_t> integrals;
    // per interval: integrand F (nodes x len); weights exp(-gamma_k t_i) (nodes x frequencies), Kronrod and Gauss
    // sums and absolute deviations (len x frequencies), real and imaginary parts apart: the innermost loops run over
    // the frequencies, contiguous, so that they vectorize whatever the stack's length
    std::vector<double> F, Wr, Wi, Kr, Ki, Gr, Gi, dabs, e, round_e, d;
    std::vector<double> errors, global_err, round_err, scale;
    std::vector<int_t> free_slots;
    std::vector<char> active;
    std::vector<MultiInterval> heap;

    void reset(const int_t len_, const int_t n_freq_) {
        len = len_;
        n_freq = n_freq_;
        n_slots = 0;
        free_slots.clear();
        heap.clear();
        F.resize(NUM_EVAL*len);
        Wr.resize(n_freq*NUM_EVAL);
        Wi.resize(n_freq*NUM_EVAL);
        Kr.resize(n_freq*len);
        Ki.resize(n_freq*len);
        Gr.resize(n_freq*len);
        Gi.resize(n_freq*len);
        dabs.resize(n_freq*len);
        e.resize(n_freq);
        round_e.resize(n_freq);
        d.resize(n_freq);
        global_err.resize(n_freq);
        round_err.resize(n_freq);
        scale.resize(n_freq);
        active.resize(n_freq);
    }
    int_t claim() {
        if (!free_slots.empty()) {
            const int_t slot = free_slots.back();
            free_slots.pop_back();
            return slot;
        }
        // each buffer on its own condition: len changes between calls, so integrals can have room for more slots
        if ((int_t)integrals.size() < (n_slots+1)*n_freq*len) integrals.resize(2*(n_slots+1)*n_freq*len);
        if ((int_t)errors.size() < (n_slots+1)*2*n_freq) errors.resize(2*(n_slots+1)*2*n_freq);
        return n_slots++;
    }
    complex_t* integral(const MultiInterval& iv) {return integrals.data() + iv.slot*n_freq*len;}
    double* err(const MultiInterval& iv) {return errors.data() + iv.slot*2*n_freq;}  // n_freq errors, n_freq round-off errors
};

static MultiInterval eval_multi_interval(const double a, const double b, const function_t& f, const complex_t gamma0,
                                         const double dgamma, QuadMultiWorkspace& ws) {
    const int_t len = ws.len;
    MultiInterval iv{a, b, ws.claim(), 0.0};
    complex_t* integral = ws.integral(iv);
    double* err = ws.err(iv);
    const double c = (a+b)*0.5;
    const double h = (b-a)*0.5;

    // integrand at the nodes, and the weights exp(-gamma_k t_i) = exp(-gamma0 t_i) rot_i^k, rot_i = exp(-i dgamma t_i)
    const int_t nf = ws.n_freq;
    for (int_t i=0; i<NUM_EVAL; i++) {
        const double t_i = c + h*X[i];
        const double* ff_i = f(t_i);
        std::copy(ff_i, ff_i + len, ws.F.data() + i*len);
        complex_t w = std::exp(-gamma0 * t_i);
        const complex_t rot = std::exp(complex_t(0.0, -dgamma * t_i));
        double* wr = ws.Wr.data() + i*nf;
        double* wi = ws.Wi.data() + i*nf;
        for (int_t k=0; k<nf; k++) {
            wr[k] = w.real();
            wi[k] = w.imag();
            w = {w.real()*rot.real() - w.imag()*rot.imag(), w.real()*rot.imag() + w.imag()*rot.real()};
        }
    }

    std::fill(ws.Kr.begin(), ws.Kr.end(), 0.0);
    std::fill(ws.Ki.begin(), ws.Ki.end(), 0.0);
    std::fill(ws.Gr.begin(), ws.Gr.end(), 0.0);
    std::fill(ws.Gi.begin(), ws.Gi.end(), 0.0);
    std::fill(ws.dabs.begin(), ws.dabs.end(), 0.0);
    for (int_t i=0; i<NUM_EVAL; i++) {
        const double* wr = ws.Wr.data() + i*nf;
        const double* wi = ws.Wi.data() + i*nf;
        const bool gauss = i % 2 == 1;  // the 10-point Gauss rule uses the odd nodes
        for (int_t j=0; j<len; j++) {
            const double a = V[i] * ws.F[i*len + j];
            double* kr = ws.Kr.data() + j*nf;
            double* ki = ws.Ki.data() + j*nf;
            for (int_t k=0; k<nf; k++) {
                kr[k] += a * wr[k];
                ki[k] += a * wi[k];
            }
            if (gauss) {
                const double g = W[i/2] * ws.F[i*len + j];
                double* gr = ws.Gr.data() + j*nf;
                double* gi = ws.Gi.data() + j*nf;
                for (int_t k=0; k<nf; k++) {
                    gr[k] += g * wr[k];
                    gi[k] += g * wi[k];
                }
            }
        }
    }
    for (int_t i=0; i<NUM_EVAL; i++) {
        const double* wr = ws.Wr.data() + i*nf;
        const double* wi = ws.Wi.data() + i*nf;
        for (int_t j=0; j<len; j++) {
            const double F_ij = ws.F[i*len + j];
            const double* kr = ws.Kr.data() + j*nf;
            const double* ki = ws.Ki.data() + j*nf;
            double* da = ws.dabs.data() + j*nf;
            for (int_t k=0; k<nf; k++) {
                const double dr = wr[k] * F_ij - 0.5*kr[k];
                const double di = wi[k] * F_ij - 0.5*ki[k];
                da[k] += V[i] * std::sqrt(dr*dr + di*di);
            }
        }
    }

    // quad_vec's error estimate, per frequency
    double* e = ws.e.data();
    double* round_e = ws.round_e.data();
    double* d = ws.d.data();
    std::fill(e, e + nf, -INFINITY);
    std::fill(round_e, round_e + nf, -INFINITY);
    std::fill(d, d + nf, -INFINITY);
    for (int_t j=0; j<len; j++) {
        const double* kr = ws.Kr.data() + j*nf;
        const double* ki = ws.Ki.data() + j*nf;
        const double* gr = ws.Gr.data() + j*nf;
        const double* gi = ws.Gi.data() + j*nf;
        const double* da = ws.dabs.data() + j*nf;
        for (int_t k=0; k<nf; k++) {
            const double er = kr[k] - gr[k], ei = ki[k] - gi[k];
            e[k] = std::max(e[k], h * std::sqrt(er*er + ei*ei));
            d[k] = std::max(d[k], h * da[k]);
            round_e[k] = std::max(round_e[k], 50 * sys_eps * h * da[k]);
            integral[k*len + j] = complex_t(h * kr[k], h * ki[k]);
        }
    }
    for (int_t k=0; k<nf; k++) {
        double ek = e[k];
        if ((d[k] != 0) && (ek != 0)) {
            const double x = std::min(1.0, 200 * ek / d[k]);
            ek = d[k] * x * std::sqrt(x);  // d min(1, (200 e/d)^1.5)
        }
        if (round_e[k] > sys_min) ek = std::max(ek, round_e[k]);
        err[k] = ek;
        err[nf + k] = round_e[k];
    }
    return iv;
}

static void quad_vec_multi(
        const function_t& f, const int_t len, const int_t n_freq, complex_t* res, const complex_t gamma0, const double dgamma,
        const double b, QuadMultiWorkspace& ws, const double epsabs=1e-200, const double epsrel=1e-6, const int_t limit=10000
) {
    const int_t min_intervals = 2;
    ws.reset(len, n_freq);
    std::vector<MultiInterval>& heap = ws.heap;

    MultiInterval interval0 = eval_multi_interval(0.0, b, f, gamma0, dgamma, ws);
    std::copy(ws.integral(interval0), ws.integral(interval0) + n_freq*len, res);
    for (int_t k=0; k<n_freq; k++) {
        ws.global_err[k] = ws.err(interval0)[k];
        ws.round_err[k] = ws.err(interval0)[n_freq + k];
        ws.active[k] = 1;
    }
    // each frequency's tolerance, as in quad_vec, and the priority of an interval over the active frequencies
    auto tolerance = [&](const int_t k) {  // max |res_k| without hypot's overflow guards, which these values do not need
        double m2 = 0.0;
        for (int_t j=0; j<len; j++) m2 = std::max(m2, std::norm(res[k*len + j]));
        return std::max(epsabs, epsrel*std::sqrt(m2));
    };
    auto priority = [&](MultiInterval& iv) {
        double pr = 0.0;
        const double* e = ws.err(iv);
        for (int_t k=0; k<n_freq; k++) if (ws.active[k]) pr = std::max(pr, e[k] / ws.scale[k]);
        iv.priority = pr;
    };
    for (int_t k=0; k<n_freq; k++) ws.scale[k] = tolerance(k);
    priority(interval0);
    heap.push_back(interval0);

    while ((heap.size() > 0) && ((int_t)heap.size() < limit)) {
        const MultiInterval interval = heap.front();
        const double c = (interval.a+interval.b)*0.5;
        MultiInterval interval1 = eval_multi_interval(interval.a, c, f, gamma0, dgamma, ws);
        MultiInterval interval2 = eval_multi_interval(c, interval.b, f, gamma0, dgamma, ws);

        const complex_t* i0 = ws.integral(interval);
        const complex_t* i1 = ws.integral(interval1);
        const complex_t* i2 = ws.integral(interval2);
        for (int_t i=0; i<n_freq*len; i++) res[i] += i1[i] + i2[i] - i0[i];
        const double* e0 = ws.err(interval);
        const double* e1 = ws.err(interval1);
        const double* e2 = ws.err(interval2);
        for (int_t k=0; k<n_freq; k++) {
            ws.global_err[k] += e1[k] + e2[k] - e0[k];
            ws.round_err[k] += e1[n_freq + k] + e2[n_freq + k];
        }

        ws.free_slots.push_back(interval.slot);
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();

        // convergence of each frequency, by quad_vec's criterion
        bool done = true, finite = true, changed = false;
        for (int_t k=0; k<n_freq; k++) {
            finite &= std::isfinite(ws.global_err[k]) && std::isfinite(ws.round_err[k]);
            const double tol = tolerance(k);
            const bool converged = (int_t)heap.size() + 2 >= min_intervals && (ws.global_err[k] < tol/8 || ws.global_err[k] < ws.round_err[k]);
            if (bool(ws.active[k]) == converged) {
                ws.active[k] = !converged;
                changed = true;
            }
            done &= converged;
        }
        priority(interval1);
        priority(interval2);
        heap.push_back(interval1);
        std::push_heap(heap.begin(), heap.end());
        heap.push_back(interval2);
        std::push_heap(heap.begin(), heap.end());
        if (done || !finite) break;
        if (changed) {
            for (MultiInterval& iv: heap) priority(iv);
            std::make_heap(heap.begin(), heap.end());
        }
    }
}
