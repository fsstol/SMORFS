#pragma once

class Risk {
public:
    int_t n_params = 0;
    bool is_exponential = false;
    double rate = NAN;

    explicit Risk(const int_t n_params) {
        this->n_params = n_params;
    }

    virtual void set_params(const double* params) = 0;
    virtual void get_params(double* target) = 0;

    // stack: [h, dh (p), d2h (p x p), H, dH (p), d2H (p x p)], derivatives with respect to the parameters
    virtual void eval_time(const double t, const int_t order, double* stack) const = 0;

    virtual ~Risk() = default;
};

// The h and H parts of a risk's stack from the derivatives of log h and of H with respect to its two parameters:
// dh = h dlog h, d2h = h (d2log h + dlog h dlog h^T)
static void two_parameter_stack(const int_t order, const double h, const double* dlh, const double* d2lh,
                                const double H, const double* dH, const double* d2H, double* stack) {
    stack[0] = h;
    stack[7] = H;
    if (order >= 1) {
        for (int a=0; a<2; a++) {
            stack[1+a] = h * dlh[a];
            stack[8+a] = dH[a];
        }
    }
    if (order >= 2) {
        for (int a=0; a<2; a++) {
            for (int b=0; b<2; b++) {
                stack[3+2*a+b] = h * (d2lh[2*a+b] + dlh[a]*dlh[b]);
                stack[10+2*a+b] = d2H[2*a+b];
            }
        }
    }
}

class Exponential: public Risk {
private:
    double log_rate;

public:
    explicit Exponential(double rate): Risk(1) {
        is_exponential = true;
        this->rate = rate;
        log_rate = log(rate);
    }

    void set_params(const double* params) override {
        log_rate = params[0];
        rate = exp(log_rate);
    }
    void get_params(double* target) override {
        target[0] = log_rate;
    }

    void eval_time(const double t, const int_t order, double* stack) const override {
        // evaluate h, dh, d2h
        stack[0] = rate;
        if (order >= 1) stack[1] = rate;
        if (order >= 2) stack[2] = rate;

        // evaluate H, dH, d2H
        const double ratet = rate*t;
        stack[3] = ratet;
        if (order >= 1) stack[4] = ratet;
        if (order >= 2) stack[5] = ratet;
    }
};

class Weibull: public Risk {
private:
    double shape;
    double scale;
    double log_shape;
    double log_scale;
    double b;

public:
    explicit Weibull(double shape, double scale): Risk(2) {
        this->shape = shape;
        this->scale = scale;
        log_shape = log(shape);
        log_scale = log(scale);
        b = pow(scale, -shape);
    }

    void set_params(const double* params) override {
        log_shape = params[0];
        log_scale = params[1];
        shape = exp(log_shape);
        scale = exp(log_scale);
        b = pow(scale, -shape);
    }
    void get_params(double* target) override {
        target[0] = log_shape;
        target[1] = log_scale;
    }

    void eval_time(const double t, const int_t order, double* stack) const override {
        // evaluate h, dh, d2h
        const double h = b * shape * pow(t, shape-1);
        const double ts = t/scale;
        double logts;

        stack[0] = h;
        if (order >= 1) {
            logts = log(ts);
            stack[1] = shape * b * pow(t, shape-1) * (1.0 + shape*logts);
            stack[2] = scale * -pow(shape/scale, 2) * pow(ts, shape-1);
        }
        if (order >= 2) {
            const double h11 = h*(logts*logts + 2*logts/shape);
            const double h12 = -h*(shape*logts + 2)/scale;
            const double h22 = h*shape*(shape+1)/(scale*scale);
            stack[3] = h11 * shape * shape + stack[1];
            stack[4] = h12 * shape * scale;
            stack[5] = stack[4];
            stack[6] = h22 * scale * scale + stack[2];
        }

        // evaluate H, dH, d2H
        const double tb = pow(ts, shape);
        stack[7] = tb;
        if (order >= 1) {
            stack[8] = shape * tb * logts;
            stack[9] = scale * -shape/scale * tb;
        }
        if (order >= 2) {
            const double H11 = logts*logts;
            const double H12 = -(shape*logts+1)/scale;
            const double H22 = shape*(shape+1)/(scale*scale);
            stack[10] = tb * H11 * shape * shape + stack[8];
            stack[11] = tb * H12 * shape * scale;
            stack[12] = stack[11];
            stack[13] = tb * H22 * scale * scale + stack[9];
        }
    }
};

// phi_m(x) = int_0^1 u^m e^(x u) du for m = 0, 1, 2: a series for |x| < 1 (no cancellation as x -> 0), closed forms
// otherwise
static void gompertz_phi(const double x, double& p0, double& p1, double& p2) {
    if (std::abs(x) < 1.0) {
        p0 = p1 = p2 = 0.0;
        double term = 1.0;  // x^k / k!
        for (int k=0; k<30; k++) {
            p0 += term / (k + 1);
            p1 += term / (k + 2);
            p2 += term / (k + 3);
            term *= x / (k + 1);
        }
    } else {
        const double ex = std::exp(x);
        p0 = std::expm1(x) / x;
        p1 = (ex*(x - 1) + 1) / (x*x);
        p2 = (ex*(x*x - 2*x + 2) - 2) / (x*x*x);
    }
}

// Gompertz: h = a e^(b t), H = a (e^(b t) - 1) / b. Parameters: the shape b (any sign; b < 0 is a defective
// distribution, H bounded) and log a
class Gompertz: public Risk {
private:
    double shape, rate, log_rate;

public:
    explicit Gompertz(double shape, double rate): Risk(2) {
        this->shape = shape;
        this->rate = rate;
        log_rate = log(rate);
    }

    void set_params(const double* params) override {
        shape = params[0];
        log_rate = params[1];
        rate = exp(log_rate);
    }
    void get_params(double* target) override {
        target[0] = shape;
        target[1] = log_rate;
    }

    void eval_time(const double t, const int_t order, double* stack) const override {
        // log h = log a + b t; H = a t phi0(b t), dH/db = a t^2 phi1(b t), d2H/db2 = a t^3 phi2(b t)
        const double h = rate * std::exp(shape * t);
        double p0, p1, p2;
        gompertz_phi(shape * t, p0, p1, p2);
        const double H = rate * t * p0;
        const double Hb = rate * t * t * p1;
        const double dlh[2] = {t, 1.0}, d2lh[4] = {0.0, 0.0, 0.0, 0.0};
        const double dH[2] = {Hb, H}, d2H[4] = {rate * t * t * t * p2, Hb, Hb, H};
        two_parameter_stack(order, h, dlh, d2lh, H, dH, d2H, stack);
    }
};

// The standard normal's hazard lambda = phi / Q (the inverse Mills ratio), its derivatives lambda' and lambda'', and
// H = -log Q. For z >= 3, by the continued fraction Q / phi = 1 / T0, T_k = z + (k + 1) / T_{k+1}: lambda = T0,
// lambda' = T0 / T1, lambda'' = 2 T0 (T2 - T1) / (T1^2 T2), with T2 - T1 = 3 / T3 - 2 / T2; written this way nothing
// cancels, where lambda (lambda - z) - 1 loses about z^2 in relative precision. Below, from erfc.
static void normal_hazard(const double z, double& lam, double& lam1, double& lam2, double& H) {
    if (z >= 3.0) {
        const int N = z < 4.0 ? 80 : (z < 6.0 ? 40 : 24);
        double T = z, T0 = 0.0, T1 = 0.0, T2 = 0.0, T3 = 0.0;
        for (int k=N; k>=0; k--) {
            T = z + (k + 1) / T;
            if (k == 3) T3 = T;
            if (k == 2) T2 = T;
            if (k == 1) T1 = T;
            if (k == 0) T0 = T;
        }
        lam = T0;
        lam1 = T0 / T1;
        lam2 = 2 * T0 * (3 / T3 - 2 / T2) / (T1 * T1 * T2);
        H = 0.5*z*z + 0.5*std::log(2*M_PI) + std::log(T0);
        return;
    }
    const double x = z / M_SQRT2;
    const double phi = std::exp(-0.5*z*z) / std::sqrt(2*M_PI);
    if (z < 0.0) {
        const double Phi = 0.5 * std::erfc(-x);
        H = -std::log1p(-Phi);
        lam = phi / (1.0 - Phi);
    } else {
        const double Q = 0.5 * std::erfc(x);
        H = -std::log(Q);
        lam = phi / Q;
    }
    const double d = lam - z;  // > 0
    lam1 = lam * d;
    lam2 = lam * (d * d + lam * d - 1.0);
}

// Log-normal: z = (log t - mu) / sigma, H = -log Q(z), h = lambda(z) / (sigma t), with lambda = phi / Q the standard
// normal's hazard. Parameters: mu (meanlog) and log sigma
class LogNormal: public Risk {
private:
    double mu, sigma, log_sigma;

public:
    explicit LogNormal(double meanlog, double sdlog): Risk(2) {
        mu = meanlog;
        sigma = sdlog;
        log_sigma = log(sdlog);
    }

    void set_params(const double* params) override {
        mu = params[0];
        log_sigma = params[1];
        sigma = exp(log_sigma);
    }
    void get_params(double* target) override {
        target[0] = mu;
        target[1] = log_sigma;
    }

    void eval_time(const double t, const int_t order, double* stack) const override {
        std::fill(stack, stack + 14, 0.0);
        if (!(t > 0.0)) return;
        const double z = (std::log(t) - mu) / sigma;
        double lam, lam1, lam2, H;
        normal_hazard(z, lam, lam1, lam2, H);
        // dz/dmu = -1/sigma, dz/dlog sigma = -z; derivatives of h = lambda(z) / (sigma t) and of H written in z, so that
        // for z > 0 every term has the same sign
        const double st = sigma * t;
        stack[0] = lam / st;
        stack[7] = H;
        if (order >= 1) {
            stack[1] = -lam1 / (sigma * st);
            stack[2] = -(z*lam1 + lam) / st;
            stack[8] = -lam / sigma;
            stack[9] = -z * lam;
        }
        if (order >= 2) {
            stack[3] = lam2 / (sigma * sigma * st);
            stack[4] = stack[5] = (z*lam2 + 2*lam1) / (sigma * st);
            stack[6] = (3*z*lam1 + z*z*lam2 + lam) / st;
            stack[10] = lam1 / (sigma * sigma);
            stack[11] = stack[12] = (z*lam1 + lam) / sigma;
            stack[13] = z*lam + z*z*lam1;
        }
    }
};

// Log-logistic: u = (t / alpha)^beta, H = log(1 + u), h = (beta / t) u / (1 + u). Parameters: log beta (shape) and
// log alpha (scale)
class LogLogistic: public Risk {
private:
    double shape, scale, log_shape, log_scale;

public:
    explicit LogLogistic(double shape, double scale): Risk(2) {
        this->shape = shape;
        this->scale = scale;
        log_shape = log(shape);
        log_scale = log(scale);
    }

    void set_params(const double* params) override {
        log_shape = params[0];
        log_scale = params[1];
        shape = exp(log_shape);
        scale = exp(log_scale);
    }
    void get_params(double* target) override {
        target[0] = log_shape;
        target[1] = log_scale;
    }

    void eval_time(const double t, const int_t order, double* stack) const override {
        std::fill(stack, stack + 14, 0.0);
        if (!(t > 0.0)) return;
        // w = log u = beta (log t - log alpha); h = (beta / t) s(w), H = softplus(w), with s the logistic function,
        // s' = s (1 - s), s'' = s' (1 - 2 s) and 1 - 2 s = -tanh(w / 2). Derivatives written directly in s, s' and s'',
        // since through log h they cancel where they vanish
        const double w = shape * (std::log(t) - log_scale);
        const double sg = 1.0 / (1.0 + std::exp(-w));
        const double s1 = sg / (1.0 + std::exp(w));      // s (1 - s)
        const double s2 = -s1 * std::tanh(0.5 * w);       // s (1 - s) (1 - 2 s)
        const double H = w > 0.0 ? w + std::log1p(std::exp(-w)) : std::log1p(std::exp(w));
        const double bt = shape / t;
        stack[0] = bt * sg;
        stack[7] = H;
        if (order >= 1) {
            // parameters (log beta, log alpha): dw = (w, -beta)
            stack[1] = bt * (sg + s1 * w);
            stack[2] = -bt * shape * s1;
            stack[8] = sg * w;
            stack[9] = -shape * sg;
        }
        if (order >= 2) {
            stack[3] = bt * (sg + 3 * s1 * w + s2 * w * w);
            stack[4] = stack[5] = -bt * shape * (2 * s1 + s2 * w);
            stack[6] = bt * shape * shape * s2;
            stack[10] = sg * w + s1 * w * w;
            stack[11] = stack[12] = -shape * (sg + s1 * w);
            stack[13] = shape * shape * s1;
        }
    }
};
