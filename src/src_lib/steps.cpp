#pragma once

// Optimizer and sampler steps on the p log-parameters: symmetric eigen-decompositions and least squares (Eigen).
// Matrices are p x p, row-major as NumPy's; eigenvectors are the columns of P, eigenvalues ascending.
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;

// Eigen-decomposition of the symmetric H from its lower triangle, as LAPACK's dsyevd; false if H is not finite or the
// solver fails
static bool sym_eig(const double* H, const int_t p, Eigen::VectorXd& eig, RowMatrixXd& P) {
    Eigen::Map<const RowMatrixXd> Hm(H, p, p);
    if (!Hm.allFinite()) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Eigen::MatrixXd(Hm.selfadjointView<Eigen::Lower>()));
    if (solver.info() != Eigen::Success) return false;
    eig = solver.eigenvalues();
    P = solver.eigenvectors();
    return true;
}

// x = P diag(w) P^T b, with w the inverse of the eigenvalues (or of their absolute values) and 0 below NumPy's
// least-squares cut-off, p * machine epsilon * the largest |eigenvalue|: the minimum-norm least-squares solution
static Eigen::VectorXd eig_solve(const Eigen::VectorXd& eig, const RowMatrixXd& P, const Eigen::VectorXd& b, const bool absolute) {
    const double cutoff = eig.size() * DBL_EPSILON * eig.cwiseAbs().maxCoeff();
    Eigen::VectorXd w(eig.size());
    for (int_t i=0; i<eig.size(); i++) w(i) = std::abs(eig(i)) > cutoff ? 1.0 / (absolute ? std::abs(eig(i)) : eig(i)) : 0.0;
    return P * (w.asDiagonal() * (P.transpose() * b));
}

// NumPy's nan_to_num as used by v2's Langevin step: NaN -> 1e-20, +-inf -> +-1e20
static inline double fillna(const double x) {
    return std::isnan(x) ? 1e-20 : (std::isinf(x) ? (x > 0 ? 1e20 : -1e20) : x);
}

extern "C" {
    // Newton step delta = -H^+ g (saddle_free: -|H|^+ g, with |H| = P |diag(eig)| P^T); also eig and P. Status 1 if H
    // cannot be decomposed.
    SMORFS_EXPORT int_t newton_step(const double* H, const double* g, const int_t p, const int_t saddle_free, double* delta, double* eig, double* P) {
        Eigen::VectorXd e;
        RowMatrixXd V;
        if (!sym_eig(H, p, e, V)) return 1;
        Eigen::Map<Eigen::VectorXd>(delta, p) = eig_solve(e, V, -Eigen::Map<const Eigen::VectorXd>(g, p), saddle_free);
        Eigen::Map<Eigen::VectorXd>(eig, p) = e;
        Eigen::Map<RowMatrixXd>(P, p, p) = V;
        return 0;
    }

    // Newton flow over time t: with delta = -H^+ g, new = delta + exp(t H) (params - delta)
    SMORFS_EXPORT int_t newton_flow_step(const double* H, const double* g, const double* params, const double t, const int_t p,
                           double* new_params, double* eig, double* P) {
        Eigen::VectorXd e;
        RowMatrixXd V;
        if (!sym_eig(H, p, e, V)) return 1;
        const Eigen::VectorXd delta = eig_solve(e, V, -Eigen::Map<const Eigen::VectorXd>(g, p), false);
        const Eigen::VectorXd expd = (e * t).array().exp();
        Eigen::Map<Eigen::VectorXd>(new_params, p) = delta + V * (expd.asDiagonal() * (V.transpose() * (Eigen::Map<const Eigen::VectorXd>(params, p) - delta)));
        Eigen::Map<Eigen::VectorXd>(eig, p) = e;
        Eigen::Map<RowMatrixXd>(P, p, p) = V;
        return 0;
    }

    // Minimum-norm least squares: delta minimising |J delta + r|, J n x p row-major, with NumPy's cut-off
    SMORFS_EXPORT int_t least_squares(const double* J, const double* r, const int_t n, const int_t p, double* delta) {
        Eigen::Map<const RowMatrixXd> Jm(J, n, p);
        if (!Jm.allFinite()) return 1;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Eigen::MatrixXd(Jm), Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.setThreshold(std::max(n, p) * DBL_EPSILON);
        Eigen::Map<Eigen::VectorXd>(delta, p) = svd.solve(-Eigen::Map<const Eigen::VectorXd>(r, n));
        return 0;
    }

    // Langevin step of v2: with H = P diag(eig) P^T, delta = P diag(expm1(step eig / 2) / eig) P^T grad and
    // sigma_t = P ((u u^T) o (P^T sigma0 P) + diag(expm1(step eig) / eig)) P^T, u = exp(step eig / 2); non-finite
    // entries replaced as NumPy's nan_to_num. Status 1 (and nothing written) if hess0 cannot be decomposed.
    SMORFS_EXPORT int_t langevin_step(const double* sigma0, const double* grad0, const double* hess0, const double step, const int_t p,
                        double* delta, double* sigma_t) {
        Eigen::VectorXd e;
        RowMatrixXd V;
        if (!sym_eig(hess0, p, e, V)) return 1;
        Eigen::VectorXd row(p), diag1(p), diag2(p);
        for (int_t i=0; i<p; i++) {
            row(i) = fillna(std::exp(0.5*step*e(i)));
            diag1(i) = fillna(std::expm1(0.5*step*e(i)) / e(i));
            diag2(i) = fillna(std::expm1(step*e(i)) / e(i));
        }
        const RowMatrixXd U0 = V.transpose() * Eigen::Map<const RowMatrixXd>(sigma0, p, p) * V;
        RowMatrixXd U_t = (row * row.transpose()).cwiseProduct(U0);
        U_t.diagonal() += diag2;
        Eigen::Map<Eigen::VectorXd> d(delta, p);
        Eigen::Map<RowMatrixXd> s(sigma_t, p, p);
        d = V * (diag1.asDiagonal() * (V.transpose() * Eigen::Map<const Eigen::VectorXd>(grad0, p)));
        s = V * U_t * V.transpose();
        d = d.unaryExpr(&fillna);
        s = s.unaryExpr(&fillna);
        return 0;
    }

    // Euler Langevin step of v2: mu_t = mu0 + step grad0 / 2, sigma_t = sigma0 + (R + R^T + Id) step, R = hess0 sigma0 / 2
    SMORFS_EXPORT void langevin_step_euler(const double* mu0, const double* sigma0, const double* grad0, const double* hess0,
                             const double step, const double* Id, const int_t p, double* mu_t, double* sigma_t) {
        Eigen::Map<Eigen::VectorXd>(mu_t, p) = Eigen::Map<const Eigen::VectorXd>(mu0, p) + 0.5*step*Eigen::Map<const Eigen::VectorXd>(grad0, p);
        const RowMatrixXd rhs = 0.5 * Eigen::Map<const RowMatrixXd>(hess0, p, p) * Eigen::Map<const RowMatrixXd>(sigma0, p, p);
        Eigen::Map<RowMatrixXd>(sigma_t, p, p) = Eigen::Map<const RowMatrixXd>(sigma0, p, p) + (rhs + rhs.transpose() + Eigen::Map<const RowMatrixXd>(Id, p, p)) * step;
    }
}
