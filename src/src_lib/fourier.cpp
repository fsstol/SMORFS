#pragma once

// Coefficients for Pade-type acceleration algorithm 1 (Cohen, Rodriguez Villegas and Zagier). With p_j the positive
// coefficients of T_n(1 + 2x), w_k = (p_{k+1} + ... + p_n) / (p_0 + ... + p_n). The algorithm's forward recursion forms
// each w_k as T_n(3) minus a running sum, which cancels as w_k falls towards p_n / T_n(3); here the tail sums are built
// backwards from p_n, adding positive terms only, so every weight keeps its relative precision.
static void acc_alg1(double* coeffs, int_t n) {
    double q = 1.0;     // p_k / p_n, from p_k = p_{k+1} (k + 1/2)(k + 1) / ((n + k)(n - k))
    double tail = 0.0;  // (p_{k+1} + ... + p_n) / p_n
    for (int_t k=n-1; k>=0; k--) {
        tail += q;
        coeffs[k] = tail;
        q *= ((k + 0.5) * (k + 1)) / (double(n + k) * double(n - k));
    }
    const double total = tail + q;  // (p_0 + ... + p_n) / p_n
    for (int_t k=0; k<n; k++) coeffs[k] /= total;
}

// The acceleration weights for n half-periods, for the tests
extern "C" SMORFS_EXPORT void acc_coefficients(double* out, const int_t n) {acc_alg1(out, n);}

#ifdef SMORFS_USE_BLAS
// Fortran BLAS, as exported by Accelerate, OpenBLAS, MKL, BLIS and R's BLAS (dllimport: its address is the BLAS's own)
#if defined(_WIN32)
extern "C" __declspec(dllimport)
#else
extern "C"
#endif
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha,
            const double* a, const int* lda, const double* b, const int* ldb, const double* beta, double* c, const int* ldc);

// A function exported by the module that provides dgemm_: another BLAS in the process (NumPy's OpenBLAS exports the
// same names) has its own thread setting
static void* blas_symbol(const char* name) {
#if defined(_WIN32)
    HMODULE m = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&dgemm_), &m)) return nullptr;
    return reinterpret_cast<void*>(GetProcAddress(m, name));
#else
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&dgemm_), &info) || !info.dli_fname) return nullptr;
    void* h = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return nullptr;
    void* f = dlsym(h, name);
    dlclose(h);
    return f;
#endif
}

// The BLAS's threading control, found once at run time: the library runs on one thread (parallel runs use processes),
// whatever BLAS it is linked to. A reference BLAS has no threads, hence no control.
struct BlasThreads {
    const char* name = "BLAS without threading control";
    int (*get)() = nullptr;     // the caller's setting
    void (*set)(int) = nullptr;  // set it
    int single = 1;              // the value meaning one thread
};
static int (*mkl_set_local)(int) = nullptr;  // MKL: thread-local setting, returns the previous one
static int (*accel_set)(int) = nullptr;      // Accelerate: BLASSetThreading (macOS 15)
static void blas_accel_set(int v) {accel_set(v);}
static int (*blis_get)() = nullptr;
static void (*blis_set)(long long) = nullptr;
static int blas_blis_get() {return blis_get();}
static void blas_blis_set(int v) {blis_set(v);}
static const BlasThreads& blas_threads() {
    static const BlasThreads t = [] {
        BlasThreads b;
        if (auto set = reinterpret_cast<void (*)(int)>(blas_symbol("openblas_set_num_threads"))) {
            b.name = "OpenBLAS";
            b.set = set;
            b.get = reinterpret_cast<int (*)()>(blas_symbol("openblas_get_num_threads"));
        } else if ((mkl_set_local = reinterpret_cast<int (*)(int)>(blas_symbol("MKL_Set_Num_Threads_Local")))) {
            b.name = "MKL";
        } else if ((blis_set = reinterpret_cast<void (*)(long long)>(blas_symbol("bli_thread_set_num_threads")))) {
            blis_get = reinterpret_cast<int (*)()>(blas_symbol("bli_thread_get_num_threads"));
            b.name = "BLIS";
            b.set = blas_blis_set;
            b.get = blis_get ? blas_blis_get : nullptr;
        } else if ((accel_set = reinterpret_cast<int (*)(int)>(blas_symbol("BLASSetThreading")))) {
            b.name = "Accelerate";
            b.set = blas_accel_set;
            b.get = reinterpret_cast<int (*)()>(blas_symbol("BLASGetThreading"));
            b.single = 1;  // BLAS_THREADING_SINGLE_THREADED
        }
        return b;
    }();
    return t;
}
static int blas_threads_during = -1, blas_threads_after = -1;  // as seen during and after the last product, for the tests
#endif

// R (m x n, row-major) = X (m x k, column-major) * B (k x n, column-major): with a BLAS if built with SMORFS_USE_BLAS,
// on one thread
static void series_product(double* R, const double* X, const double* B, const int_t m, const int_t k, const int_t n) {
    if (m == 0 || n == 0) return;  // an empty product (e.g. an empty time grid): BLAS rejects ldc = 0
    if (k == 0) {std::fill(R, R + m*n, 0.0); return;}
#ifdef SMORFS_USE_BLAS
    const BlasThreads& bt = blas_threads();
    int saved = 0;
    if (mkl_set_local) saved = mkl_set_local(1);
    else if (bt.set) {
        saved = bt.get ? bt.get() : -1;
        if (saved != bt.single) bt.set(bt.single);
    }
    blas_threads_during = mkl_set_local ? 1 : (bt.get ? bt.get() : -1);
    // as column-major, R^T = B^T X^T
    const int m_ = n, n_ = m, k_ = k, lda = k, ldb = m, ldc = n;
    const double one = 1.0, zero = 0.0;
    dgemm_("T", "T", &m_, &n_, &k_, &one, B, &lda, X, &ldb, &zero, R, &ldc);
    if (mkl_set_local) mkl_set_local(saved);
    else if (bt.set && saved >= 0 && saved != bt.single) bt.set(saved);
    blas_threads_after = mkl_set_local ? saved : (bt.get ? bt.get() : -1);
#else
    using namespace Eigen;
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(R, m, n).noalias() = Map<const MatrixXd>(X, m, k) * Map<const MatrixXd>(B, k, n);
#endif
}

// The series' product: which BLAS, and its thread setting during and after the last product (-1: unknown); for the tests
extern "C" SMORFS_EXPORT const char* blas_info(int_t* threads) {
#ifdef SMORFS_USE_BLAS
    threads[0] = blas_threads_during;
    threads[1] = blas_threads_after;
    return blas_threads().name;
#else
    threads[0] = threads[1] = 1;
    return "Eigen";
#endif
}

// Basis B (2 terms x n_t, column-major) of the accelerated Fourier series: res = [Re T, Im T] * B, where row k holds
// the acceleration weight times the rotation exp(i pi k t / (l t0)) and the scale exp(A t / (2 l t0)) / (l t0)
// coeffs: scratch of 2 periods entries
static void fourier_basis(double* B, double* coeffs, const double* t, const int_t n_t, const double t0, const double A, const int_t l, const int_t periods) {
    const int_t half_periods = 2*periods;
    acc_alg1(coeffs, half_periods);
    const int_t terms = half_periods * l;
    for (int_t i=0; i<n_t; i++) {
        double* B_i = B + i*2*terms;
        const double f = std::exp(t[i] * A / (2 * l * t0)) / (l * t0);
        const complex_t rot = std::exp(t[i] * I * M_PI / (l * t0));
        complex_t u = 1.0;
        for (int_t k=0; k<terms; k++) {
            const double c = coeffs[k / l] * f;
            B_i[k] = c * u.real();
            B_i[terms + k] = -c * u.imag();
            u *= rot;
        }
        B_i[0] -= 0.5 * f;  // the k = 0 term has weight 1/2
    }
}

// Fourier series evaluation, with Pade-type acceleration algorithm 1, as one real matrix product
extern "C" SMORFS_EXPORT void fourier_eval1(
        double* res,
        const complex_t* stack, const int_t n_stack,
        const double* t, const int_t n_t, const double t0,
        const double A, const int_t l, const int_t periods
) {
    using namespace Eigen;
    const int_t terms = 2 * periods * l;
    MatrixXd B(2*terms, n_t);
    std::vector<double> coeffs(2*periods);
    fourier_basis(B.data(), coeffs.data(), t, n_t, t0, A, l, periods);
    MatrixXd X(n_stack, 2*terms);
    for (int_t k=0; k<terms; k++) {
        const complex_t* T_k = stack + k*n_stack;
        for (int_t s=0; s<n_stack; s++) {
            X(s, k) = T_k[s].real();
            X(s, terms + k) = T_k[s].imag();
        }
    }
    series_product(res, X.data(), B.data(), n_stack, 2*terms, n_t);
}

// Expand series rows (value, gradient p, Hessian upper triangle p(p+1)/2) x n_t into full p x p Hessians
extern "C" SMORFS_EXPORT void expand_hessian(const double* compact, double* full, const int_t n_rows, const int_t p, const int_t n_t) {
    const int_t hc = 1 + p + p*(p+1)/2;
    const int_t hf = 1 + p + p*p;
    for (int_t r=0; r<n_rows; r++) {
        const double* src = compact + r*hc*n_t;
        double* dst = full + r*hf*n_t;
        std::copy(src, src + (1+p)*n_t, dst);
        int_t c = 1 + p;
        for (int_t l=0; l<p; l++) {
            for (int_t k=l; k<p; k++, c++) {
                std::copy(src + c*n_t, src + (c+1)*n_t, dst + (1+p+l*p+k)*n_t);
                if (k != l) std::copy(src + c*n_t, src + (c+1)*n_t, dst + (1+p+k*p+l)*n_t);
            }
        }
    }
}
