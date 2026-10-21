#pragma once

// With a BLAS, its threading functions are looked up at run time (fourier.cpp), in the module that provides dgemm_.
// windows.h first, without the macros that clash with the standard library (min, max) and with R's headers (ERROR)
#if defined(SMORFS_USE_BLAS) && defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#elif defined(SMORFS_USE_BLAS)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <complex>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

// restrict-qualified pointers: MSVC spells the keyword __restrict
#if defined(_MSC_VER)
#define SMORFS_RESTRICT __restrict
#else
#define SMORFS_RESTRICT __restrict__
#endif

// the C interface's functions: exported from a Windows DLL, where MSVC exports nothing unless told to
#if defined(_WIN32)
#define SMORFS_EXPORT __declspec(dllexport)
#else
#define SMORFS_EXPORT
#endif

#define EIGEN_MPL2_ONLY
#include "eigen/Eigen/Core"
#include "eigen/Eigen/LU"
#include "eigen/Eigen/Eigenvalues"
#include "eigen/Eigen/SVD"

typedef int64_t int_t;
typedef std::complex<double> complex_t;
typedef std::function<double*(double)> function_t;

static const complex_t I(0.0,1.0);
