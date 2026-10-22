import ctypes
import os
import sys
import time
from typing import Callable

import numpy as np
from tqdm import tqdm

int_t = ctypes.c_int64
double_t = ctypes.c_double

ptr_struct = ctypes.POINTER(ctypes.c_void_p)
ptr_int = ctypes.POINTER(int_t)
ptr_double = ctypes.POINTER(double_t)
ptr_complex = ptr_struct
ptr_ptr_int = ctypes.POINTER(ptr_int)
ptr_ptr_double = ctypes.POINTER(ptr_double)

# the library built by compile.sh or CMake at the repository root (semi_markov.dll on Windows), found from this file
# rather than the working directory
LIBRARY = "semi_markov.dll" if sys.platform == "win32" else "semi_markov.so"
lib: ctypes.CDLL = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", LIBRARY))
for f in [lib.create_state, lib.create_exponential, lib.create_weibull, lib.create_gompertz, lib.create_lognormal,
          lib.create_loglogistic, lib.create_semi_markov]:
    f.restype = ptr_struct
for f in [lib.destroy_state, lib.destroy_exponential, lib.destroy_weibull, lib.destroy_risk, lib.destroy_semi_markov]:
    f.argtypes = [ptr_struct]

lib.create_exponential.argtypes = [double_t]
lib.create_weibull.argtypes = [double_t, double_t]
for f in [lib.create_gompertz, lib.create_lognormal, lib.create_loglogistic]:
    f.argtypes = [double_t, double_t]
lib.eval_risk.argtypes = [ptr_struct, double_t, int_t, ptr_double]
lib.eval_time.argtypes = [ptr_struct, ptr_struct, ptr_double, ptr_double, double_t, int_t]
lib.quad_vec.argtypes = [ptr_struct, ptr_struct, ptr_complex, ptr_double, ptr_double, double_t, double_t, double_t, int_t]
lib.quad_vec_multi_state.argtypes = [ptr_struct, ptr_struct, ptr_complex, double_t, double_t, int_t, double_t, int_t]
lib.fourier_eval1.argtypes = [ptr_double, ptr_complex, int_t, ptr_double, int_t, double_t, double_t, int_t, int_t]
lib.add_transition.restype = int_t
lib.set_terminal.argtypes = [ptr_struct, int_t]
lib.update.restype = int_t
lib.get_cache_stats.argtypes = [ptr_struct, ptr_int]
lib.acc_coefficients.argtypes = [ptr_double, int_t]
lib.blas_info.argtypes = [ptr_int]
lib.blas_info.restype = ctypes.c_char_p
lib.sm_set_parameters.argtypes = [ptr_struct, ptr_double]
lib.sm_get_parameters.argtypes = [ptr_struct, ptr_double]
lib.get_params.argtypes = [ptr_struct, ptr_double]
lib.set_params.argtypes = [ptr_struct, ptr_double]
lib.get_sizes.argtypes = [ptr_struct, ptr_int]
lib.get_layout.argtypes = [ptr_struct, ctypes.c_void_p, ctypes.c_void_p, ptr_int]
lib.evaluate.argtypes = [ptr_struct, int_t, ptr_int, ptr_int, ptr_double, int_t, double_t, double_t, int_t, int_t, double_t,
                         int_t, ptr_double, ptr_double]
lib.evaluate.restype = int_t
for f in [lib.newton_step, lib.newton_flow_step, lib.least_squares, lib.langevin_step]:
    f.restype = int_t
lib.newton_step.argtypes = [ptr_double, ptr_double, int_t, int_t, ptr_double, ptr_double, ptr_double]
lib.newton_flow_step.argtypes = [ptr_double, ptr_double, ptr_double, double_t, int_t, ptr_double, ptr_double, ptr_double]
lib.least_squares.argtypes = [ptr_double, ptr_double, int_t, int_t, ptr_double]
lib.langevin_step.argtypes = [ptr_double, ptr_double, ptr_double, double_t, int_t, ptr_double, ptr_double]
lib.langevin_step_euler.argtypes = [ptr_double, ptr_double, ptr_double, ptr_double, double_t, ptr_double, int_t, ptr_double, ptr_double]


def as_c(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=float)


def c_ptr(x: np.ndarray):
    return x.ctypes.data_as(ptr_double)


# Options for Fourier evaluation
class FourierOptions:
    A = 3e1
    l = 5  # half-period
    periods = 12
    m = 11  # acceleration window, not used in Pade-type algorithm
    verbose = False
