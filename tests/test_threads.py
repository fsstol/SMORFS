"""The library runs on one thread, whatever BLAS it uses (the performance work).

With a BLAS (SMORFS_USE_BLAS), the library finds its threading control at run time (OpenBLAS, MKL, BLIS, Accelerate) and
sets one thread around each matrix product, restoring the caller's setting afterwards. Checks:
- which product the library uses (lib.blas_info), and, with a BLAS that has a threading control, one thread during the
  last product and the caller's setting after it;
- CPU time over wall time for 50 second-order prevalence evaluations on the 1,429-point grid (dominated by the product):
  at most 1.25, where a multi-threaded product would use several cores.
"""
import time

import numpy as np

from common import load_h2, row


def run(ctx):
    h2 = load_h2()
    import smorfs_benchmark as B
    from semi_markov.base import int_t, lib, ptr_int
    m = h2.build("model")
    th = np.asarray(B.THETA, float)
    alt = th.copy()
    alt[0] += 1e-9
    m.eval_order = 2
    t = np.asarray(h2.P.t, float)
    m.set_parameters(th)
    m.get_prevalence(t, h2.P.t0)  # warm-up
    info = np.zeros(2, dtype=np.int64)
    before_name = lib.blas_info(info.ctypes.data_as(ptr_int)).decode()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for r in range(50):
        m.set_parameters(alt if r % 2 == 0 else th)  # nothing reused between calls
        m.get_prevalence(t, h2.P.t0)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    name = lib.blas_info(info.ctypes.data_as(ptr_int)).decode()
    during, after = int(info[0]), int(info[1])
    rows = [row("threads", "matrix product", name, "info", None,
                f"threads during the last product {during}, after it {after} (-1: no threading control)")]
    if name not in ("Eigen", "BLAS without threading control"):
        rows.append(row("threads", f"{name}: one thread during the product", during, "1", during == 1))
    ratio = cpu / wall
    rows.append(row("threads", "CPU time / wall time, 50 second-order prevalence evaluations", ratio, "<= 1.25",
                    ratio <= 1.25, f"wall {wall:.2f} s"))
    return rows
