"""quadws: the shared-quadrature workspace under AddressSanitizer (review of the speedup branch, finding F1).

tests/asan_quad_workspace.cpp, compiled with -fsanitize=address against the library's sources (as compile.sh, with
Accelerate on macOS), runs a Hessian evaluation of the bladder structure at the MLE and then an objective at Weibull
shape x 0.14, the call sequence that overflowed QuadMultiWorkspace::errors. Gate: the program prints "done", exits 0
and AddressSanitizer reports nothing. Information only where the compiler cannot build with AddressSanitizer.
"""
import os
import platform
import shutil
import subprocess
import tempfile

from common import ROOT, row

SOURCE = os.path.join(ROOT, "tests", "asan_quad_workspace.cpp")


def run(ctx):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return [row("quadws", "Hessian at the MLE, then objective at shape x 0.14, under ASan", "no compiler", "info",
                    None)]
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "quadws")
        cmd = [cxx, "-std=c++20", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", SOURCE, "-o", exe]
        if platform.system() == "Darwin":
            cmd[1:1] = ["-DSMORFS_USE_BLAS", "-framework", "Accelerate"]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            return [row("quadws", "Hessian at the MLE, then objective at shape x 0.14, under ASan",
                        "cannot build with ASan", "info", None, build.stderr.strip().splitlines()[-1][:120])]
        proc = subprocess.run([exe], capture_output=True, text=True, timeout=600,
                              env=dict(os.environ, ASAN_OPTIONS="halt_on_error=1"))
    report = "AddressSanitizer" in proc.stderr
    first = next((ln for ln in proc.stderr.splitlines() if "ERROR: AddressSanitizer" in ln), "")
    ok = proc.returncode == 0 and "done" in proc.stdout and not report
    return [row("quadws", "Hessian at the MLE, then objective at shape x 0.14, under ASan",
                "clean" if ok else f"exit {proc.returncode}", "clean", ok, first.split(": ", 1)[-1][:120])]
