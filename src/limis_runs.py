"""Task 9: independent LIMIS runs of main.py, one process per seed, spread across the cores.

Each run is main.py's main_rates_all(seed) in its own process, exactly as `python main.py --seed SEED`, with three
settings overridden at run time (parameters.py is not edited):
    run_IMIS = True     run LIMIS
    load_IMIS = False   sample; do not reload old samples
    save_IMIS = True    save samples, log targets, log mixture densities and the proposal components to
                        LIMIS_DIR/samples/ (IMIS2 then exits)
The LIMIS settings (n0, b, epochs_IMIS, delta, ptime, steps, deg) are parameters.py's unless --n0, --b or --epochs
are given (smoke test only). The prior is uniform on the log scale, centred at ../output/urinary bladder mle.npy,
half-width SingularCancer.limis_n_se information-matrix SEs (model.limis_bounds), or max(--min-half-width, that) per
parameter when --min-half-width is given (task 9b, the wide prior); --fixed-bounds NAME=LO:HI,... replaces that rule by
fixed natural-scale bounds for the parameters named, as in main_rates' bounds table (task 9c: Incidence_1 and
Incidence_2, the Weibull shape and scale). A run raises before sampling if that file is missing, its information
matrix is not positive definite, a name is unknown or a fixed box does not contain the MLE. --limis-dir (default
../output/IMIS2) is where the samples and logs go (SingularCancer.limis_dir). Runs share no state: main_rates seeds
NumPy's global generator with the run's seed, and each process is single-threaded (BLAS threads set to 1).
Only the first seed launched writes the target table ../output/urinary bladder_rates.csv (main.write_rates_table);
the others skip it, so no two processes write the same file.

Run from src/:
    python limis_runs.py --seeds 0-19 --workers 20                       # task 9
    python limis_runs.py --seeds 0-19 --workers 20 --min-half-width 0.3 --limis-dir ../output/IMIS2_wide  # task 9b
    python limis_runs.py --seeds 0-19 --workers 20 --min-half-width 0.3 \
        --fixed-bounds Incidence_1=7:8,Incidence_2=130:140 --limis-dir ../output/IMIS2_weibull           # task 9c
    python limis_runs.py --seeds 0-1 --n0 40 --b 20 --epochs 2           # smoke test (tests/limis_smoke.py)
Logs: LIMIS_DIR/logs/seed_SS.log (one per run) and LIMIS_DIR/logs/runs.csv (seed, status, seconds).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CHILD = r'''
import json, sys, time
import matplotlib; matplotlib.use("Agg")
import model, main
seed, n0, b, epochs, write_table = (int(x) for x in sys.argv[1:6])
limis_dir = sys.argv[6] if len(sys.argv) > 6 else model.SingularCancer.limis_dir
min_half_width = float(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "none" else None
main.write_rates_table = bool(write_table)
model.SingularCancer.limis_dir = limis_dir
main.limis_min_half_width = min_half_width
fixed_bounds = json.loads(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != "none" else None
main.limis_fixed_bounds = None if fixed_bounds is None else {k: tuple(v) for k, v in fixed_bounds.items()}
for mod in (model, main):
    mod.run_IMIS, mod.load_IMIS, mod.save_IMIS = True, False, True
    if n0 > 0: mod.n0 = n0
    if b > 0: mod.b = b
    if epochs > 0: mod.epochs_IMIS = epochs
print(f"LIMIS seed {seed}: n0 = {model.n0}, b = {model.b}, epochs = {model.epochs_IMIS}, delta = {model.delta}, "
      f"ptime = {model.ptime}, steps = {model.steps}, deg = {model.deg}, limis_dir = {limis_dir}, "
      f"min_half_width = {min_half_width}" + ("" if fixed_bounds is None else f", fixed_bounds = {fixed_bounds}"),
      flush=True)
main.main_rates_all(seed=seed)
'''
SINGLE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def parse_fixed_bounds(text):
    """NAME=LO:HI,NAME=LO:HI -> {NAME: [LO, HI]}, natural scale, 0 < LO < HI"""
    out = {}
    for part in text.split(","):
        name, _, rng = part.partition("=")
        lo, _, hi = rng.partition(":")
        try:
            lo, hi = float(lo), float(hi)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not NAME=LO:HI: {part!r}")
        if not (name and 0.0 < lo < hi):
            raise argparse.ArgumentTypeError(f"not NAME=LO:HI with 0 < LO < HI: {part!r}")
        out[name] = [lo, hi]
    return out


def run_one(seed, args, write_table=False):
    log_path = os.path.join(args.logs, f"seed_{seed:02d}.log")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", **SINGLE_THREAD)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, "-c", CHILD, str(seed), str(args.n0), str(args.b), str(args.epochs),
                               str(int(write_table)), args.limis_dir,
                               "none" if args.min_half_width is None else repr(args.min_half_width),
                               "none" if args.fixed_bounds is None else json.dumps(args.fixed_bounds)],
                              stdout=log, stderr=subprocess.STDOUT, env=env)
    seconds = time.perf_counter() - t0
    prefix = f"{args.limis_dir}/samples/urinary bladder"
    files = [f"{prefix}_{k}_{seed}.{ext}" for k, ext in
             (("samples", "npy"), ("logtargets", "npy"), ("logweights", "npy"), ("components", "npz"))]
    ok = proc.returncode == 0 and all(os.path.exists(f) and os.path.getmtime(f) >= time.time() - seconds - 1
                                      for f in files)
    print(f"seed {seed:2d}: {'ok' if ok else 'FAILED'} (exit {proc.returncode}) in {seconds / 60:.1f} min, "
          f"log {log_path}", flush=True)
    return seed, ok, proc.returncode, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0-19", help="e.g. 0-19 or 0,3,5-7")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--n0", type=int, default=0, help="smoke test only; default parameters.py's n0")
    parser.add_argument("--b", type=int, default=0, help="smoke test only; default parameters.py's b")
    parser.add_argument("--epochs", type=int, default=0, help="smoke test only; default parameters.py's epochs_IMIS")
    parser.add_argument("--limis-dir", default="../output/IMIS2", help="where samples/ and logs/ go")
    parser.add_argument("--min-half-width", type=float, default=None,
                        help="log-scale floor of the prior half-width, max(this, limis_n_se SE); task 9b: 0.3")
    parser.add_argument("--fixed-bounds", type=parse_fixed_bounds, default=None, metavar="NAME=LO:HI,...",
                        help="fixed natural-scale prior bounds for these parameters; task 9c: "
                             "Incidence_1=7:8,Incidence_2=130:140")
    args = parser.parse_args()
    if not os.path.exists("main.py"):
        sys.exit("run from src/: cd src && python limis_runs.py ...")
    if not os.path.exists("../output/urinary bladder mle.npy"):
        sys.exit("../output/urinary bladder mle.npy is missing: LIMIS is centred at the MLE")
    args.logs = f"{args.limis_dir}/logs"
    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(f"{args.limis_dir}/samples", exist_ok=True)
    seeds = parse_seeds(args.seeds)
    print(f"{len(seeds)} LIMIS runs (seeds {args.seeds}) on {args.workers} workers, {os.cpu_count()} cores, into "
          f"{args.limis_dir}, min_half_width {args.min_half_width}"
          + ("" if args.fixed_bounds is None else f", fixed_bounds {args.fixed_bounds}"), flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda s: run_one(s, args, write_table=(s == seeds[0])), seeds))
    wall = time.perf_counter() - t0
    with open(os.path.join(args.logs, "runs.csv"), "w") as fh:
        fh.write("seed,ok,exit_code,seconds\n")
        for seed, ok, code, seconds in results:
            fh.write(f"{seed},{ok},{code},{seconds:.1f}\n")
    failed = [s for s, ok, _, _ in results if not ok]
    print(f"done in {wall / 60:.1f} min; {len(seeds) - len(failed)} ok, failed: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
