"""The library's caches recompute whenever their inputs change (branch speedup, entries 20 and 21).

One model makes a sequence of get_arrival / get_prevalence calls that change one input at a time. Every result must
equal, bit for bit, the same call on a freshly built model (empty caches), and the cache counters
(lib.get_cache_stats: transform computations and reuses, basis computations and reuses) must show a reuse exactly
when nothing the cache depends on changed.
Series basis (entry 20): depends on the time grid t (bit for bit), t0 and the Fourier options A, l, periods.
Transforms (entry 21): depend on the parameters (however they were set), eval_order, t0 and the Fourier options, the
cut-off states, the initial state and the model's layout (update). Arrival or prevalence, cumulative and terminal flags
only change the assembly, so they reuse the transforms. Changes persist from step to step; the fresh model replays all
of them before its call.
"""
import numpy as np

from common import load_h2, row


def run(ctx):
    h2 = load_h2()
    import smorfs_benchmark as B
    from semi_markov.base import lib, ptr_int

    m = h2.build("model")
    t0 = h2.P.t0
    th = np.asarray(B.THETA, float)

    def stats():
        out = np.zeros(4, dtype=np.int64)
        lib.get_cache_stats(m.ptr, out.ctypes.data_as(ptr_int))
        return out

    def call(model, kind, t, tz, order, opts):
        model.set_parameters(th)
        model.eval_order = order
        model.options.A, model.options.l, model.options.periods = opts
        res = model.get_arrival(t, tz) if kind == "arrival" else model.get_prevalence(t, tz)
        return {s.name: (r.value, r.gradient, r.hessian) for s, r in res.items()}

    def same(a, b):
        return all(all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(a[k], b[k])) for k in a)

    t1 = np.linspace(0.0, 100.0, 301)
    t1b = t1.copy()
    t1b[150] = np.nextafter(t1b[150], np.inf)  # one bit
    grids = [np.linspace(0.0, 100.0, n) for n in (101, 121, 141, 161, 181)]  # more grids than the cache holds
    base_opts = (30.0, 5, 12)
    # (label, kind, t, t0, order, options, basis reused?)
    steps = [("first call", "arrival", t1, t0, 1, base_opts, False),
             ("same grid", "arrival", t1, t0, 1, base_opts, True),
             ("same grid, prevalence", "prevalence", t1, t0, 1, base_opts, True),
             ("one bit of t changed", "arrival", t1b, t0, 1, base_opts, False),
             ("t0 changed", "arrival", t1, t0 + 1.0, 1, base_opts, False),
             ("A changed", "arrival", t1, t0, 1, (31.0, 5, 12), False),
             ("l changed", "arrival", t1, t0, 1, (30.0, 6, 12), False),
             ("periods changed", "arrival", t1, t0, 1, (30.0, 5, 13), False)]
    steps += [(f"grid {i} of 5", "arrival", g, t0, 0, base_opts, False) for i, g in enumerate(grids)]
    steps += [("grid 0 again (evicted)", "arrival", grids[0], t0, 0, base_opts, False),
              ("grid 4 again (kept)", "arrival", grids[4], t0, 0, base_opts, True)]

    bad_values, bad_counts = [], []
    for label, kind, t, tz, order, opts, reuse in steps:
        before = stats()
        got = call(m, kind, t, tz, order, opts)
        after = stats()
        fresh = call(h2.build("model"), kind, t, tz, order, opts)
        if not same(got, fresh):
            bad_values.append(label)
        if (after[3] - before[3] == 1) != reuse or (after[2] - before[2] == 1) == reuse:
            bad_counts.append(label)
    note = f"{len(steps)} calls"
    rows = [row("cache", "basis: results equal a fresh model, bit for bit", f"{len(steps) - len(bad_values)}/{len(steps)}",
                "all", not bad_values, ", ".join(bad_values) or note),
            row("cache", "basis: reused exactly when t, t0, A, l, periods unchanged", f"{len(steps) - len(bad_counts)}/{len(steps)}",
                "all", not bad_counts, ", ".join(bad_counts) or note)]
    return rows + transform_rows(h2, th, t0, same)


def transform_rows(h2, th, t0, same):
    from semi_markov.base import lib, ptr_int
    m = h2.build("model")
    t = np.linspace(0.0, 100.0, 301)
    one_bit = th.copy()
    one_bit[5] = np.nextafter(one_bit[5], np.inf)
    risk_moved = np.array([np.log(0.031)])

    def stats(model):
        out = np.zeros(4, dtype=np.int64)
        lib.get_cache_stats(model.ptr, out.ctypes.data_as(ptr_int))
        return out

    # mods: changes applied to the model and kept; call: (kind, order, t0, (A, l, periods), cumulative, cut-off names)
    base = ("arrival", 1, t0, (30.0, 5, 12), False, ())
    with_ = lambda **kw: tuple(kw.get(k, v) for k, v in zip(("kind", "order", "t0", "opts", "cum", "cut"), base))
    steps = [
        ("first call", [lambda md: md.set_parameters(th)], base, False),
        ("prevalence after arrival", [], with_(kind="prevalence"), True),
        ("cumulative", [], with_(cum=True), True),
        ("cumulative for one state", [], with_(cum=["T1"]), True),
        ("terminal flag changed", [lambda md: setattr(md.T1, "terminal", True)], base, True),
        ("terminal flag restored", [lambda md: setattr(md.T1, "terminal", False)], base, True),
        ("one bit of one parameter (set_parameters)", [lambda md: md.set_parameters(one_bit)], base, False),
        ("one risk's set_params", [lambda md: md.R_T1_D1.set_params(risk_moved)], base, False),
        ("same again, prevalence", [], with_(kind="prevalence"), True),
        ("eval_order 2", [], with_(order=2), False),
        ("eval_order 2, prevalence", [], with_(order=2, kind="prevalence"), True),
        ("eval_order 0", [], with_(order=0), False),
        ("cut-off state U1", [], with_(cut=("U1",)), False),
        ("same cut-off again", [], with_(cut=("U1",)), True),
        ("cut-off removed", [], base, False),
        ("t0 changed", [], with_(t0=t0 + 1.0), False),
        ("A changed", [], with_(opts=(31.0, 5, 12)), False),
        ("periods changed", [], with_(opts=(30.0, 5, 13)), False),
        ("back to the base call", [], base, False),
        ("update()", [lambda md: md.update()], base, False),
        ("initial state U1", [lambda md: md.set_initial(md.U1)], base, False),
        ("initial state H0 again", [lambda md: md.set_initial(md.H0)], base, False),
        ("same again", [], base, True),
    ]

    def call(model, spec):
        kind, order, tz, opts, cum, cut = spec
        model.eval_order = order
        model.options.A, model.options.l, model.options.periods = opts
        if isinstance(cum, list):
            cum = [getattr(model, name) for name in cum]
        cut = [getattr(model, name) for name in cut]
        res = model.get_arrival(t, tz, cutoff_states=cut, cumulative=cum) if kind == "arrival" else model.get_prevalence(t, tz)
        return {s.name: (r.value, r.gradient, r.hessian) for s, r in res.items()}

    applied, bad_values, bad_counts = [], [], []
    for label, mods, spec, reuse in steps:
        for mod in mods:
            mod(m)
        applied += mods
        before = stats(m)
        got = call(m, spec)
        after = stats(m)
        fresh = h2.build("model")
        for mod in applied:
            mod(fresh)
        if not same(got, call(fresh, spec)):
            bad_values.append(label)
        if (after[1] - before[1] == 1) != reuse or (after[0] - before[0] == 1) == reuse:
            bad_counts.append(label)
    note = f"{len(steps)} calls"
    return [row("cache", "transforms: results equal a fresh model, bit for bit", f"{len(steps) - len(bad_values)}/{len(steps)}",
                "all", not bad_values, ", ".join(bad_values) or note),
            row("cache", "transforms: reused exactly when their inputs are unchanged", f"{len(steps) - len(bad_counts)}/{len(steps)}",
                "all", not bad_counts, ", ".join(bad_counts) or note)]
