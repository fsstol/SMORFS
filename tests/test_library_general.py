"""The library on model structures other than the bladder model's (CORRECTIONS.md, entries 10, 11 and 14).

G1: a cycle through a quadrature state with two risks. A -> B Weibull, A -> D exponential, B -> A and B -> C
    exponential; C and D terminal. Gradients of every arrival and prevalence value against central differences, and
    Hessians against central differences of the analytic gradients (relative to each quantity's largest
    structurally non-zero entry; components that are zero by the model's structure, such as the arrival into the initial
    state or an unreachable state, absolutely).
G2: one risk object on two transitions out of a state (B -> C and B -> D share a rate), after a Weibull onset A -> B.
    Same derivative checks, and the values against quadrature of the closed forms.
Structure errors: a second transition between the same two states, and a transition to a state not in the model,
    must raise ValueError.
G4: a Weibull onset into a random exponential graph of 16 states (fixed seed) with several cycles, so the linear systems
    have components of several states: derivative checks as G1, and values against the convolution of the onset
    density with the matrix exponential.
G3: a smooth Weibull onset H -> A into an all-exponential cycle (A <-> B, both exiting to C); values against the
    convolution of the onset density with the matrix exponential.
Value errors are relative to each quantity's largest value over the times. They leave out the initial state's prevalence: it jumps from 0 to 1 at t = 0, where the Fourier series
converges slowly (Gibbs ringing); with an exponential exit from the initial state every state
downstream inherits a kink at 0 and the error is about 1e-1 at t = 1 with 120 terms. That is the inversion's limit,
not the structure's, so each model here starts with a smooth Weibull onset.
"""
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

from common import load_h2, row

T = np.array([1.0, 3.0, 7.0, 15.0, 30.0])
T0 = 10.0  # the inversion's accuracy depends on t0 against the model's time scale: 2e-7 to 1e-5 here, 2e-2 at t0 = 70
EPS = 1e-5


def build(sm_mod, kind):
    S = sm_mod
    m = S.SemiMarkov(kind)
    st = {k: S.State(k) for k in "HABCD"}
    if kind == "G1":
        st["A"].add_transition(st["B"], S.WeibullStable("A->B", k=1.6, s=8.0))
        st["A"].add_transition(st["D"], S.ExponentialStable("A->D", rate=0.05))
        st["B"].add_transition(st["A"], S.ExponentialStable("B->A", rate=0.4))
        st["B"].add_transition(st["C"], S.ExponentialStable("B->C", rate=0.2))
    elif kind == "G2":
        st["A"].add_transition(st["B"], S.WeibullStable("A->B", k=4.0, s=10.0))
        shared = S.ExponentialStable("B->C,D", rate=0.3)
        st["B"].add_transition(st["C"], shared)
        st["B"].add_transition(st["D"], shared)
    else:
        st["H"].add_transition(st["A"], S.WeibullStable("H->A", k=4.0, s=10.0))
        st["A"].add_transition(st["B"], S.ExponentialStable("A->B", rate=0.3))
        st["A"].add_transition(st["C"], S.ExponentialStable("A->C", rate=0.1))
        st["B"].add_transition(st["A"], S.ExponentialStable("B->A", rate=0.5))
        st["B"].add_transition(st["C"], S.ExponentialStable("B->C", rate=0.2))
    for k in ("C", "D"):
        st[k].terminal = True
    if kind == "G4":
        return build_g4(sm_mod)
    names = "HABC" if kind == "G3" else "ABCD"
    for k in names:
        m.add_state(st[k])
    m.set_initial(st[names[0]])
    m.update()
    m.states = st  # keep the Python objects alive with the model
    return m


G4_N = 16  # exponential states after the onset
G4_TERMINAL = 4


def g4_rates():
    """A random exponential graph on states 0..15 (fixed seed): 12 transient states with 2 or 3 exits each, forwards and
    backwards (so several cycles), and 4 terminal states."""
    rng = np.random.default_rng(7)
    Q = np.zeros((G4_N, G4_N))
    for i in range(G4_N - G4_TERMINAL):
        dests = rng.choice([j for j in range(G4_N) if j != i], size=rng.integers(2, 4), replace=False)
        for j in dests:
            Q[i, j] = rng.uniform(0.05, 0.5)
        Q[i, i] = -Q[i].sum()
    return Q


def build_g4(S):
    m = S.SemiMarkov("G4")
    H = S.State("H")
    X = [S.State(f"X{i}") for i in range(G4_N)]
    Q = g4_rates()
    H.add_transition(X[0], S.WeibullStable("H->X0", k=4.0, s=10.0))
    for i in range(G4_N):
        for j in range(G4_N):
            if i != j and Q[i, j] > 0:
                X[i].add_transition(X[j], S.ExponentialStable(f"X{i}->X{j}", rate=Q[i, j]))
    for x in X[G4_N - G4_TERMINAL:]:
        x.terminal = True
    for st in [H] + X:
        m.add_state(st)
    m.set_initial(H)
    m.update()
    m.states = {"H": H, **{f"X{i}": x for i, x in enumerate(X)}}
    return m


def evaluate(m, th, order):
    m.set_parameters(th)
    m.eval_order = order
    arr = m.get_arrival(T, T0)
    prev = m.get_prevalence(T, T0)
    keys = [s for s in m.state_indices]
    out = []
    for res in (arr, prev):
        for s in keys:
            r = res[s]
            out.append((r.value, r.gradient if order >= 1 else None, r.hessian if order >= 2 else None))
    return out


def structural_masks(m):
    """For each output of evaluate (arrival, then prevalence, of each state), the parameters that can affect it, from
    the transition graph. A parameter of a risk out of state a can affect the prevalence of s only if a is reachable
    from the initial state and s is reachable from a (a itself included), and the arrival into s only if a also reaches
    a predecessor of s. Every other derivative component, and every Hessian entry with such a parameter, is zero."""
    states = list(m.state_indices)
    p = len(m.get_parameters())
    reach = {}
    for src in states:  # states reachable from src, src included
        seen, stack = {src}, [src]
        while stack:
            for d in stack.pop().transitions:
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        reach[src] = seen
    params_out = {}
    for a in states:
        params_out[a] = [m.risk_indices[r] + i for r in a.transitions.values() for i in range(r.get_n_params())]
    preds = {s: [a for a in states if s in a.transitions] for s in states}
    masks = []
    for kind in ("arrival", "prevalence"):
        for s in states:
            mask = np.zeros(p, dtype=bool)
            for a in reach[m.initial]:
                if (s in reach[a]) if kind == "prevalence" else any(u in reach[a] for u in preds[s]):
                    mask[params_out[a]] = True
            masks.append(mask)
    return masks


def derivative_errors(m):
    """Largest derivative errors against central differences: relative to each output's largest structurally non-zero
    component, and absolute for the components that are zero by structure (structural_masks), whose scale is round-off
    and would otherwise inflate a relative error."""
    th = m.get_parameters()
    p = th.shape[0]
    base = evaluate(m, th, 2)
    masks = structural_masks(m)
    g_err, h_err, sym, g_zero, h_zero = 0.0, 0.0, 0.0, 0.0, 0.0
    for q, (v, g, h) in enumerate(base):
        g_fd = np.empty_like(g)
        h_fd = np.empty_like(h)
        for k in range(p):
            e = np.zeros(p)
            e[k] = EPS
            up, dn = evaluate(m, th + e, 1)[q], evaluate(m, th - e, 1)[q]
            g_fd[k] = (up[0] - dn[0]) / (2 * EPS)
            h_fd[:, k] = (up[1] - dn[1]) / (2 * EPS)
        live = masks[q]
        live2 = np.outer(live, live)
        if live.any():
            g_scale = max(np.max(np.abs(g[live])), 1e-300)
            h_scale = max(np.max(np.abs(h[live2])), 1e-300)
            g_err = max(g_err, np.max(np.abs(g[live] - g_fd[live])) / g_scale)
            h_err = max(h_err, np.max(np.abs(h[live2] - h_fd[live2])) / h_scale)
            sym = max(sym, np.max(np.abs(h - h.transpose([1, 0, 2]))) / h_scale)
        if (~live).any():
            g_zero = max(g_zero, np.max(np.abs(g[~live] - g_fd[~live])))
        if (~live2).any():
            h_zero = max(h_zero, np.max(np.abs(h[~live2] - h_fd[~live2])))
    m.set_parameters(th)
    return max(g_err, g_zero), max(h_err, h_zero), sym, (g_err, g_zero, h_err, h_zero)


def weibull_density(t, k=4.0, s=10.0):
    return k / s * (t / s) ** (k - 1) * np.exp(-(t / s) ** k)


def g2_reference():
    k, s, r = 4.0, 10.0, 0.3
    surv_A = lambda t: np.exp(-(t / s) ** k)
    arr_B = lambda t: k / s * (t / s) ** (k - 1) * surv_A(t)
    prev_B = lambda t: quad(lambda u: arr_B(u) * np.exp(-2 * r * (t - u)), 0, t, epsabs=0, epsrel=1e-12, limit=200)[0]
    return {"arr B": arr_B(T), "prev B": np.array([prev_B(t) for t in T]),
            "arr C": r * np.array([prev_B(t) for t in T])}


def run(ctx):
    load_h2()  # moves to src/, where the package loads the library
    import semi_markov as S
    rows = []
    for kind in ("G1", "G2", "G4"):
        m = build(S, kind)
        g_err, h_err, sym, parts = derivative_errors(m)
        note = (f"{m.n_states} states, {m.n_params} parameters, central differences eps {EPS:g}; relative {parts[0]:.1e} "
                f"/ {parts[2]:.1e}, structural zeros absolute {parts[1]:.1e} / {parts[3]:.1e} (gradient / Hessian)")
        rows += [row("general", f"{kind}: gradient vs central differences", g_err, "<= 1e-5", g_err <= 1e-5, note),
                 row("general", f"{kind}: Hessian vs differences of the gradient", h_err, "<= 1e-5", h_err <= 1e-5, note),
                 row("general", f"{kind}: Hessian symmetry", sym, "<= 1e-12", sym <= 1e-12)]

    m = build(S, "G2")
    m.eval_order = 0
    st = m.states
    arr, prev = m.get_arrival(T, T0), m.get_prevalence(T, T0)
    got = {"arr B": arr[st["B"]].value, "prev B": prev[st["B"]].value,
           "arr C": arr[st["C"]].value}
    ref = g2_reference()
    err = max(np.max(np.abs(got[k] - ref[k])) / np.max(np.abs(ref[k])) for k in ref)
    rows.append(row("general", "G2: values vs quadrature of the closed forms", err, "<= 1e-4", err <= 1e-4,
                    "arrival in B and C, prevalence of B"))

    m = build(S, "G3")
    m.eval_order = 0
    st = m.states
    Q = np.array([[-0.4, 0.3, 0.1], [0.5, -0.7, 0.2], [0.0, 0.0, 0.0]])
    conv = lambda t, j: quad(lambda u: weibull_density(u) * expm(Q * (t - u))[0, j], 0, t, epsabs=0, epsrel=1e-12, limit=200)[0]
    P = np.array([[conv(t, j) for j in range(3)] for t in T])
    arr, prev = m.get_arrival(T, T0), m.get_prevalence(T, T0)
    got = np.concatenate([prev[st["A"]].value, prev[st["B"]].value, arr[st["A"]].value, arr[st["B"]].value, arr[st["C"]].value])
    ref = np.concatenate([P[:, 0], P[:, 1], weibull_density(T) + 0.5 * P[:, 1], 0.3 * P[:, 0], 0.1 * P[:, 0] + 0.2 * P[:, 1]])
    n_T = len(T)
    err = max(np.max(np.abs(got[i:i+n_T] - ref[i:i+n_T])) / np.max(np.abs(ref[i:i+n_T])) for i in range(0, len(ref), n_T))
    rows.append(row("general", "G3: values vs matrix exponential (cycle)", err, "<= 1e-4", err <= 1e-4,
                    "prevalence of A and B, arrival in A (onset and returns), B and C"))

    # G4: values against the convolution of the onset density with the matrix exponential
    from scipy.integrate import quad_vec
    from scipy.sparse.csgraph import connected_components
    Q = g4_rates()
    m = build(S, "G4")
    m.eval_order = 0
    st = m.states
    P = np.array([quad_vec(lambda u: weibull_density(u) * expm(Q * (t - u))[0], 0, t, epsabs=0, epsrel=1e-12)[0] for t in T])
    arr, prev = m.get_arrival(T, T0), m.get_prevalence(T, T0)
    err = 0.0
    for j in range(G4_N):
        ref_prev = P[:, j]
        ref_arr = P @ np.where(np.arange(G4_N) != j, Q[:, j], 0.0) + (weibull_density(T) if j == 0 else 0.0)
        for got, ref in ((prev[st[f"X{j}"]].value, ref_prev), (arr[st[f"X{j}"]].value, ref_arr)):
            err = max(err, np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    n_comp, labels = connected_components(Q[:G4_N, :G4_N] != 0, directed=True, connection="strong")
    sizes = sorted((int(x) for x in np.bincount(labels)), reverse=True)
    rows.append(row("general", "G4: values vs matrix exponential (random graph)", err, "<= 1e-4", err <= 1e-4,
                    f"{G4_N + 1} states, strongly connected components of sizes {sizes[:4]} ...; all prevalences and arrivals"))

    # survival = 1 - lifetime, with the lifetime's derivatives negated
    m = build(S, "G1")
    m.eval_order = 2
    res = m.get_prevalence(T, T0)
    life, surv = res[S.LIFETIME_STATE], res[S.SURVIVAL_STATE]
    err = max(np.max(np.abs(surv.value + life.value - 1)), np.max(np.abs(surv.gradient + life.gradient)),
              np.max(np.abs(surv.hessian + life.hessian)))
    rows.append(row("general", "survival = 1 - lifetime (value, gradient, Hessian)", err, "<= 1e-15", err <= 1e-15))

    # structure errors are reported, not silently accepted
    a, b, c = S.State("a"), S.State("b"), S.State("c")
    a.add_transition(b, S.ExponentialStable("a->b", rate=1.0))
    try:
        a.add_transition(b, S.ExponentialStable("a->b again", rate=2.0))
        dup = "accepted"
    except ValueError:
        dup = "raises"
    rows.append(row("general", "second transition to the same destination", dup, "raises ValueError", dup == "raises"))
    m = S.SemiMarkov("missing")
    m.add_state(a)
    m.set_initial(a)
    try:
        m.update()
        missing = "accepted"
    except ValueError:
        missing = "raises"
    rows.append(row("general", "transition to a state not in the model", missing, "raises ValueError", missing == "raises"))

    # a change of structure after update() must raise, never give the old structure's results (review, F3)
    def extended(model):
        model.states["A"].add_transition(model.states["C"], S.WeibullStable("A->C", k=2.0, s=20.0))
    m = build(S, "G1")
    m.eval_order = 1
    m.get_prevalence(T, T0)
    extended(m)
    try:
        m.get_prevalence(T, T0)
        stale = "accepted"
    except RuntimeError:
        stale = "raises"
    rows.append(row("general", "transition added after update(): evaluation", stale, "raises RuntimeError",
                    stale == "raises"))
    m.update()
    fresh = build(S, "G1")
    extended(fresh)
    fresh.update()
    theta = m.get_parameters()
    for model in (m, fresh):  # both from the same vector: a risk built from natural-scale values can differ in the last bit
        model.set_parameters(theta)
    same = True
    for model in (m, fresh):
        model.eval_order = 2
    for get in ("get_arrival", "get_prevalence"):
        a = {str(k): v for k, v in getattr(m, get)(T, T0).items()}  # results are keyed by each model's own states
        b = {str(k): v for k, v in getattr(fresh, get)(T, T0).items()}
        same = same and a.keys() == b.keys() and all(np.array_equal(getattr(a[k], f), getattr(b[k], f))
                                                     for k in a for f in ("value", "gradient", "hessian"))
    rows.append(row("general", "after update() again: equals a fresh model, bit for bit", same, "True", same,
                    "arrival and prevalence, second order"))
    m = build(S, "G1")
    m.eval_order = 0
    m.get_prevalence(T, T0)
    m.add_state(S.State("E"))
    try:
        m.get_prevalence(T, T0)
        stale = "accepted"
    except RuntimeError:
        stale = "raises"
    rows.append(row("general", "state added after update(): evaluation", stale, "raises RuntimeError",
                    stale == "raises"))
    return rows
