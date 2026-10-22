"""The optimizer and sampler steps of the C++ library against v2's NumPy formulas (the performance work).

100 random symmetric 13 x 13 matrices with eigenvalues of magnitude 1 to 1e4 and either sign, a quarter of them with
two zero eigenvalues: saddle-free and plain Newton steps (v2: lstsq of P |D| P^T and of H), the Newton flow, the
Langevin step, and minimum-norm least squares (40 x 13, a quarter rank-deficient). Gate 1e-9, relative.
A NaN Hessian must give the error status, so that langevin_step returns sigma0 as v2 did when eigh raised.
"""
import numpy as np

from common import load_h2, row

P_SIZE = 13


def run(ctx):
    load_h2()
    from semi_markov.base import c_ptr, lib
    rng = np.random.default_rng(1)
    p = P_SIZE
    worst = {}

    def upd(k, v):
        worst[k] = max(worst.get(k, 0.0), float(v))

    fill = lambda x: np.nan_to_num(x, copy=False, nan=1e-20, posinf=1e20, neginf=-1e20)
    for trial in range(100):
        Q = np.linalg.qr(rng.normal(size=(p, p)))[0]
        lam = rng.choice([-1.0, 1.0], size=p) * 10.0 ** rng.uniform(0, 4, size=p)
        if trial % 4 == 0:
            lam[:2] = 0.0
        H = Q @ np.diag(lam) @ Q.T
        H = 0.5 * (H + H.T)
        g, params = rng.normal(size=p), rng.normal(size=p)
        eig, P = np.linalg.eigh(H)
        d, e, V = np.empty(p), np.empty(p), np.empty((p, p))
        for sf in (1, 0):
            D = np.diag(np.abs(eig)) if sf else None
            ref = np.linalg.lstsq(P @ D @ P.T if sf else H, -g, rcond=None)[0]
            lib.newton_step(c_ptr(H), c_ptr(g), p, sf, c_ptr(d), c_ptr(e), c_ptr(V))
            upd("saddle-free Newton" if sf else "Newton", np.linalg.norm(d - ref) / np.linalg.norm(ref))
        upd("eigenvalues", np.max(np.abs(e - eig)) / np.max(np.abs(eig)))
        t = 1e-4
        ref_d = np.linalg.lstsq(H, -g, rcond=None)[0]
        ref = ref_d + P @ np.diag(np.exp(eig * t)) @ P.T @ (params - ref_d)
        out = np.empty(p)
        lib.newton_flow_step(c_ptr(H), c_ptr(g), c_ptr(params), t, p, c_ptr(out), c_ptr(e), c_ptr(V))
        upd("Newton flow", np.linalg.norm(out - ref) / np.linalg.norm(ref))
        J, r = rng.normal(size=(40, p)), rng.normal(size=40)
        if trial % 4 == 0:
            J[:, 3] = J[:, 2]
        ref = np.linalg.lstsq(J, -r, rcond=None)[0]
        lib.least_squares(c_ptr(np.ascontiguousarray(J)), c_ptr(r), 40, p, c_ptr(out))
        upd("least squares", np.linalg.norm(out - ref) / np.linalg.norm(ref))
        sig0 = np.cov(rng.normal(size=(50, p)).T)
        step = 10.0 ** rng.uniform(-6, -3)
        with np.errstate(all="ignore"):
            rw = fill(np.exp(0.5 * step * eig))
            d1, d2 = fill(np.expm1(0.5 * step * eig) / eig), fill(np.expm1(step * eig) / eig)
        ref_d = fill(P @ np.diag(d1) @ P.T @ g)
        ref_s = fill(P @ (np.outer(rw, rw) * (P.T @ sig0 @ P) + np.diag(d2)) @ P.T)
        s = np.empty((p, p))
        lib.langevin_step(c_ptr(sig0), c_ptr(g), c_ptr(H), step, p, c_ptr(d), c_ptr(s))
        upd("Langevin delta", np.linalg.norm(d - ref_d) / np.linalg.norm(ref_d))
        upd("Langevin sigma", np.linalg.norm(s - ref_s) / np.linalg.norm(ref_s))

    rows = [row("steps", f"{k} vs v2's NumPy, 100 matrices", v, "<= 1e-9", v <= 1e-9) for k, v in worst.items()]
    d, s = np.empty(p), np.empty((p, p))
    status = lib.langevin_step(c_ptr(np.eye(p)), c_ptr(np.ones(p)), c_ptr(np.full((p, p), np.nan)), 1e-3, p, c_ptr(d), c_ptr(s))
    rows.append(row("steps", "NaN Hessian gives the error status", status, "1", status == 1))
    return rows
