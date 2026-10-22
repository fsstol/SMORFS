import os, sys, importlib, time, numpy as np, pandas as pd
def _repo_root():
    """Find the repository root (the folder containing src/model.py), wherever this script lives."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(d, "src", "model.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot find src/model.py: run this from inside the repository.")
ROOT = _repo_root()
VER = os.path.join(ROOT, "verification")
def vfile(name):
    """A verification input, from verification/ if present, else from the repository root."""
    p = os.path.join(VER, name)
    return p if os.path.exists(p) else os.path.join(ROOT, name)
os.chdir(os.path.join(ROOT, 'src')); sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
from scipy.ndimage import gaussian_filter1d
import main as M                     # v2 entry point: its target functions, unmodified
P = M                                # parameters are star-imported into main
eps = 1e-4
NAMES = ["shape","scale","U1-U2","U1-T1","U1-D","U2-U3","U2-T2","U2-D","U3-T3","U3-D","T1-D","T2-D","T3-D"]

def exact_targets(stage_groups):
    df = pd.read_csv("../data/single-multiple/rates_urinary_bladder.csv")
    df = df.loc[df[P.diag_col] != "2000-2021"].copy()
    mo = pd.read_csv("../data/single-multiple/mortality_urinary_bladder.csv")
    mo = mo.loc[mo["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)"].copy()
    doa = pd.read_pickle("../data/single-multiple/doa_urinary bladder.pkl").reset_index(drop=False)
    M.apply_corrected_denominators(df, mo, doa)  # only with main.corrected_denominators (entry 79), as main_rates
    # main.py halves Population here; deliberately NOT applied (SEER crude rate = Count/Population)
    df[P.stage_col] = df[P.stage_col].fillna("Blank(s)")
    df = df.loc[~df[P.stage_col].isin(['Blank(s)', 'Unknown/unstaged'])]
    tg = {k: {} for k in P.target_names}
    for name, stages in stage_groups.items():
        a, b, c = df[df[P.stage_col].isin(stages)], mo[mo[P.stage_col].isin(stages)], doa[doa[P.stage_col].isin(stages)]
        tg["clinical incidence"][name] = M.get_target_incidence(a)
        tg["clinical mortality"][name] = M.get_target_mortality(b)
        tg["preclinical mortality"][name] = M.get_target_doa(c.loc[c["outcome"] == 1])
        tg["preclinical other mortality"][name] = M.get_target_doa(c.loc[c["outcome"] == 2])
    return tg

def build(modname):
    mod = importlib.import_module(modname)
    S = P.get_survival_function()
    m = mod.SingularCancer("Urinary Bladder", 7.0, 140.0)
    m.targets = exact_targets(m.stage_groups)
    m.survival_function = S; m.survival_weights = S.cumsum() * P.dt
    m.survival_grad = -np.gradient(gaussian_filter1d(S, int(1.0 / P.dt), mode='nearest'), P.dt)
    return m

def ev(m, th, order):
    m.set_parameters(np.asarray(th, float)); m.eval_order = order
    return m.eval_targets()
f = lambda m, th: ev(m, th, 0)[0]
def grad_ok(m, th):                  # analytic for the 11 rates (exact), finite differences for shape and scale
    g = np.array(ev(m, th, 1)[1])
    for k in (0, 1):
        e = np.zeros(13); e[k] = eps; g[k] = (f(m, th+e) - f(m, th-e)) / (2*eps)
    return g
def fd_grad(m, th):
    g = np.empty(13)
    for k in range(13):
        e = np.zeros(13); e[k] = eps; g[k] = (f(m, th+e) - f(m, th-e)) / (2*eps)
    return g
def hess_ok(m, th):
    H = np.empty((13, 13))
    for k in range(13):
        e = np.zeros(13); e[k] = eps; H[:, k] = (grad_ok(m, th+e) - grad_ok(m, th-e)) / (2*eps)
    return 0.5 * (H + H.T)
def newton(m, th, iters=6, tol=1.0):
    fv = f(m, th); hist = [fv]
    for it in range(iters):
        g = grad_ok(m, th)
        if np.linalg.norm(g) < tol: break
        H = hess_ok(m, th); w, V = np.linalg.eigh(H); step = -V @ ((V.T @ g) / np.abs(w)); a = 1.0
        while f(m, th + a*step) > fv and a > 1e-4: a *= 0.5
        th = th + a*step; fv = f(m, th); hist.append(fv)
    return th, fv, np.linalg.norm(grad_ok(m, th)), hist
