import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(0)

n_cancers = 30

# Optimization parameters
verbose = False
hot_start_load = True
hot_start_save = False

epochs_grad = 0
epochs_hess = 0

lr_exp = 3e-8
lr_weibull = lr_exp/10
lr_newton = 0.25
# Newton epochs (epochs_hess, an upper bound on the iterations): full saddle-free Newton steps with backtracking, the
# default; newton_damped = True restores v2's schedule, lr_newton/sqrt(n) times the saddle-free step with no line
# search, which compensated for derivative errors now fixed and needs about 1,650 epochs to converge
newton_damped = False
newton_tol = 1e-5  # stop when the norm of the analytic gradient is below this
newton_max_step = 1.0  # step cap: the step is scaled so that no parameter moves by more than this (log scale); None: none
newton_max_halvings = 20  # backtracking: halve until the objective decreases enough; after this many, stop

# IMIS parameters
run_IMIS = True
load_IMIS = False  # sample (v2: True reloaded the 20 saved runs instead)
save_IMIS = True  # save this run's samples and weights to ../output/IMIS2/samples/

epochs_IMIS = 100
n0 = 10000  # initial prior samples
b = 1000  # new samples per iteration
# epochs_IMIS = 2
# n0 = 100  # initial samples
# b = 100  # new samples per iteration
delta = 1e-1  # identity factor in initial covariance
ptime = 1e-1  # pseudo-time per epoch
steps = 5  # number of pseudo-time subdivisions
deg = 3  # t-distribution degrees of freedom


# Model parameters
months_cutoff = 2
t0 = 70.0
dt = t0 / 1e3
t = np.arange(0.0, 100.0, dt)
infinity = 1e20

target_names = ["clinical incidence", "clinical mortality", "preclinical mortality", "preclinical other mortality"]

id_col = "Patient ID"
site_col = "Site recode ICD-O-3/WHO 2008"
site_col_mult = "Site recode ICD-O-3/WHO 2008 (for SIRs) (Event Variable)"
site_col_rates = "Site recode ICD-O-3/WHO 2008 (individual)"
age_col = "Age recode with single ages and 90+"
# age_col_rates = "Age recode with <1 year olds"
age_col_rates = "Age recode with single ages and 85+"
stage_col = "Combined Summary Stage (2004+)"
# diag_col = "Year of diagnosis (2004-2020 by 5)"
diag_col = "Year of diagnosis"

# age_groups = {"00 years": (0.0, 1.0), "01-04 years": (1.0, 5.0)}
# age_groups.update({f"{y:02d}-{y+4:02d} years": (float(y), float(y)+5.0) for y in range(5, 85, 5)})
age_groups = {f"{y:02d} years": (float(y), float(y)+1.0) for y in range(0, 85, 1)}
age_groups["85+ years"] = (85.0, 100.0)
# age_groups["inf"] = (100.0, 120.0)

# ages_remove = ["00 years", "01-04 years"] + [f"{y:02d}-{y+4:02d} years" for y in range(5, 35, 5)]
ages_remove = []
ages_remove.extend(f"{y:02d} years" for y in range(0, 35))
ages_remove.extend(f"{y:02d} years" for y in range(81, 85))
ages_remove.append("85+ years")

# ages_eval = {f"{y-1:02d} years" for y in [40, 60, 80]}  # 39, 59, 79
ages_eval = {f"{y:02d} years" for y in range(35, 81)}  # TODO: fix

cancer_groups = {
    "Colorectal": ["Sigmoid Colon", "Ascending Colon", "Transverse Colon", "Descending Colon", "Rectum", "Rectosigmoid Junction", "Cecum", "Splenic Flexure"],
}

lambda0 = 1.0e-6
Weibull_map = {
    "Default": (8.0, 70.0),
}

map_groups = dict()
for group, sites in cancer_groups.items():
    map_groups.update({s: group for s in sites})

horizon = 2021-2004+1  # usable horizon with staging
horizon_full = 2021-1975+1  # full horizon


def get_survival_function():
    # Life table source: https://www.ssa.gov/oact/STATS/table4c6.html
    lifetable = pd.read_csv("../data/US_life_table_2021.csv", index_col="Age")["Total"] / 1e5
    survival: pd.Series = lifetable
    n_t = len(t)
    diff = np.abs(t.reshape((n_t, 1)) - survival.index.values)
    idx = diff.argmin(axis=1)
    survival = survival[idx].values
    return survival


def autoscale(ax=None, axis='y', margin_lo=0.1, margin_hi=0.1):
    '''Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf
    def setlim(a, b):
        return

    for artist in ax.collections + ax.lines:
        x,y = get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = min(low, newlow)
        newhigh = max(high, newhigh)

    setlim(newlow*margin_lo, newhigh*margin_hi)


def calculate_new_limit(fixed, dependent, limit):
    '''Calculates the min/max of the dependent axis given
    a fixed axis with limits
    '''
    mask = (fixed >=limit[0]) & (fixed <= limit[1])
    window = dependent[mask]
    window = window[~np.isnan(window)]  # targets not drawn (SingularCancer.plot_suppress_events) are NaN
    if len(window) > 0:
        low, high = window.min(), window.max()
    else:
        low, high = np.inf, -np.inf
    return low, high


def get_xy(artist):
    '''Gets the xy coordinates of a given artist
    '''
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y


def restart_plot():
    plt.clf()
    fig, axes = plt.subplots(2, 2, squeeze=False)
    fig.set_size_inches(12, 10)

    major_ticks = np.arange(0, 121, 10)
    minor_ticks = np.arange(0, 121, 5)
    axes = axes.flatten()
    for ax in axes:
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)


def save_plot(name):
    fig = plt.gcf()
    axes = fig.get_axes()
    for ax in axes:
        plt.sca(ax)
        plt.xlim((35.0, 80.0))
        autoscale(margin_lo=0.9, margin_hi=1.1)
        plt.legend()
    plt.tight_layout()
    # plt.savefig(f"../output/{name}.png", dpi=300)
    plt.savefig(f"../output/{name}.pdf", dpi=300)
