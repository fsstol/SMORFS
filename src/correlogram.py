import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd

pd.options.display.width = 0

def pairwise_correlation(x, y):
    xm = x - np.mean(x, axis=0, keepdims=True)
    ym = y - np.mean(y, axis=0, keepdims=True)
    return xm.T @ ym /  (np.sqrt(
        np.sum(xm**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.sum(ym**2, axis=0, keepdims=True)))


def correlogram1(df: pd.DataFrame, risk_renames: dict, bounds=None):
    # bounds: None, or one (low, high) per column of df; every panel's x-axis (and y-axis off the diagonal) is then
    # set to them, instead of seaborn's autoscale
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = round(corr_r, 2)
        ax = plt.gca()
        # font_size = 40 + 40 * abs(corr_r)
        font_size = 55
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction", ha='center', va='center', fontsize=font_size)

    def corrfunc(x, y, **kws):
        r, p = stats.pearsonr(x, y)
        p_stars = ''
        if p <= 0.05:
            p_stars = '*'
        if p <= 0.01:
            p_stars = '**'
        if p <= 0.001:
            p_stars = '***'
        ax = plt.gca()
        ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes, color='red', fontsize=50)

    sns.set_theme(style='white', font_scale=1.6)
    g = sns.PairGrid(df, aspect=1.5, diag_sharey=False, despine=False)
    # g.map_lower(sns.kdeplot, cmap='Blues', fill=True, gridsize=100, levels=100, thresh=0.01)
    g.map_lower(sns.kdeplot, cmap='Blues', fill=True, gridsize=100, thresh=0.01)
    # g.map_lower(sns.regplot, lowess=False, ci=None,
    #             line_kws={'color': 'red', 'lw': 1},
    #             scatter_kws={'color': 'black', 's': 20})
    # g.map_diag(sns.distplot, color='black',
    #            kde_kws={'color': 'red', 'cut': 0.7, 'lw': 1},
    #            hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
    # g.map_diag(sns.displot, color='black')
    g.map_diag(sns.histplot, kde=True, kde_kws=dict(cut=0.7), alpha=.4, edgecolor=(1, 1, 1, .4))
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(corrdot)
    g.map_upper(corrfunc)
    g.figure.subplots_adjust(wspace=0, hspace=0)

    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if bounds is not None:
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                g.axes[i, j].set_xlim(bounds[j])
                if i != j:
                    g.axes[i, j].set_ylim(bounds[i])

    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        name = risk_renames.get(col, col)
        ax.set_title(name, y=0.82, fontsize=26)

def correlogram2(df: pd.DataFrame, risk_renames: dict):
    sns.set(style='white')

    def corrfunc(x, y, **kws):
        r, p = stats.pearsonr(x, y)
        p_stars = ''
        if p <= 0.05:
            p_stars = '*'
        if p <= 0.01:
            p_stars = '**'
        if p <= 0.001:
            p_stars = '***'
        ax = plt.gca()
        ax.annotate('r = {:.2f} '.format(r) + p_stars, xy=(0.05, 0.9), xycoords=ax.transAxes)

    def annotate_colname(x, **kws):
        ax = plt.gca()
        name = risk_renames.get(x.name, x.name)
        ax.annotate(name, xy=(0.05, 0.9), xycoords=ax.transAxes, fontweight='bold')

    def cor_matrix(df):
        g = sns.PairGrid(df, palette=['red'])
        # Use normal regplot as `lowess=True` doesn't provide CIs.
        g.map_upper(sns.regplot, scatter_kws={'s':10})
        g.map_diag(sns.histplot, kde=True, kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
        g.map_diag(annotate_colname)
        g.map_lower(sns.kdeplot, cmap='Blues_d')
        g.map_lower(corrfunc)
        # Remove axis labels, as they're in the diagonals.
        for ax in g.axes.flatten():
            ax.set_ylabel('')
            ax.set_xlabel('')
        return g

    cor_matrix(df)
