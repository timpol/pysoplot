"""
Monte Carlo error propagation functions

"""

import numpy as np
import matplotlib.pyplot as plt

from . import cfg
from . import plotting


# Fail codes
NON_CONVERGENCE = 1
NEGATIVE_AGE = 2
NEGATIVE_AR_SIM = 3
NEGATIVE_AR_SOL = 4


def draw_theta(fit, trials, flags):
    """
    Draw random linear regression slope and intercept values.
     """
    if fit['type'] == 'classical' and fit['excess_scatter']:
        df = fit['n'] - 2
        theta = multivariate_t_rvs(fit['theta'], fit['covtheta'], df, trials)
    else:
        theta = cfg.rng.multivariate_normal(
            fit['theta'], fit['covtheta'], trials
        )
    return np.transpose(theta), flags


def multivariate_t_rvs(m, S, df, n):
    """
    Generate random variables of multivariate t distribution

    Code based on:
    <https://github.com/statsmodels/statsmodels/blob/master/statsmodels/
    sandbox/distributions/multivariate.py#L90>
    Originally written by Enzo Michelangeli, style changes by josef-pktd.

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    """
    m = np.asarray(m)
    d = len(m)

    if df == np.inf:
        x = 1.
    else:
        x = cfg.rng.chisquare(df, n) / df
    z = cfg.rng.multivariate_normal(np.zeros(d), S, (n,))
    # same output format as random.multivariate_normal
    return m + z / np.sqrt(x)[:, None]


def update_flags(flags, cond, code_num):
    """
    Update array of failed iteration flags.
    """
    if len(cond) != len(flags):
        raise ValueError('flags0, cond, and msk  must be of equal length')
    # Do not overwrite code if iteration is already flagged.
    cond = np.logical_and(cond, flags == 0)
    flags = np.where(cond, code_num, flags)
    return flags


def check_ages(t, c, flags, negative_ages=False):
    """
    Check simulated ages have converged properly.

    Parameters
    ----------
    t : np.ndarray
        Simulated ages.
    c : np.ndarray, logical
        Convergence status.
    """
    # check convergence:
    flags = update_flags(flags, np.logical_or(~c, np.isnan(t)), NON_CONVERGENCE)
    # check negative ages:
    if not negative_ages:
        flags = update_flags(flags, t < 0., NEGATIVE_AGE)
    return flags


def summary_stats(x, p=0.95):
    """Compute summary statistics from array of Monte Carlo simulated values.
    """
    p = p * 100  # convert to percent
    x_av = np.nanmean(x)
    sx = np.nanstd(x)

    # Calcaulte upper and lower limits on coverage interval.
    x_lower = np.nanpercentile(x, (100 - p) / 2)
    x_upper = np.nanpercentile(x, (100 - p) / 2 + p)
    x_pm = np.nanmean((x_upper - x_av, x_av - x_lower))

    return x_av, sx, (x_lower, x_upper), x_pm


def histogram(ax, y, xlabel=None):
    """
    Plot histogram of mc simulated values.
    """
    n = int(np.ceil(np.sqrt(y.shape[0])))
    n = n if n < 1000 else 1000
    ax.hist(y, bins=n, density=True, **cfg.hist_bars_kw)
    ax.set_xlabel(xlabel)

    # if plot_gaussian_fit:
    #     mu = np.nanmean(y)
    #     sigma = np.nanstd(y)
    #     xx = np.linspace(*ax.get_xlim(), 300)
    #     if sigma != 0.:
    #         ax.plot(xx, stats.norm.pdf(xx, mu, sigma),
    #                 **HistOpt.gaussian_line_kw)


def scatterplot(ax, x, y, xlabel=None, ylabel=None):
    """
    Create scatterplot of correlated mc simulated values.
    """
    ax.plot(x, y, **cfg.scatter_markers_kw)
    # reset axis limits as default doesn't always seem to work very well...
    xmin, xmax = x.min(), x.max()
    dx = xmax - xmin
    ymin, ymax = y.min(), y.max()
    dy = ymax - ymin
    ax.set_xlim((xmin - 0.1 * dx, xmax + 0.1 * dx))
    ax.set_ylim((ymin - 0.1 * dy, ymax + 0.1 * dy))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plotting.apply_plot_settings(ax.get_figure(), plot_type='hist',
            tight_layout=True, diagram=None)


def age_hist(ts, age_type, a=None, b=None, x=None, y=None):
    """
    Create histogram plots of simulated regression parameters and ages.
    """
    assert age_type in ('tw', 'wc', 'iso-Pb6U8', 'iso-Pb7U5', 'Pb6U8', 'Pb7U5',
                        'mod207Pb')

    fig, ax = plt.subplots(2, 2, **cfg.fig_kw)

    histogram(ax[0, 0], ts, xlabel='age (Ma)')
    if age_type in ('tw', 'wc') or age_type.startswith('iso'):
        histogram(ax[1, 0], a, xlabel='y-int.')
        histogram(ax[0, 1], b, xlabel='slope')
        scatterplot(ax[1, 1], b, a, xlabel='slope', ylabel='y-int.')
    elif age_type == 'Pb6U8':
        histogram(ax[1, 0], x, xlabel='$^{206}$Pb/$^{238}$U')
    elif age_type == 'Pb7U5':
        histogram(ax[1, 0], x, xlabel='$^{207}$Pb/$^{235}$U')
    elif age_type == 'mod207Pb':
        histogram(ax[1, 0], x, xlabel='$^{206}$Pb/$^{238}$U')
        histogram(ax[0, 1], y, xlabel='$^{207}$Pb/$^{235}$U')
        scatterplot(ax[1, 1], x, y, xlabel='$^{206}$Pb/$^{238}$U',
                    ylabel='$^{207}$Pb/$^{235}$U')

    for a in ax.flatten():
        plotting.apply_plot_settings(a.get_figure(), plot_type='hist',
                tight_layout=True, diagram=None)
    fig.tight_layout()

    return fig


def fXU_hist(age_type, flags, fThU=None, fPaU=None,):
    """
    Plot simulated fractionation factors for Guillong and Sakata ages.
    """
    assert (fThU is not None) or (fPaU is not None)
    good = flags == 0
    fig, ax = plt.subplots(2, 2, **cfg.fig_kw)
    ax = ax.flatten()
    i = 0
    if fThU is not None:
        histogram(ax[i], fThU[good], xlabel="fThU")
        i += 1
    if fPaU is not None:
        histogram(ax[i], fPaU[good], xlabel='fPaU')
    return fig


def ar_hist(A238, A235, init=(True, True)):
    """
    Create histogram of simulated activity ratio values for successful
    iterations.
    """
    fig, ax = plt.subplots(2, 2, **cfg.fig_kw)
    axx = ax.flatten()
    i = 0
    if A238 is not None:
        histogram(axx[i], A238[0])
        axx[i].set_xlabel("%s\n[$^{234}$U/$^{238}$U]" %
                          ('initial' if init[0] else 'present-day'))
        i += 1
        histogram(axx[i], A238[1])
        axx[i].set_xlabel("%s\n[$^{230}$Th/$^{238}$U]" %
                          ('initial' if init[1] else 'present-day'))
        i += 1
        histogram(axx[i], A238[2])
        axx[i].set_xlabel("initial\n[$^{226}$Ra/$^{238}$U]")
        i += 1
    if A235 is not None:
        histogram(axx[i], A235)
        axx[i].set_xlabel("initial\n[$^{231}$Pa/$^{235}$U]")

    for a in ax.flatten():
        plotting.apply_plot_settings(fig, plot_type='hist',
                     norm_isotope='204Pb', diagram=None)
    return fig


def ar_sol_hist(t, A48_init, A08_init):
    """
    Create histograms of initial activity ratio solutions.
    """
    fig, ax = plt.subplots(2, 2, **cfg.fig_kw)
    i = 0
    if A48_init is not None:
        histogram(ax[i][0], A48_init)
        ax[i][0].set_xlabel("initial\n[$^{234}$U/$^{238}$U] soln.")
        scatterplot(ax[i][1], A48_init, t)
        ax[i][1].set_ylabel('age (Ma)')
        ax[i][1].set_xlabel("initial\n[$^{234}$U/$^{238}$U] soln.")
        i += 1
    if A08_init is not None:
        histogram(ax[i][0], A08_init)
        ax[i][0].set_xlabel("initial\n[$^{230}$Th/$^{238}$U] soln.")
        scatterplot(ax[i][1], A08_init, t)
        ax[i][1].set_ylabel('age (Ma)')
        ax[i][1].set_xlabel("initial\n[$^{230}$Th/$^{238}$U] soln.")
        i += 1

    for a in ax.flatten():
        plotting.apply_plot_settings(fig, plot_type='isochron',
                     norm_isotope='204Pb', diagram=None)
    return fig


def intercept_plot_axis_limits(ax, x, y, diagram='tw', min_yspan=1e-03,
               tw_f=((3, 3), (2, 5)), wc_f=((3, 3), (3, 3))):
    """
    Naive estimate of reasonable concordia intercept axis limits.
    """
    f = tw_f if diagram == 'tw' else wc_f
    ymin, ymed, ymax = np.quantile(y, (0.05, 0.50, 0.95))
    xmin, xmed, xmax = np.quantile(x, (0.05, 0.50, 0.95))
    yspan = ymax - ymin
    xspan = xmax - xmin

    if diagram == 'tw' and yspan < min_yspan:
        yspan = min_yspan

    xmin = xmin - f[0][0] * xspan
    xmax = xmax + f[0][1] * xspan
    ymin = ymin - f[1][0] * yspan
    ymax = ymax + f[1][1] * yspan

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)


def intercept_plot_ax_limits(ax, b, x, y, diagram='tw', min_yspan = 0.001,
                             maxiter=100):
    """
    Compute intercept plot axis limits.
    """
    ymin, ymed, ymax = np.quantile(y, (0.025, 0.50, 0.975))
    xmin, xmed, xmax = np.quantile(x, (0.025, 0.50, 0.975))
    yspan = ymax - ymin
    xspan = xmax - xmin

    dx = 3 * xspan
    ax.set_xlim((xmin - dx, xmax + dx))

    i = 0
    fu = 6 if diagram == 'tw' else 3
    fl = 3
    while (ymax - ymin) < min_yspan and i < maxiter:
        i += 1
        yspan *= 2
        ymin = ymin - fl * yspan
        ymax = ymax + fu * yspan

    ax.set_ylim((ymin, ymax))
