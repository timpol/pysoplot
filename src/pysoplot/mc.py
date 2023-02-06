"""
Monte Carlo uncertainty propagation functions.

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

from . import ludwig
from . import minimise
from . import cfg
from . import concordia
from . import plotting
from . import useries
from . import stats


# Fail codes
NON_CONVERGENCE = 1
NEGATIVE_AGE = 2
NEGATIVE_RATIO_SIM = 3
NEGATIVE_RATIO_SOL = 4


def draw_theta(fit, trials, failures):
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
    return np.transpose(theta), failures


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


def update_failures(failures, cond, code_num):
    """
    Update array of failed iteration codes without overwriting pre-existing
    codes.
    """
    if len(cond) != len(failures):
        raise ValueError('flags0, cond, and msk  must be of equal length')
    # Do not overwrite code if iteration is already flagged.
    cond = np.logical_and(cond, failures == 0)
    failures = np.where(cond, code_num, failures)
    return failures


def check_ages(t, c, failures, negative_ages=False):
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
    failures = update_failures(failures, np.logical_or(~c, np.isnan(t)), NON_CONVERGENCE)
    # check negative ages:
    if not negative_ages:
        failures = update_failures(failures, t < 0., NEGATIVE_AGE)
    return failures


def summary_stats(x, p=0.95):
    """
    Compute summary statistics from array of Monte Carlo simulated values.
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
    ax.set_xlabel(xlabel, **cfg.axis_labels_kw)

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
    plotting.apply_plot_settings(ax.get_figure(), plot_type='hist')


def age_hist(ts, age_type, a=None, b=None, x=None, y=None):
    """
    Create histogram plots of simulated regression parameters and ages.
    """
    assert age_type in ('tw', 'wc', 'iso-206Pb', 'iso-207Pb', '206Pb*', '207Pb*',
                        'mod207Pb')

    fig, ax = plt.subplots(2, 2, **cfg.fig_kw)

    histogram(ax[0, 0], ts, xlabel='age (Ma)')
    if age_type in ('tw', 'wc') or age_type.startswith('iso'):
        histogram(ax[1, 0], a, xlabel='y-int.')
        histogram(ax[0, 1], b, xlabel='slope')
        scatterplot(ax[1, 1], b, a, xlabel='slope', ylabel='y-int.')
    elif age_type == '206Pb*':
        histogram(ax[1, 0], x, xlabel='$^{206}$Pb/$^{238}$U')
    elif age_type == '207Pb*':
        histogram(ax[1, 0], x, xlabel='$^{207}$Pb/$^{235}$U')
    elif age_type == 'mod207Pb':
        histogram(ax[1, 0], x, xlabel='$^{206}$Pb/$^{238}$U')
        histogram(ax[0, 1], y, xlabel='$^{207}$Pb/$^{235}$U')
        scatterplot(ax[1, 1], x, y, xlabel='$^{206}$Pb/$^{238}$U',
                    ylabel='$^{207}$Pb/$^{235}$U')

    for a in ax.flatten():
        plotting.apply_plot_settings(a.get_figure(), plot_type='hist')
    fig.tight_layout()

    return fig


def ratio_hist(A238, A235, init=(True, True)):
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


def ratio_solution_hist(t, A48_init, A08_init):
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


# ===========================================================================
# Disequilibrium U-Pb age functions
# ===========================================================================

def concint_diseq_age(t, fit, A, sA, init, trials=50_000, dc_errors=False,
        diagram='tw', u_errors=False, negative_ratios=True, negative_ages=True,
        hist=(False, False), intercept_plot=True,
        intercept_plot_kw=None, conc_kw=None):
    """
    Propagate disequilibrium U-Pb concordia-intercept age uncertainties using
    Monte Carlo simulation.

    Parameters
    ----------
    t : float
        age solution
    fit : dict
        linear regression fit parameters
    A : np.ndarray
        one-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U]
    sA : np.ndarray
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A
    init : array-like
        two-element list of boolean values, the first is True if [234U/238U]
        is an initial value and False if a present-day value, the second is True
        if [230Th/238U] is an initial value and False if a present-day value

    """
    assert diagram == 'tw'  # wc not yet implemented
    failures = np.zeros(trials, dtype='uint8')

    if intercept_plot_kw is None:
        intercept_plot_kw = {}
    if conc_kw is None:
        conc_kw = {}

    theta, failures = draw_theta(fit, trials, failures)
    U = cfg.rng.normal(cfg.U, cfg.sU, trials) if u_errors else cfg.U
    Lam238, failures = draw_decay_const(trials, failures, dc_errors=dc_errors)
    Lam235, failures = draw_decay_const(trials, failures, dc_errors=dc_errors,
                                     series='235U')
    coef238 = ludwig.bateman(Lam238, series='238U')
    coef235 = ludwig.bateman(Lam235, series='235U')

    pA = np.empty((4, trials))
    pA[:3, :], failures = draw_ar(A[:-1], sA[:-1], trials, failures, positive_only=not negative_ratios)
    pA[3, :], failures = draw_ar(A[-1], sA[-1], trials, failures, series='235U',
                              positive_only=not negative_ratios)

    args = (*theta, pA[:3], pA[-1], Lam238, Lam235, coef238, coef235, U)

    # run vectorised Newton routine:
    fmin, dfmin = minimise.concint(diagram='tw', init=init)
    ts, c, zd = optimize.newton(fmin, np.ones(trials) * t, dfmin, args=args,
                                full_output=True)
    failures = check_ages(ts, c, failures, negative_ages=negative_ages)

    # get initial activity ratio solutions
    a234_238_i, a230_238_i = useries.init_ratio_solutions(ts, pA[:3, :], init, Lam238)

    if not negative_ratios:
        if a234_238_i is not None:
            failures = update_failures(failures, a234_238_i < 0., NEGATIVE_RATIO_SOL)
        if a230_238_i is not None:
            failures = update_failures(failures, a230_238_i < 0., NEGATIVE_RATIO_SOL)

    ok = failures == 0
    if sum(ok) == 0:
        raise RuntimeError('all Monte Carlo simulation trials failed')

    age_95ci = np.quantile(ts[ok], (0.025, 0.975))

    # compute results
    results = {
        'age_type': 'concordia-intercept',
        'age_1s': np.std(ts[ok]),
        'age_95ci': age_95ci,
        'age_95pm': np.nanmean([t - age_95ci[0], age_95ci[1] - t]),
        'mean_age': np.nanmean(ts[ok]),
        'median_age': np.nanmedian(ts[ok]),
        'trials': trials,
        'fails': np.sum(failures != 0),
        'not_converged': np.sum(failures == NON_CONVERGENCE),
        'negative_ages': np.sum(failures == NEGATIVE_AGE),
        'negative_ratios': np.sum(failures == NEGATIVE_RATIO_SIM),
        'negative_ratio_soln': np.sum(failures == NEGATIVE_RATIO_SOL)
    }

    if a234_238_i is not None:
        results['mean_[234U/238U]_i'] = np.nanmean(a234_238_i[ok])
        results['median_[234U/238U]_i'] = np.nanmedian(a234_238_i[ok])
        results['[234U/238U]i_1sd'] = np.std(a234_238_i[ok])
        results['[234U/238U]i_95ci'] = np.quantile(a234_238_i[ok], (0.025, 0.975))
        results['cor_age_[234U/238U]i'] = np.corrcoef(np.row_stack((a234_238_i[ok],
                                                a234_238_i[ok])))[0, 1]

    if a230_238_i is not None:
        results['mean_[230Th/238U]_i'] = np.nanmean(a230_238_i[ok])
        results['median_[230Th/238U]_i'] = np.nanmedian(a230_238_i[ok])
        results['[230Th/238U]i_1sd'] = np.nanstd(a230_238_i[ok])
        results['[230Th/238U]i_95ci'] = np.quantile(a230_238_i[ok], (0.025, 0.975))
        results['cor_age_[230Th/238U]i'] = np.corrcoef(np.row_stack((a230_238_i[ok],
                                                                     a230_238_i[ok])))[0, 1]

    # compile plots:
    if intercept_plot:
        fig = diseq_intercept_plot(ts, fit, pA[:3, :], pA[3], Lam238, Lam235, coef238, coef235,
                                   U, failures, dp=None, init=init, dc_errors=dc_errors,
                                   u_errors=u_errors, **intercept_plot_kw)
        ax = fig.get_axes()[0]
        # TODO: allow concordia func to accept pre-simulated A values to account
        # rejected values...
        concordia.plot_diseq_concordia(ax, A, init, diagram, sA=sA,
                                       negative_ratios=negative_ratios, **conc_kw)
        results['fig'] = fig

    if any(hist):
        if hist[0]:
            a, b = theta
            xs, ys = concordia.diseq_xy(ts, pA, init, diagram)
            fig = age_hist(ts[ok], diagram, a[ok], b[ok], xs[ok], ys[ok])
            results['age_hist'] = fig
        if hist[1]:
            fig = ratio_hist(np.transpose(np.transpose(pA[:3])[ok]),
                             pA[-1][ok], init)
            results['ratio_hist'] = fig
        if hist[1] and (a234_238_i is not None or a230_238_i is not None):
            if a234_238_i is not None:
                a234_238_i = a234_238_i[ok]
            if a230_238_i is not None:
                a230_238_i = a230_238_i[ok]
            fig = ratio_solution_hist(ts[ok], a234_238_i, a230_238_i)
            results['ratio_hist'] = fig

    return results


def isochron_diseq_age(t, fit, A, sA, init=(True, True), trials=50_000,
        dc_errors=False, negative_ratios=True, negative_ages=True,
        hist=(False, False), age_type='iso-206Pb'):
    """
    Monte Carlo disequilibrium "classical" isochron age uncertainties.

    """
    assert age_type in ('iso-206Pb', 'iso-207Pb')
    failures = np.zeros(trials, dtype='uint8')

    theta, failures = draw_theta(fit, trials, failures)

    if age_type == 'iso-206Pb':
        pA, failures = draw_ar(A, sA, trials, failures,
                               positive_only=not negative_ratios)
        Lam, failures = draw_decay_const(trials, failures, dc_errors=dc_errors)
        coef = ludwig.bateman(Lam, series='238U')
    else:
        pA, failures = draw_ar(A, sA, trials, failures, series='235U',
                            positive_only=not negative_ratios)
        Lam, failures = draw_decay_const(trials, failures, dc_errors=dc_errors,
                                         series='235U')
        coef = ludwig.bateman(Lam, series='235U')

    args = (theta[1], pA, Lam, coef)

    # run vectorised Newton routine:
    fmin, dfmin = minimise.isochron(age_type=age_type, init=init)
    ts, c, zd = optimize.newton(fmin, np.ones(trials) * t, dfmin, args=args,
                                full_output=True)
    failures = check_ages(ts, c, failures, negative_ages=negative_ages)

    # back-calculate initial activity ratio solutions
    if age_type == 'iso-206Pb':
        a234_238_i, a230_238_i = useries.init_ratio_solutions(ts, pA, init, Lam)
        if not negative_ratios:
            if a234_238_i is not None:
                failures = update_failures(failures, a234_238_i < 0.,
                               NEGATIVE_RATIO_SOL)
            if a230_238_i is not None:
                failures = update_failures(failures, a230_238_i < 0.,
                               NEGATIVE_RATIO_SOL)

    ok = failures == 0
    if np.sum(ok) == 0:
        raise RuntimeError('all Monte Carlo simulation trials failed')

    age_95ci = np.quantile(ts[ok], (0.025, 0.975))
    # compute results
    results = {
        'age_type': f'{age_type} isochron age',
        'age_1s': np.nanstd(ts[ok]),
        'age_95ci': age_95ci,
        'age_95pm': np.nanmean((t - age_95ci[0], age_95ci[1] - t)),
        'mean_age': np.nanmean(ts[ok]),
        'median_age': np.nanmedian(ts[ok]),
        'trials': trials,
        'fails': np.sum(failures != 0),
        'not_converged': np.sum(failures == NON_CONVERGENCE),
        'negative_ages': np.sum(failures == NEGATIVE_AGE),
        'negative_ratios': np.sum(failures == NEGATIVE_RATIO_SIM),
        'negative_ratio_soln': np.sum(failures == NEGATIVE_RATIO_SOL)
    }

    if age_type == 'iso-206Pb':
        if a234_238_i is not None:
            results['mean_[234U/238U]_i'] = np.nanmean(a234_238_i[ok])
            results['median_[234U/238U]_i'] = np.nanmedian(a234_238_i[ok])
            results['[234U/238U]i_1sd'] = np.nanstd(a234_238_i[ok])
            results['[234U/238U]i_95ci'] = np.quantile(a234_238_i[ok], (0.025, 0.975))
            results['cor_[234U/238U]i_t'] = \
                np.corrcoef(np.row_stack((ts[ok], a234_238_i[ok])))[0, 1]

        if a230_238_i is not None:
            results['mean_[230Th/238U]_i'] = np.nanmean(a230_238_i[ok])
            results['median_[230Th/238U]_i'] = np.nanmedian(a230_238_i[ok])
            results['[230Th/238U]i_1sd'] = np.nanstd(a230_238_i[ok])
            results['[230Th/238U]i_95ci'] = np.quantile(a230_238_i[ok], (0.025, 0.975))
            results['cor_[230Th/238U]i_t'] = \
                np.corrcoef(np.row_stack((ts[ok], a230_238_i[ok])))[0, 1]

    # compile plots
    if any(hist):
        if hist[0]:
            a, b = theta
            fig = age_hist(ts[ok], age_type, a[ok], b[ok])
            results['age_hist'] = fig
        if hist[1]:
            A238 = None if age_type != 'iso-206Pb' else pA.T[ok].T
            A235 = None if age_type != 'iso-207Pb' else pA.T[ok].T
            fig = ratio_hist(A238, A235, init=init)
            results['ratio_hist'] = fig
        if age_type == 'iso-206Pb':
            if hist[1] and (not init[0] or not init[1]):
                if not init[0]:
                    a234_238_i = a234_238_i[ok]
                if not init[1]:
                    a230_238_i = a230_238_i[ok]
                fig = ratio_solution_hist(ts[ok], a234_238_i, a230_238_i)
                results['ratio_hist'] = fig

    return results


def forced_concordance(t57, A48i, fit_57, fit_86, A, sA, init, trials=50_000,
            negative_ratios=True, negative_ages=True, hist=(0, 0, 0)):
    """
    Monte Carlo uncertainties for forced-concordance initial [234U/238U] value.
    
    """
    assert init[1], 'expected initial [230Th/238U] value'
    
    failures = np.zeros(trials)
    theta57 = cfg.rng.multivariate_normal(fit_57['theta'], fit_57['covtheta'], trials)
    theta86 = cfg.rng.multivariate_normal(fit_86['theta'], fit_86['covtheta'], trials)

    pA = np.empty((4, trials))
    pA[-1, :], failures = draw_ar(A[-1], sA[-1], trials, failures, series='235U',
                               positive_only=not negative_ratios)
    pA[:3, :], failures = draw_ar(A[:-1], sA[:-1], trials, failures, positive_only=not negative_ratios)

    Lam238, failures = draw_decay_const(trials, failures, dc_errors=False)
    Lam235, failures = draw_decay_const(trials, failures, dc_errors=False, series='235U')
    coef238 = ludwig.bateman(Lam238, series='238U')
    coef235 = ludwig.bateman(Lam238, series='235U')

    args57 = (theta57[:, 1], pA[-1], Lam235, coef235)

    # run vectorised Newton routine to get iso-57 ages
    fmin, dfmin = minimise.isochron(age_type='iso-207Pb')
    ts, c, zd = optimize.newton(fmin, np.ones(trials) * t57, dfmin, args=args57,
                                full_output=True)
    failures = check_ages(ts, c, failures, negative_ages=negative_ages)

    # run vectorised Newton routine to get initial [234U/238U] values:
    fmin, dfmin = minimise.concordant_A48()
    A48i_s, c, zd = optimize.newton(fmin, np.full(trials, A48i), dfmin,
                args=(ts, theta86[:, 1], pA[:3], Lam238, coef238), full_output=True)
    failures = check_ages(A48i_s, c, failures, negative_ages=negative_ratios)  # pretend ar48i solutions are ages

    ok = (failures == 0)
    if np.sum(ok) == 0:
        raise RuntimeError('all Monte Carlo simulation trials failed')
    age_95ci = np.quantile(ts[ok], (0.025, 0.975))
    A48i_95ci = np.quantile(A48i_s[ok], (0.025, 0.975))

    # compute results
    results = {
        'age_type': 'forced-concordance [234U/238U]_i',
        '207Pb_age_1sd': np.nanstd(ts[ok]),
        '207Pb_age_95ci': age_95ci,
        '207Pb_age_95pm': [np.nanmean((age_95ci[1] - t57, t57 - age_95ci[0]))],
        'mean_207Pb_age': np.nanmean(ts[ok]),
        'median_207Pb_age': np.nanmedian(ts[ok]),
        '[234U/238U]i_1sd': np.nanstd([A48i_s[ok]]),
        '[234U/238U]i_95ci': A48i_95ci,
        '[234U/238U]i_95pm': [np.nanmean((A48i_95ci[1] - A48i, A48i - A48i_95ci[0]))],
        'mean_[234U/238U]i': np.nanmean([A48i_s[ok]]),
        'median_[234U/238U]i': np.nanmedian([A48i_s[ok]]),
        'trials': trials,
        'fails': np.sum(failures != 0),
        'not_converged': np.sum(failures == NON_CONVERGENCE),
        'negative_ages': np.sum(failures == NEGATIVE_AGE),
        'negative_ratios': np.sum(failures == NEGATIVE_RATIO_SIM)
    }

    if any(hist):
        if hist[0]:
            a, b = np.transpose(theta57)
            fig = age_hist(ts[ok], 'iso-207Pb', a[ok], b[ok])
            results['age_hist'] = fig
        if hist[1]:
            A238 = [A48i_s[ok], *pA[1:3].T[ok].T]
            A235 = pA[-1].T[ok].T
            fig = ratio_hist(A238, A235)
            results['ratio_hist'] = fig

    return results


def pbu_diseq_age(t, x, Vx, DThU=None, DThU_1s=None, DPaU=None, DPaU_1s=None,
        alpha=None, alpha_1s=None, age_type='206Pb*', trials=50_000,
        negative_ages=False, negative_ratios=False):
    """
    Monte Carlo age uncertainties for single aliquot Pb/U or 207Pb-corrected
    ages where Th/U_min is determined iteratively from rad. 208Pb/206Pb.

    """
    assert age_type in ('206Pb*', '207Pb*', 'cor207Pb'), 'age_type not recognised'

    if np.any(np.isnan(t)):  # nan ages should be removed
        raise ValueError('cannot run Monte Carlo simulation if nans in ages array')

    n = t.size
    failures = np.zeros((trials, n), dtype='uint8')

    # pre-allocate ages arrays to store results
    ts = np.zeros((trials, n))

    fmin, dfmin = minimise.pbu(age_type=age_type)
    if age_type in ('206Pb*', 'cor207Pb'):
        Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        coef238 = ludwig.bateman(Lam238)
    if age_type in ('207Pb*', 'cor207Pb'):
        Lam235 = (cfg.lam235, cfg.lam231)
        coef235 = ludwig.bateman(Lam235, series='235U')

    # Common variables
    if age_type in ('206Pb*', 'cor207Pb'):
        DThU_sim = cfg.rng.normal(DThU, DThU_1s, trials)
    if age_type in ('207Pb*', 'cor207Pb'):
        DPaU_sim = cfg.rng.normal(DPaU, DPaU_1s, trials)
    if age_type == 'cor207Pb':
        alpha_sim = cfg.rng.normal(alpha, alpha_1s, trials)

    if not negative_ratios:
        if age_type in ('206Pb*', 'cor207Pb'):
            cond1 = np.less(DThU_sim, 0).reshape(-1, 1)
            cond2 = failures == 0
            failures = np.where(np.logical_and(cond1, cond2), NEGATIVE_RATIO_SIM,
                                failures)
        if age_type in ('207Pb*', 'cor207Pb'):
            cond1 = np.less(DPaU_sim, 0).reshape(-1, 1)
            cond2 = failures == 0
            failures = np.where(np.logical_and(cond1, cond2), NEGATIVE_RATIO_SIM,
                                failures)

    # (Possibly) correlated variables
    if age_type == 'cor207Pb':
        xy_sim = cfg.rng.multivariate_normal(x.flatten(), Vx, trials)
        x_sim, y_sim = np.split(xy_sim, 2, axis=1)
    else:
        x_sim = cfg.rng.multivariate_normal(x, Vx, trials)

    for i in range(n):
        t0 = np.full(trials, t[i])

        if age_type == '206Pb*':
            args = (x_sim[:, i], [cfg.a234_238_eq, DThU_sim, cfg.a226_238_eq],
                    Lam238, coef238)
        elif age_type == '207Pb*':
            args = (x_sim[:, i], DPaU_sim, Lam235, coef235)
        else:
            args = (x_sim[:, i], y_sim[:, i], [cfg.a234_238_eq, DThU_sim,
                    cfg.a226_238_eq, DPaU_sim], alpha_sim, cfg.U,
                    Lam238, Lam235, coef238, coef235)

        r, c, zd = optimize.newton(fmin, t0, dfmin, args=args, full_output=True,
                                   maxiter=30)

        ts[:, i] = np.where(np.logical_and(c, ~np.isnan(r)), r, np.nan)
        failures[:, i] = update_failures(failures[:, i],
                            np.isnan(ts[:, i]), NON_CONVERGENCE)

        # Check for negative ages.
        if not negative_ages:
            cond = np.logical_and(failures[:, i] == 0, np.isnan(ts[:, i]))
            failures[:, i][cond] = NEGATIVE_AGE

        #TOOD: check for reverse discordant 207Pb-corrected solutions?

    # To compute covariances, trials that fail for one aliquot should
    # discarded for all (?)
    ok = np.all(failures == 0, axis=1)
    if np.sum(ok) == 0:
        raise ValueError('all Monte Carlo trials failed')

    # # compile results
    age_95ci = [np.quantile(t[ok], (0.025, 0.975)) for t in ts.T]
    results = {
        'age_type': age_type,
        'age_1s': [np.nanstd(t[ok]) for t in ts.T],
        'age_95pm': [np.nanmean((t - m, p - t)) for m, p in age_95ci],
        'age_95ci': age_95ci,
        'mean_age': [np.nanmean(t[ok]) for t in ts.T],
        'median_age': [np.nanmedian(t[ok]) for t in ts.T],
        'cov_t': np.cov(np.transpose(ts[ok])),
        'trials': trials,
        'fails': np.sum(failures != 0, axis=0),
        'not_converged': np.sum(failures == NON_CONVERGENCE, axis=0),
        'negative_ages': np.sum(failures == NEGATIVE_AGE, axis=0),
        'negative_ratios': np.sum(failures == NEGATIVE_RATIO_SIM, axis=0),
        'negative_ratio_soln': np.sum(failures == NEGATIVE_RATIO_SOL, axis=0),
    }

    if any('hist'):
        warnings.warn('Monte Carlo histograms not yet implemented for '
                      'single aliquot iterative ages')

    return results


def pbu_iterative_age(t, ThU_min, x, Vx, ThU_melt, ThU_melt_1s, Th232_U238=None,
            V_Th232_U238=None, Pb208_206=None, V_Pb208_206=None, DPaU=None,
            DPaU_1s=None, alpha=None, alpha_1s=None, age_type='206Pb*',
            trials=50_000, negative_ages=False, negative_ratios=False,
            hist=(False, False)):
    """
    Com
    """
    assert age_type in ('206Pb*', 'cor207Pb')

    if np.any(np.isnan(t)):  # nan ages should be removed
        raise ValueError('cannot run Monte Carlo simulation if nans in ages array')

    meas_232Th_238U = True if Th232_U238 is not None else False
    n = t.size
    failures = np.zeros((trials, n), dtype='uint8')

    # pre-allocate ages arrays to store results
    ThU_min_sim = np.zeros((trials, n))
    ts = np.zeros((trials, n))

    fmin, dfmin = minimise.pbu_iterative(meas_232Th_238U=meas_232Th_238U,
                                         age_type=age_type)
    Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    coef238 = ludwig.bateman(Lam238)
    if age_type == 'cor207Pb':
        Lam235 = (cfg.lam235, cfg.lam231)
        coef235 = ludwig.bateman(Lam235, series='235U')

    # Common variables
    ThU_melt_sim = np.random.normal(ThU_melt, ThU_melt_1s, trials)
    if age_type == 'cor207Pb':
        DPaU_sim = cfg.rng.normal(DPaU, DPaU_1s, trials)
        alpha_sim = cfg.rng.normal(alpha, alpha_1s, trials)

    if not negative_ratios:
        cond1 = np.less(ThU_melt_sim, 0).reshape(-1, 1)
        cond2 = failures == 0
        failures = np.where(np.logical_and(cond1, cond2), NEGATIVE_RATIO_SIM,
                            failures)
        if age_type =='cor207Pb':
            cond1 = np.less(DPaU_sim, 0).reshape(-1, 1)
            cond2 = failures == 0
            failures = np.where(np.logical_and(cond1, cond2), NEGATIVE_RATIO_SIM,
                                failures)

    # (Possibly) correlated variables
    if age_type == 'cor207Pb':
        xy_sim = cfg.rng.multivariate_normal(x.flatten(), Vx, trials)
        x_sim, y_sim = np.split(xy_sim, 2, axis=1)
    else:
        x_sim = cfg.rng.multivariate_normal(x, Vx, trials)

    if meas_232Th_238U:
        Th232_U238_sim = cfg.rng.multivariate_normal(
            Th232_U238, V_Th232_U238, trials)
    else:
        Pb208_206_sim = cfg.rng.multivariate_normal(
            Pb208_206, V_Pb208_206, trials)

    for i in range(n):
        t0 = np.full(trials, t[i])

        if meas_232Th_238U:
            if age_type == '206Pb*':
                args = (x_sim[:, i], Th232_U238_sim[:, i], ThU_melt_sim, Lam238,
                        coef238)
            else:
                args = (x_sim[:, i], y_sim[:, i], Th232_U238_sim[:, i],
                        ThU_melt_sim, DPaU_sim, alpha_sim, Lam238, Lam235,
                        coef238, coef235)
        else:
            if age_type == '206Pb*':
                args = (x_sim[:, i], Pb208_206_sim[:, i], ThU_melt_sim, Lam238,
                        coef238)
            else:
                args = (x_sim[:, i], y_sim[:, i], Pb208_206_sim[:, i],
                        ThU_melt_sim, DPaU_sim, alpha_sim, Lam238, Lam235,
                        coef238, coef235)

        r, c, zd = optimize.newton(fmin, t0, dfmin, args=args, full_output=True,
                                   maxiter=30)

        ts[:, i] = np.where(np.logical_and(c, ~np.isnan(r)), r, np.nan)
        failures[:, i] = update_failures(failures[:, i],
                            np.isnan(ts[:, i]), NON_CONVERGENCE)

        f1, f2, _, f4 = ludwig.f_comp(ts[:, i], [cfg.a234_238_eq, np.nan,
                            cfg.a226_238_eq], Lam238, coef238)
        ThU_min_sim[:, i] = ThU_melt_sim * (x_sim[:, i] - (f1 + f2 + f4)) / (Lam238[0]/Lam238[2]
                                * (coef238[7] * np.exp((Lam238[0]-Lam238[2]) * ts[:, i])
                                + coef238[8] * np.exp((Lam238[0]-Lam238[3]) * ts[:, i])
                                + np.exp(Lam238[0] * ts[:, i])))
        
        # Check for negative ages or ratios.
        if not negative_ages:
            cond = np.logical_and(failures[:, i] == 0, np.isnan(ts[:, i]))
            failures[:, i][cond] = NEGATIVE_AGE

        if not negative_ratios:
            if meas_232Th_238U:
                failures[:, i] = update_failures(failures[:, i],
                               Th232_U238_sim[:, i] < 0, NEGATIVE_RATIO_SIM)
            else:
                failures[:, i] = update_failures(failures[:, i],
                               Pb208_206_sim[:, i] < 0, NEGATIVE_RATIO_SIM)
            failures[:, i] = update_failures(failures[:, i],
                                 ThU_min_sim[:, i] < 0, NEGATIVE_RATIO_SOL)

        #TOOD: check for reverse discordant 207Pb-corrected solutions?

    # To compute covariances, iteration that fails for one age point should
    # discarded for all?
    ok = np.all(failures == 0, axis=1)
    if np.sum(ok) == 0:
        raise ValueError('all Monte Carlo trials failed')

    # # compile results
    age_95ci = [np.quantile(t[ok], (0.025, 0.975)) for t in ts.T]
    ThU_min_95ci = [np.quantile(x[ok], (0.025, 0.975)) for x in ThU_min_sim.T]
    results = {
        'age_type': age_type,
        'age_1s': [np.nanstd(t[ok]) for t in ts.T],
        'age_95pm': [np.nanmean((t - m, p - t)) for m, p in age_95ci],
        'age_95ci': age_95ci,
        'mean_age': [np.nanmean(t[ok]) for t in ts.T],
        'median_age': [np.nanmedian(t[ok]) for t in ts.T],
        'cov_t': np.cov(np.transpose(ts[ok])),
        'ThU_min_1s': [np.nanstd(x[ok]) for x in ThU_min_sim.T],
        'ThU_min_95pm': [np.nanmean((ThU_min_sim - m, p - ThU_min_sim)) for m, p in ThU_min_95ci],
        'ThU_min_95ci': ThU_min_95ci,
        'mean_ThU_min': [np.nanmean(x[ok]) for x in ThU_min_sim.T],
        'median_ThU_min': [np.nanmedian(x[ok]) for x in ThU_min_sim.T],
        'trials': trials,
        'fails': np.sum(failures != 0, axis=0),
        'not_converged': np.sum(failures == NON_CONVERGENCE, axis=0),
        'negative_ages': np.sum(failures == NEGATIVE_AGE, axis=0),
        'negative_ratios': np.sum(failures == NEGATIVE_RATIO_SIM, axis=0),
        'negative_ratio_soln': np.sum(failures == NEGATIVE_RATIO_SOL, axis=0),
    }

    if any('hist'):
        warnings.warn('Monte Carlo histograms not yet implemented for '
                      'single aliquot iterative ages')

    return results


# ===============================================
# Simulate constants and activity ratios
# ===============================================

def draw_ar(A, sA, trials, failures=None, positive_only=False,
            series='238U'):
    """  
    Draw random activity ratio values.

    """
    assert series in ('238U', '235U')

    if series == '238U':
        A48 = cfg.rng.normal(A[0], sA[0], trials)
        A08 = cfg.rng.normal(A[1], sA[1], trials)
        A68 = cfg.rng.normal(A[2], sA[2], trials)
        if failures is not None and positive_only:
            for x in (A48, A08, A68):
                failures = update_failures(failures, x < 0., NEGATIVE_RATIO_SIM)
        pA = np.array((A48, A08, A68))

    else:
        pA = cfg.rng.normal(A, sA, trials)
        if positive_only:
            failures = update_failures(failures, pA < 0., NEGATIVE_RATIO_SIM)

    return pA, failures


def draw_decay_const(trials, failures, dc_errors=False, series='238U', cor=False):
    """  
    Draw random decay constant values.

    """
    assert series in ('238U', '235U')

    if series == '238U':
        # note: effects of error correlations are probably always negligible ?
        if dc_errors:
            if cor:
                cov_84 = cfg.cor_238_234 * (cfg.s238 * cfg.s234)
                cov_80 = cfg.cor_238_230 * (cfg.s238 * cfg.s230)
                cov_40 = cfg.cor_234_230 * (cfg.s234 * cfg.s230)

                V = np.array([[cfg.s238 ** 2, cov_84, cov_80, 0.],
                              [cov_84, cfg.s234 ** 2, cov_40, 0.],
                              [cov_80, cov_40, cfg.s230 ** 2, 0.],
                              [0., 0., 0., cfg.s226 ** 2]])

                Lam = cfg.rng.multivariate_normal(
                    (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226), V, trials
                )

            else:
                Lam = cfg.rng.normal((cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226),
                                     (cfg.s238, cfg.s234, cfg.s230, cfg.s226),
                                     (trials, 4))
            Lam = np.transpose(Lam)
        else:
            Lam = np.array((cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226))

    else:
        if dc_errors:
            lam235 = cfg.rng.normal(cfg.lam235, cfg.s235, trials)
            lam231 = cfg.rng.normal(cfg.lam231, cfg.s231, trials)
            Lam = np.array((lam235, lam231))
        else:
            Lam = np.array((cfg.lam235, cfg.lam231))

    return Lam, failures


# ===============================================
# Disequilibrium plots
# ===============================================

def diseq_intercept_plot(ts, fit, A8, A15, Lam238, Lam235, coef238, coef235, U, failures,
                         init=(True, True), dp=None, dc_errors=False, u_errors=False,
                         diagram='tw', xlim=(None, None), ylim=(None, None),
                         intercept_points=True, intercept_ellipse=False):
    """
    Plot simulated disequilibrium concordia intercept points.
    """
    fig, ax = plt.subplots(**cfg.fig_kw, subplot_kw=cfg.subplot_kw)
    ax = fig.axes[0]
    ok = failures == 0

    # Filter out failed iteration values:
    if dc_errors:
        Lam238 = np.transpose(np.transpose(Lam238)[ok])
        Lam235 = np.transpose(np.transpose(Lam235)[ok])
        coef238 = np.transpose(np.transpose(coef238)[ok])
        coef235 = np.transpose(np.transpose(coef235)[ok])
    if u_errors:
        U = U[ok]

    A8 = np.transpose(np.transpose(A8)[ok])
    A15 = A15[ok]

    x = 1. / ludwig.f(ts[ok], A8, Lam238, coef238, init=init)
    y = ludwig.g(ts[ok], A15, Lam235, coef235) * x / U
    intercept_plot_axis_limits(ax, x, y, diagram=diagram)

    if intercept_points:
        ax.plot(x, y, **cfg.conc_intercept_markers_kw, label='intercept markers')

    if intercept_ellipse:
        cov = np.cov(x, y)
        # e = plotting.confidence_ellipse2(ax, np.nanmean(x), np.nanmean(y), cov,
        #              **cfg.conc_intercept_ellipse_kw, label='intercept ellipse')
        sx = np.nanstd(x)
        sy = np.nanstd(y)
        r_xy = cov[0, 1] / (sx * sy)
        e = plotting.confidence_ellipse(ax, np.nanmean(x), sx, np.nanmean(y), sy,
                                        r_xy=r_xy, ellipse_kw=cfg.conc_intercept_ellipse_kw)  # add label
        ax.add_patch(e)

    # Plot data ellipses if dp given:
    if dp is not None:
        plotting.plot_dp(ax, *dp, reset_axis_limits=False)

    # Label axes and apply plot settings etc.
    plotting.apply_plot_settings(fig, plot_type='intercept', diagram=diagram,
                                 xlim=xlim, ylim=ylim)
    # Plot regression line and envelope
    plotting.rline(ax, fit['theta'])
    plotting.renv(ax, fit)

    return fig


# ===============================================
# Input checks
# ===============================================

def resolvable_diseq(a, sa, which='a234_238', alpha=0.05, sec_eq=True):
    """
    Check if measured [234U/238] and/or [230Th/238U] value is analytically
    resolvable from equilibrium.

    """
    if sec_eq:
        eq = 1.0
        seq = 0.0
    else:
        if which == 'a234_238':
            eq = cfg.lam234 / (cfg.lam234 / cfg.lam238)
            seq = 0.
        else:
            # TODO: double check this
            eq = cfg.lam230 * cfg.lam234 / (
                    (cfg.lam234 - cfg.lam238) * (cfg.lam230 - cfg.lam238))
            seq = 0.

    p = stats.two_sample_p(a, sa, eq, seq)
    resolvable = p < alpha

    return resolvable, p
