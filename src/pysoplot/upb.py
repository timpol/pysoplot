"""
Functions and routines for U-Pb geochronology.

Notes
-----
See :mod:`pysoplot.dqpb` for equivalent functions that account for
disequilibrium in the uranium-series decay chains.

"""

"""
References
----------
.. [Horstwood2016]
    Horstwood, M.S.A., Košler, J., Gehrels, G., Jackson, S.E., McLean, N.M., Paton,
    C., Pearson, N.J., Sircombe, K., Sylvester, P., Vermeesch, P., Bowring, J.F.,
    Condon, D.J., Schoene, B., 2016. Community-derived standards for LA-ICP-MS
    U-(Th-)Pb geochronology - uncertainty propagation, age interpretation and
    data reporting. Geostandards and Geoanalytical Research 40, 311–332.
    https://doi.org/10.1111/j.1751-908X.2016.00379.x
.. [Ludwig1980]
    Ludwig, K.R., 1980. Calculation of uncertainties of U-Pb isotope data.
    Earth and Planetary Science Letters 212–202.
.. [Ludwig2000]
    Ludwig, K.R., 2000. Decay constant errors in U–Pb concordia-intercept ages.
    Chemical Geology 166, 315–318. https://doi.org/10.1016/S0009-2541(99)00219-3
.. [Powell2020]
    Powell, R., Green, E.C.R., Marillo Sialer, E., Woodhead, J., 2020. Robust
    isochron calculation. Geochronology 2, 325–342.
    https://doi.org/10.5194/gchron-2-325-2020
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton

from . import isochron
from . import cfg
from . import transform
from . import mc as mc
from . import plotting
from . import exceptions

from .concordia import conc_xy, conc_slope, conc_age_x, plot_concordia


diagram_names = {
    'tw': 'Tera-Wasserburg',
    'wc': 'Wetheril Concordia'
}


def concint_age(fit, method='L1980', diagram='tw', dc_errors=False):
    """
    Compute concordia-intercept age and uncertainty using various algorithms.

    Parameters
    ----------
    fit : :obj:`dict`
        Regression fit parameters.
    method : {'Powell', 'L1980', 'L2000'}, optional
        Algorithm used to compute age and age uncertainties.

    Returns
    -------
    results : dict
        Age results.

    Notes
    -----
        - The 'Powell' method focuses on lower intercept age only (< ~1 Ga),
          but works with all fit types. Age uncertainties are symmetric.
        - The 'L1980' method only works with classical fits. Age uncertainties
          are asymmetric.
        - The 'L2000' method works with all fit types and optionally accounts \
          for decay constant errors, but requires 2 intercept age solutions.

    """
    assert method in ('Powell', 'L1980', 'L2000')
    assert diagram in ('tw', 'wc')

    if fit['type'] == 'robust' and method == 'L1980':
        raise ValueError('cannot implement L1980 for data fitted with a '
                         'robust regression model')
    if dc_errors and method != 'L2000':
        raise ValueError('decay constant errors can only be included with '
                         'method L2000')

    # initialise vars for storing optional results:
    t = None
    t_95ci = None
    t2 = None
    t2_95pm = None
    t2_95ci = None
    cor_t12 = None

    if method == 'Powell':
        # Only compute lower intercept age
        t, st = concint_powell(fit['theta'], fit['covtheta'], uncert=True)
        t_95pm = fit['t_crit'] * st

    else:
        try:    # lower intercept
            t = concint_ludwig(fit['theta'], diagram=diagram, t0=-100.,
                               maxiter=40)
        except:
            warnings.warn('lower intercept age did not converge after max '
                          'number of iterations')
        try:    # upper intercept
            t2 = concint_ludwig(fit['theta'], diagram=diagram, t0=5000.,
                                maxiter=40)
        except:
            warnings.warn('upper intercept age did not converge after max '
                          'number of iterations')

        # check intercept ages
        if t is None:
            if t2 is not None:
                t = t2
                t2 = None
            else:
                raise RuntimeError('no intercept ages found')

        # compute age uncertainties
        if method == 'L1980':
            # lower intercept age:
            t_95ci = concint_uncert_ludwig1980(t, fit['theta'], fit['theta_95ci'],
                    fit['xbar'], diagram=diagram)
            t_95pm = np.mean([t_95ci[1] - t, t - t_95ci[0]])
            # upper intercept age:
            if t2 is not None:
                t2_95ci = concint_uncert_ludwig1980(t2, fit['theta'], fit['theta_95ci'],
                        fit['xbar'], diagram=diagram)
                t2_95pm = np.mean([t2_95ci[1] - t2, t2 - t2_95ci[0]])

        else:
            # L2000:
            assert diagram in ('tw', 'wc')
            if None in (t, t2):
                raise ValueError('age uncertainties could not be computed using '
                            'Ludwig2000 method because < 2 intercept age '
                            'solutions found')
            if diagram == 'tw':     # transform fit to wc
                theta, covtheta = transform.transform_fit(fit, transform_to='wc')
            else:
                theta, covtheta = fit['theta'], fit['covtheta']
            t_95pm, t2_95pm, cov_t12 = concint_uncert_ludwig2000(t, t2, theta,
                         covtheta, fit['t_crit'], dc_errors=dc_errors)
            # correlation between t1 and t2:
            cor_t12 = cov_t12 / (t_95pm * t2_95pm)

    results = {
        'age_type': 'concordia-intercept',
        'diagram': diagram_names[diagram],
        'decay_const_errors': dc_errors,
        'age': t,
        'age_95ci': t_95ci,
        'age_95pm': t_95pm,
        'upper_age': t2,
        'upper_age_95ci': t2_95ci,
        'upper_age_95pm': t2_95pm,
        'cor': cor_t12,
    }

    return results


def isochron_age(fit, age_type='iso-206Pb', dc_errors=False, norm_isotope='204Pb'):
    """
    Classical U-Pb isochron age and uncertainty.

    Parameters
    ----------
    fit : :obj:`dict`
        Regression fit parameters.
    age_type : {'iso-206Pb', 'iso-207Pb'}, optional
        Isochron type.

    Returns
    -------
    results : dict
        Age results.

    """
    assert age_type in ('iso-206Pb', 'iso-207Pb')
    b_95pm = fit['theta_95ci'][1]
    t, t_95pm = isochron.age(fit['theta'][1], sb=b_95pm, age_type=age_type,
                             dc_errors=dc_errors)
    results = {
        'age_type': f'{age_type} isochron age',
        'age': t,
        'age_95pm': t_95pm,
    }
    return results


def pbu_ages(dp, age_type='206Pb*', cov=False, alpha=None, wav=False):
    """
    Pb/U ages and uncertainty.
    !!! Not yet fully coded !!!
    """
    assert age_type in ('206Pb*', '207Pb*', 'cor207Pb')
    st = None

    if age_type in ('206Pb*', '207Pb*'):
        x, sx = dp
        n = x.shape[0]
        t, st = pbu_age(x, sx, age_type=age_type)
    else:
        # use for initial guess at diseq. mod207 ages only!
        assert alpha is not None
        x, sx, y, sy, r_xy = dp
        n = x.shape[0]
        a = alpha
        b = (y - a) / x
        t = np.empty(x.size)
        for i in range(n):
            t[i] = concint_powell((a, b[i]), uncert=False, maxiter=30, diagram='tw')

    # compute age uncertainties
    results = {
        'age_type': age_type,
        'age': t,
        'age_1s': st,
        'n': n,
    }
    if wav:
        raise ValueError('not yet coded')

    return results


#=====================================
# Concordia-intercept age algorithms
#=====================================

def concint_ludwig(theta, diagram='tw', t0=10., maxiter=40):
    """Compute concordia-intercept age using routine in the appendix of Ludwig
    (1980).
    """
    a, b = theta
    t1 = t0
    i = 0

    while i < maxiter:
        i += 1
        t = t1
        M = conc_slope(t, diagram)
        fx, fy = conc_xy(t, diagram)
        B = fy - M * fx
        x = (a - B) / (M - b)
        t1 = conc_age_x(x, diagram)
        if np.isclose(t, t1, rtol=1e-08, atol=1e-08):
            return t1

    raise exceptions.ConvergenceError('concordia-intercept age routine failed '
                      'to converge after maximum number of iterations')


def concint_uncert_ludwig1980(t, theta, theta_95ci, xbar, diagram='wc',
          maxiter=40):
    """
    Compute concordia-intercept age uncertainties using the approach of Ludwig (1980).
    The algorithm can account for asymmetric age uncertainties, but not decay
    constant errors.

    Notes
    -----
    This algorithm requires x-bar to be specified so it is really only suitable
    for use with classical regression lines.

    """
    a, b = theta
    sa, sb = theta_95ci
    t_95ci = []

    for sign in (1, -1):
        t0 = t
        i = 0
        while i < maxiter:
            i += 1
            M = conc_slope(t0, diagram)
            fx, fy = conc_xy(t0, diagram)
            B = fy - M * fx
            D = 2 *((B - a) * (M - b) + xbar * sb ** 2)
            E = (M - b) ** 2 - sb ** 2
            V = (B - a) ** 2 - sa ** 2
            x = (-D + sign * np.sqrt(D ** 2 - 4 * E * V)) / (2 * E)
            t1 = conc_age_x(x, diagram)

            if np.isclose(t1, t0, rtol=1e-08, atol=1e-08):
                t_95ci.append(t1)
                break

            t0 = t1

        if i == maxiter:
            raise exceptions.ConvergenceError('Ludwig concordia-intercept '
                      'age error routine did not converge after maximum number '
                      'of iterations')

    t_95ci.sort()   # just in case
    return t_95ci


def concint_powell(theta, covtheta=None, uncert=False, maxiter=30,
                   diagram='tw'):
    """
    Compute Tera-Wasserburg concordia intercept age and uncertainty focussing
    on lower intercept only using the method of Powell et al. (2020). Converges
    rapidly, but will not work for intercept ages greater than ~100 Ma.
    Does not account for decay constant errors.

    """
    if diagram == 'wc':
        raise ValueError('Wetheril concordia version of Powell algorithm not '
                 'yet implemented')
    if uncert and covtheta is None:
        raise ValueError('covtheta cannot be None if uncert is True')

    lam238, lam235, U = cfg.lam238, cfg.lam235, cfg.U
    a, b = theta
    if b > 0:
        warnings.warn('Powell algorithm mat not converge for positive '
                     'Tera-Wasserburg slope')
    i = 0
    t = x = 1500.

    while i < maxiter:
        i += 1
        tc = t
        t = 1. / lam238 * np.log(1. + 1. / x)
        x = (1. / U * (np.exp(lam235 * t) - 1.) /
             (np.exp(lam238 * t) - 1.) - a) / b

        if np.isclose(t, tc, rtol=1e-09):

            if t < 0:
                warnings.warn(f'negative lower intercept age {t:.3f} Ma - '
                              f'check input data')
            if not uncert:
                return t

            den = b * np.exp(lam238 * t) * lam238 + \
                  (-np.exp(lam235 * t) * lam235 + np.exp(lam235 * t
                    + lam238 * t) * lam235 + np.exp(lam238 * t) * lam238
                   - np.exp(lam235 * t + lam238 * t) * lam238) / U
            jac = np.array([(np.exp(lam238 * t) - 1.) / den * (np.exp(lam238 * t)
                        - 1.), (np.exp(lam238 * t) - 1.) / den])
            st = np.sqrt(jac @ covtheta @ jac.T)

            return t, st

    raise exceptions.ConvergenceError('concordi-intercept age routine did not '
              'converge after maximum number of iterations')


def concint_uncert_ludwig2000(t1, t2, theta, covtheta, t_mult, dc_errors=True):
    """
    Compute lower and upper concordia intercept age and uncertainties in one
    go based on equations in Ludwig (2000). Can account for decay constant
    errors. Assumes age uncertainties are symmetric.

    Equations require data points to be in Wetheril concordia form. A
    transformation should first be performed if data points are in
    Tera-Wasserburg form.

    Notes
    -----
    Ludwig (2000) uses 'm' for regression line slope and 'b' for regression
    line y-int, whereas this function uses 'b' for regression line slope and
    'a' for y-intercept.

    """
    assert t2 > t1
    lam238, lam235, U = cfg.lam238, cfg.lam235, cfg.U
    a, b = theta
    covtheta = t_mult ** 2 * covtheta

    # Define some useful quantities
    delta = np.exp(lam235 * t1) - np.exp(lam235 * t2)
    m1 = (lam238 * np.exp(lam238 * t1) - b * lam235 * np.exp(lam235 * t1)) / delta
    m2 = -(lam238 * np.exp(lam238 * t2)
           - b * lam235 * np.exp(lam235 * t2)) / delta
    b1 = -(np.exp(lam235 * t2) - 1.) * m1
    b2 = -(delta + (np.exp(lam235 * t2) - 1.)) * m2

    # Need to solve Ax = B (where from Ludwig's notation Omega -> A and
    # Y -> B):

    # initialise A and B as empty arrays
    A = np.empty((3, 3))
    B = np.empty((3, 1))

    # Eq. (13)
    A[0, 0] = m1 ** 2; A[0, 1] = m2 ** 2; A[0, 2] = 2 * m1 * m2
    A[1, 0] = b1 ** 2; A[1, 1] = b2 ** 2; A[1, 2] = 2 * b1 * b2
    A[2, 0] = m1 * b1; A[2, 1] = m2 * b2; A[2, 2] = m1 * b2 + m2 * b1

    # Eq (14) - regression line error components only
    B[0, 0] = covtheta[1, 1]
    B[1, 0] = covtheta[0, 0]
    B[2, 0] = covtheta[0, 1]

    # If dc_errors, add decay constant error components:
    if dc_errors:
        s238 = 2. * cfg.s238
        s235 = 2. * cfg.s235
        m3 = b * (t2 * np.exp(lam235 * t2) - t1 * np.exp(lam235 * t1)) / delta
        m4 = (t1 * np.exp(lam238 * t1) - t2 * np.exp(lam238 * t2)) / delta
        b3 = - b * t2 * np.exp(lam235 * t2) - (np.exp(lam235 * t2) - 1.) * m3
        b4 = t2 * np.exp(lam238 * t2) - (np.exp(lam235 * t2) - 1.) * m4

        # Eq. (14)
        B[0, 0] += (m3 * s235) ** 2 + (m4 * s238) ** 2
        B[1, 0] += (b3 * s235) ** 2 + (b4 * s238) ** 2
        B[2, 0] += m3 * b3 * s235 ** 2 + m4 * b4 * s238 ** 2

    # Solve for x, Eq. (12)
    x = np.linalg.inv(A) @ B
    t_95pm, t2_95pm, cov = np.sqrt(x[0][0]), np.sqrt(x[1][0]), x[2][0]
    return t_95pm, t2_95pm, cov


#==================================================================
# Single analysis Pb/U ages
#==================================================================

def pbu_age(x, sx=None, age_type='206Pb*', dc_errors=False):
    """
    Single analysis Pb/U age.

    Notes
    -----
    Uncertainties in a suite of single-analysis Pb/U ages should be
    propagated by quadratic addition after taking the weighted mean (e.g.,
    Horstwood, 2016).

    """
    assert age_type in ('206Pb*', '207Pb*')
    dc = cfg.lam238 if age_type == '206Pb*' else cfg.lam235
    sdc = cfg.s238 if age_type == '206Pb*' else cfg.s235
    t = 1. / dc * np.log(x + 1.)
    if sx is None:
        return t
    # Propagate age uncertainty...
    dx = (1. / dc) * 1. / (x + 1.)                        # dt/dx
    ddc = t / dc if dc_errors else 0.                     # dt/ddc
    st = np.sqrt((dx * sx) ** 2 + (ddc * sdc) ** 2)
    return t, st


#=================================
# Age minimisation functions
#=================================

def concint_age_min(diagram='tw'):
    """
    Concordia-intercept age minimisation functions for Monte Carlo
    simulation.
    """
    assert diagram in ('tw', 'wc')

    if diagram == 'tw':
        def fmin(t, a, b, lam238, lam235, U):
            return b + a * (np.exp(lam238 * t) - 1.) \
                   - (np.exp(lam235 * t) - 1.) / U
        def dfmin(t, a, b, lam238, lam235, U):
            return a * lam238 * np.exp(lam238 * t) \
                    - lam235 / U * np.exp(lam235 * t)

    else:
        raise ValueError('not yet implemented')

    return fmin, dfmin


#==================================================================
# Monte Carlo errors
#==================================================================

def mc_concint(t, fit, trials=50_000, diagram='tw', dc_errors=False,
        U_errors=False, intercept_plot=False, hist=False,
        xlim=(None, None), ylim=(None, None), env=False,
        age_ellipses=False, marker_max=None, marker_ages=(), auto_marker_ages=True,
        remove_overlaps=False, intercept_points=True, intercept_ellipse=False,
        negative_ages=True, age_prefix='Ma'):
    """
    Compute Monte Carlo age uncertainties for equilibrium concordia intercept
    age.

    Parameters
    -----------
    t : float
        Calculated age (Ma).
    fit : dict
        Linear regression fit parameters.
    trials : int
        Number of Monte Carlo trials.

    """
    # TODO: take in conc_opt dict as arg
    assert diagram in ('tw', 'wc')

    if env and not dc_errors:
        warnings.warn('cannot plot equilbrium concordia envelope if dc_errors '
                      'set to False')
        env = False
    if age_ellipses and not dc_errors:
        warnings.warn('cannot plot equilibrium age ellipse markers if dc_errors '
                      'set to False')
        age_ellipses = False

    failures = np.zeros(trials)

    # draw randomised slope-intercept values
    (a, b), failures = mc.draw_theta(fit, trials, failures)

    # draw randomised constants
    if dc_errors:
        lam238 = cfg.rng.normal(cfg.lam238, cfg.s238, trials)
        lam235 = cfg.rng.normal(cfg.lam235, cfg.s235, trials)
    else:
        lam238, lam235 = cfg.lam238, cfg.lam235

    U = cfg.rng.normal(cfg.U, cfg.sU, trials) if U_errors else cfg.U

    # solve for age using vectorised Newton
    fmin, dfmin = concint_age_min(diagram='tw')
    ts, c, zd = newton(fmin, np.full(trials, t), fprime=dfmin,
            args=(a, b, lam238, lam235, U), tol=1e-09, rtol=0, disp=False,
                  maxiter=50, full_output=True)

    ts = np.where(c, ts, np.nan)
    failures = mc.check_ages(ts, ~np.isnan(ts), failures, negative_ages=negative_ages)

    ok = (failures == 0)
    if np.sum(ok) == 0:
        raise ValueError('no successful Monte Carlo simulations')

    age_95ci = np.quantile(ts[ok], (0.025, 0.975))
    results = {
        'age_type': 'concordia-intercept',
        'diagram': diagram_names[diagram],
        'age_1s': np.nanstd(ts[ok]),
        'age_95ci': age_95ci,
        'age_95pm': np.nanmean([t - age_95ci[0], age_95ci[1] - t]),
        'mean_age': np.nanmean(ts[ok]),
        'median_age': np.nanmedian(ts[ok]),
        'trials': trials,
        'fails': sum(failures != 0),
        'not_converged': np.sum(failures == mc.NON_CONVERGENCE),
        'negative_ages': np.sum(failures == mc.NEGATIVE_AGE)
    }

    if hist:
        fig = mc.age_hist(ts, diagram, a=a, b=b)
        results['age_hist'] = fig

    # make intercept plot:
    if intercept_plot:
        fig, ax = plt.subplots(**cfg.fig_kw)

        # x, y intercep points
        lam238 = lam238 if not dc_errors else lam238[ok]
        lam238 = lam238 if not dc_errors else lam238[ok]
        U = U if not U_errors else U[ok]
        xs = 1. / (np.exp(lam238 * ts[ok]) - 1.)
        ys = (np.exp(lam235 * ts[ok]) - 1.) * xs / U

        if intercept_points:
            ax.plot(xs, ys, label="intercept markers", **cfg.conc_intercept_markers_kw)
        if intercept_ellipse:
            x = np.nanmean(xs)
            y = np.nanmean(ys)
            sx = np.nanstd(xs)
            sy = np.nanstd(ys)
            r_xy = np.corrcoef(xs, ys)[0, 1]
            e = plotting.confidence_ellipse(ax, x, sx, y, sy, r_xy,
                        ellipse_kw=cfg.conc_intercept_ellipse_kw,
                        mpl_label='intercept ellipse')
            ax.add_patch(e)

        mc.intercept_plot_ax_limits(ax, fit['theta'][1], xs, ys, diagram='tw')
        plotting.apply_plot_settings(fig, plot_type='intercept', diagram=diagram,
                                     xlim=xlim, ylim=ylim)

        # add other plot elements
        plotting.plot_rfit(ax, fit)
        plot_concordia(ax, diagram, plot_markers=True, env=env, 
                       age_ellipses=age_ellipses, marker_max=marker_max,
                       marker_ages=marker_ages, auto_markers=auto_marker_ages,
                       remove_overlaps=remove_overlaps, age_prefix=age_prefix)
        results['fig'] = fig

    return results


def mc_pbu():
    """ Not yet coded.
    """
    pass
