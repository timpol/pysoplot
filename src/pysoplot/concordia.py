"""
Concordia plotting routines for (dis)equilibrium U-Pb datasets.

"""

import warnings
import numpy as np

from scipy import integrate
from scipy import optimize

from . import plotting, ludwig
from . import cfg
from . import misc
from . import useries
from . import stats
from .exceptions import ConvergenceError


exp = np.exp


#=================================
# Concordia plotting routines
#=================================

def plot_concordia(ax, diagram='tw', point_markers=True, age_ellipses=False,
        env=False, marker_max=None, marker_ages=(), auto_markers=True,
        remove_overlaps=True, age_prefix='Ma'):
    """
    Plot disequilibrium U-Pb concordia curve on concordia diagram.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes object to plot concordia curve in.
    diagram : {'tw', 'wc'}
        Concordia diagram type.
    point_markers : bool, optional
        If True, plot concordia regular single point age markers.
    age_ellipses : bool, optional
        If True plot concordia age ellipse markers that represent effects
        of decay constant uncertainties.
    env : bool, optional
        If True, plot concordia uncertainty envelope showing effects of
        decay constant uncertainties on trajectory of concordia curve.
    marker_max : float, optional
        User specified age marker max (Ma).
    marker_ages : array-like, optional
        List of user defined age marker locations (in same units as age_prefix).
    auto_markers : bool, optional
        If True, this function will attempt to find the most suitable
        concordia age marker locations.
    remove_overlaps : bool, optional
        If True, this function will remove first overlapping concordia age
        marker and all older labels.

    Raises
    -------
    UserWarning: if concordia lies entirely outside the axis limits.

    """
    assert diagram in ('tw', 'wc'), "diagram must be 'wc' (Wetheril) or 'tw' (Tera-Wasserburg)"
    assert ax.get_xlim()[1] > ax.get_xlim()[0], 'x-axis limits must be in ascending order'
    assert ax.get_ylim()[1] > ax.get_ylim()[0], 'y-axis limits must be in ascending order'

    ax.autoscale(enable=False, axis='both')     # freeze axis limits

    # ...
    tbounds = cfg.conc_age_bounds
    if auto_markers and (marker_max is not None):
        if not marker_max > tbounds[0]:
            raise ValueError('marker_max value must be greater than the lower '
                             'conc_age_bound value')

    code, tlim = eq_age_limits(ax, diagram=diagram, tlim=(0.001, 4600.))
    if code == 1:
        warnings.warn('concordia appears to lie entirely outside axis limits')
        return

    # get equally spaced points
    ct, cx, cy = eq_equi_points(*tlim, ax.get_xlim(), ax.get_ylim(),
                             diagram, ngp=500_000, n=100)
    # plot line
    ax.plot(cx, cy, **cfg.conc_line_kw, label='concordia line')
    # plot envelope
    if env:
        plot_envelope(ax, diagram, xc=cx)
    if point_markers or age_ellipses:
        if auto_markers and (marker_max is not None) and (marker_max < tlim[1]):
            tlim[1] = marker_max
            if marker_max < tlim[0]:
                warnings.warn('marker_max age is less than auto lower limit '
                              '- no markers to plot')
                return
        markers_dict = generate_age_markers(ax, *tlim, tbounds, diagram,
                auto=auto_markers, marker_ages=marker_ages,
                ell=age_ellipses, age_prefix=age_prefix,
                point_markers=point_markers)
        markers_dict = plot_age_markers(ax, markers_dict)

        # label markers
        if cfg.individualised_labels:
            individualised_labels(ax, markers_dict, diagram,
                                  remove_overlaps=remove_overlaps)
        else:
            labels(ax, markers_dict)


def plot_diseq_concordia(ax, A, meas, sA=None, diagram='tw', env=False,
            point_markers=True, age_ellipses=False, marker_max=None,
            marker_ages=(), auto_markers=True, remove_overlaps=True,
            age_prefix='Ma', spaghetti=False):
    """
    Plot disequilibrium U-Pb concordia curve on concordia diagram.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes object to plot concordia curve in.
    A : array-like
        one-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U]
    meas : array-like
        two-element list of boolean values, the first is True if [234U/238U]
        is a present-day value and False if an initial value, the second is True
        if [230Th/238U] is an present-day value and False if an initial value
    sA : array-like, optional
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A
    diagram : {'tw', 'wc'}
        Concordia diagram type.
    point_markers : bool, optional
        If True, plot concordia regular single point age markers.
    age_ellipses : bool, optional
        If True plot concordia age ellipse markers that represent effects
        of decay constant uncertainties.
    env : bool, optional
        If True, plot concordia uncertainty envelope showing effects of
        decay constant uncertainties on trajectory of concordia curve.
    marker_max : float, optional
        User specified age marker max (Ma).
    marker_ages : array-like, optional
        List of user defined age marker locations (in same units as age_prefix).
    auto_markers : bool, optional
        If True, this function will attempt to find the most suitable
        concordia age marker locations.
    remove_overlaps : bool, optional
        If True, this function will remove first overlapping concordia age
        marker and all older labels.
    spaghetti : bool
        Plot each simulated line using arbitrary colours (no longer used).

    Raises
    -------
    UserWarning: if concordia lies entirely outside the axis limits.

    """
    assert diagram == 'tw', 'Wetheril concordia not yet implemented'
    assert ax.get_xlim()[1] > ax.get_xlim()[0], 'x-axis limits must be in ascending order'
    assert ax.get_ylim()[1] > ax.get_ylim()[0], 'y-axis limits must be in ascending order'
    if env or age_ellipses:
        assert sA is not None, 'sA must be given to plot concordia envelope or age ellipses'

    ax.autoscale(enable=False, axis='both')     # freeze axis limits

    # Get hard age limits.
    if meas[1]:
        tbounds = cfg.diseq_conc_age_bounds[2]
    elif meas[0]:
        tbounds = cfg.diseq_conc_age_bounds[1]
    else:
        tbounds = cfg.diseq_conc_age_bounds[0]

    if auto_markers and (marker_max is not None):
        if not marker_max > tbounds[0]:
            raise ValueError('marker_max value must be greater than the lower '
                             'conc_age_bound value')

    code, tlim, tbounds = diseq_age_limits(ax, A, meas, diagram='tw', tbounds=tbounds,
                                  max_age=marker_max)
    if code == 1:
        warnings.warn('concordia appears to lie entirely outside axis limits')
        return

    # get equally spaced points
    ct, cx, cy = diseq_equi_points(*tlim, ax.get_xlim(), ax.get_ylim(), A, meas,
                        diagram, ngp=500_000, n=100)
    # plot line
    ax.plot(cx, cy, **cfg.conc_line_kw, label='concordia line')

    # Check if activity ratios are resolvable from equilibrium. Do not plot
    # envelope or ellipses if not.
    # Check measured activity ratios.
    if env or age_ellipses:
        if meas[0]:
            p = stats.two_sample_p(A[0], sA[0], cfg.a234_238_eq, cfg.a234_238_eq_1s)
            if p > 0.05:
                warnings.warn(f'cannot plot concordia envelope or age ellipses if '
                        f'[234U/238U] is not sufficiently resolved from equilibrium')
                env, age_ellipses = False, False
        if meas[1]:
            p = stats.two_sample_p(A[0], sA[0], cfg.a234_238_eq, cfg.a234_238_eq_1s)
            if p > 0.05:
                warnings.warn(f'cannot plot concordia envelope or age ellipses if '
                        f'[230Th/238U] is not sufficiently resolved from equilibrium')
                env, age_ellipses = False, False

    # plot envelope
    pA = None
    if env:
        pA = plot_diseq_envelope(ax, ct, cx, cy, *tlim, tbounds, A, sA, meas,
                            diagram='tw', trials=10_000, spaghetti=False)
    if point_markers or age_ellipses:
        if auto_markers and (marker_max is not None) and (marker_max < tlim[1]):
            tlim[1] = marker_max
            if marker_max < tlim[0]:
                warnings.warn('marker_max age is less than auto lower limit '
                              '- no markers to plot')
                return
            if age_ellipses and (sA is None or all([x == 0 for x in sA])):
                warnings.warn('cannot plot disequilibrium concordia age ellipses if no '
                          'uncertainity assigned to activity ratios')
                if not point_markers:
                    return
                age_ellipses = False
        markers_dict = generate_age_markers(ax, *tlim, tbounds, diagram,
                        A=A, sA=sA, meas=meas, auto=auto_markers,
                        marker_ages=marker_ages, eq=False,
                        ell=age_ellipses, age_prefix=age_prefix,
                        point_markers=point_markers)
        markers_dict = plot_age_markers(ax, markers_dict, pA=pA)

        # label markers
        if cfg.individualised_labels:
            individualised_labels(ax, markers_dict, diagram,
                                  remove_overlaps=remove_overlaps)
        else:
            labels(ax, markers_dict)


#==============================================================================
# Eq concordia functions
#==============================================================================

def eq_xy(t, diagram):
    """
    Return x, y for given t along (equilibrium) concordia curve.
    """
    assert diagram in ('tw', 'wc')
    if diagram == 'tw':
        x = 1. / (exp(cfg.lam238 * t) - 1.)
        y = (1. / cfg.U) * (exp(cfg.lam235 * t) - 1.) / (exp(cfg.lam238 * t) - 1.)
    elif diagram == 'wc':
        y = exp(cfg.lam238 * t) - 1.
        x = exp(cfg.lam235 * t) - 1.
    return x, y


def eq_age_x(x, diagram):
    """
    Age of point on concordia at given x value.
    """
    assert diagram in ('tw', 'wc')
    if diagram == 'wc':
        t = 1. / cfg.lam235 * np.log(1. + x)
    else:
        t = 1. / cfg.lam238 * np.log(1. + 1. / x)
    return t


def eq_slope(t, diagram):
    """
    Compute tangent to concordia at given t. I.e. dy/dx for given t.
    """
    assert diagram in ('tw', 'wc')
    lam238, lam235, U = cfg.lam238, cfg.lam235, cfg.U
    if diagram == 'wc':
        return lam238 / lam235 * exp((lam238 - lam235) * t)
    elif diagram == 'tw':
        den = (exp(lam238 * t) - 1.)
        dx = -lam238 * exp(lam238 * t) / den ** 2
        dy = lam235 * exp(lam235 * t) / (U * den) - (lam238 *
                exp(lam238 * t) * (exp(lam235 * t) - 1.)) / (U * den ** 2)
        return dy / dx


def eq_age_ellipse(t, diagram):
    """
    Age ellipse params for displaying effects of decay constant
    errors on equilibrium concordia age markers. Requires computing uncertainty
    in x and y for a given t value using first-order error propagation.
    """
    assert diagram in ('tw', 'wc')
    lam238, lam235, s238, s235 = cfg.lam238, cfg.lam235, cfg.s238, cfg.s235
    if diagram == 'wc':
        sx = t * exp(lam235 * t) * s235
        sy = t * exp(lam238 * t) * s238
        cov_xy = 0. * t                         # float or array of zeros
    else:
        x, y = eq_xy(t, diagram)
        sx = - x ** 2 * t * exp(lam238 * t) * s238
        sy = x * t * np.sqrt((exp(lam235 * t) * s235 / cfg.U) ** 2 + (
                (y * exp(lam238 * t) * s238) ** 2))
        cov_xy = x ** 3 * y * (t * exp(lam238 * t) * s238) ** 2
    r_xy = cov_xy / (sx * sy)
    return sx, sy, r_xy


def eq_envelope(x, diagram):
    """
    Uncertainty in y for a given x value along the concordia to
    display the effects of decay constant errors on trajectory of the concordia
    curve. Requires computing uncertainty in y for a given x value using
    first-order error propagation.
    """
    assert diagram in ('tw', 'wc')
    lam238, lam235, s238, s235 = cfg.lam238, cfg.lam235, cfg.s238, cfg.s235
    t = eq_age_x(x, diagram)
    if diagram == 'wc':
        sy =  t * exp(lam238 * t) * np.sqrt(s238 ** 2
                + (lam238 / lam235 * s235) ** 2)
    else:
        sy = x * t * exp(lam235 * t) / cfg.U * np.sqrt(s235 ** 2
                + (lam235 / lam238 * s238) ** 2)
    return sy


def eq_velocity(t, xlim, ylim, diagram):
    """
    Estimate dr/dt, which is "eq_velocity" along an equilibrium concordia curve in
    x-y space. Uses axis coordinates to circumvent scaling issues.
    """
    # TODO: this could be calculated analytically for eq case ??
    h = 1e-08 * t
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    x2, y2 = eq_xy(t + h, diagram)
    x1, y1 = eq_xy(t - h, diagram)
    v = np.sqrt(((x2 - x1) / xspan) ** 2 + ((y2 - y1) / yspan) ** 2) / (2. * h)
    return v


def eq_equi_points(t1, t2, xlim, ylim, diagram, ngp=500_000, n=500):
    """
    Uses numerical method to obtain age points that are approximately evenly
    spaced in x, y along equilibrium U-Pb concordia between age limits t1 and t2.
    """
    # TODO: could this be done analytically too?
    assert t2 > t1
    t = np.linspace(t1, t2, ngp)
    # suppress numpy warnings
    with np.errstate(all='ignore'):
        dr = eq_velocity(t, xlim, ylim, diagram)
    # Cumulative integrated area under eq_velocity curve (aka cumulative
    # "distance") at each t_j from t1 to t2:
    cum_r = integrate.cumtrapz(dr, t, initial=0)
    # Divide cumulative area under eq_velocity curve into equal portions.
    rj = np.arange(n + 1) * cum_r[-1] / n
    # Find t_j value at each r_j:
    idx = np.searchsorted(cum_r, rj, side="left")
    idx[-1] = ngp - 1 if idx[-1] >= ngp else idx[-1]
    t = t[idx]
    x, y = eq_xy(t, diagram)
    return t, x, y


def eq_age_limits(ax, diagram='tw', tlim=(0.001, 4600.)):
    """

    """
    assert diagram in ('tw', 'wc'), "diagram must be 'wc' (Wetheril) or 'tw' (Tera-Wasserburg)"

    ax_xmin, ax_xmax = ax.get_xlim()
    ax_ymin, ax_ymax = ax.get_ylim()

    # --- testing ----
    # tt = np.linspace(1e-03, 4.6e3, 100_000)
    # xx, yy = eq_xy(tt, diagram)
    # ax.plot(xx, yy, 'ro')
    # ax.get_figure().show()
    # -----------

    # cx_min, cy_max = eq_xy(tlim[1], diagram)
    # cx_max, cy_min = eq_xy(tlim[0], diagram)
    cx_min, cy_min = eq_xy(tlim[0], diagram)
    cx_max, cy_max = eq_xy(tlim[1], diagram)
    if diagram == 'tw':
        cx_min, cx_max = cx_max, cx_min

    # rest ax limits if they extend past hard concordia limits:
    if ax_xmin < cx_min:
        ax_xmin = cx_min
    if ax_xmax > cx_max:
        ax_xmax = cx_max
    if ax_ymin < cy_min:
        ax_ymin = cy_min
    if ax_ymax > cy_max:
        ax_ymax = cy_max

    t_min, t_max = None, None

    if diagram == 'tw':

        t_xmax = 1. / cfg.lam238 * np.log(1. / ax_xmax + 1.)
        t_xmin = 1. / cfg.lam238 * np.log(1. / ax_xmin + 1.)

        # check bottom left corner of ax is not above / right of concordia curve
        if eq_xy(t_xmin, diagram)[1] < ax_ymin:
            return 1, [t_min, t_max]
        # check top right corner of ax is not below / left of concordia curve
        if eq_xy(t_xmax, diagram)[1] > ax_ymax:
            return 1, [t_min, t_max]

        # check curve goes through ax_xmax b/w ylim
        if ax_ymin < eq_xy(t_xmax, diagram)[1] < ax_ymax:
            t_min = t_xmax
        # check if curve intersects ymin b/w xlim
        else:
            r = optimize.brentq(lambda t: eq_xy(t, diagram)[1] - ax_ymin,
                                4.6e3, 1e-3, full_output=True, disp=False)
            if r[1].converged:
                t_min = r[0]
            else:
                return -1, [t_min, t_max]

        # get max t value
        if ax_ymin < eq_xy(t_xmin, diagram)[1] < ax_ymax:
            t_max = t_xmin
        else:
            r = optimize.brentq(lambda t: eq_xy(t, diagram)[1] - ax_ymax,
                                4.6e3, 1e-3, full_output=True, disp=False)
            if r[1].converged:
                t_max = r[0]
            else:
                return -1, [t_min, t_max]

    elif diagram == 'wc':

        t_ymax = 1. / cfg.lam238 * np.log(ax_ymax + 1.)
        t_ymin = 1. / cfg.lam238 * np.log(ax_ymin + 1.)

        # check bottom left corner of axes is not above / left of concordia curve
        if eq_xy(t_ymin, diagram)[0] > ax_xmax:
            return 1, [t_min, t_max]
        # check top left corner of axes is not below / left of concordia curve
        if eq_xy(t_ymax, diagram)[0] < ax_xmin:
            return 1, [t_min, t_max]

        # get min t value
        # check curve goes through ax_ymin b/w xlim
        if ax_xmin < eq_xy(t_ymin, diagram)[0] < ax_xmax:
            t_min = t_ymin
        # check if curve intersects ymin b/w xlim
        else:
            r = optimize.brentq(lambda t: eq_xy(t, diagram)[0] - ax_xmin,
                                4.6e3, 1e-3, full_output=True, disp=False)
            if r[1].converged:
                t_min = r[0]
            else:
                return -1, [t_min, t_max]

        # get max t value
        if ax_xmin < eq_xy(t_ymax, diagram)[1] < ax_xmax:
            t_max = t_ymax
        else:
            r = optimize.brentq(lambda t: eq_xy(t, diagram)[0] - ax_xmax,
                                4.6e3, 1e-3, full_output=True, disp=False)
            if r[1].converged:
                t_max = r[0]
            else:
                return -1, [t_min, t_max]

    return 0, [t_min, t_max]


#==============================================================
# Diseq concordia functions
#==============================================================

def diseq_xy(t, A, meas, diagram):
    """Return x, y for given t along disequilibrium concordia curve.
    """
    assert diagram in ('tw', 'wc')
    if diagram == 'tw':
        x = 1. / ludwig.f(t, A[:-1], meas=meas)
        y = ludwig.g(t, A[-1]) * x / cfg.U
    elif diagram == 'wc':
        y = ludwig.f(t, A[:-1], meas=meas)
        x = ludwig.g(t, A[-1])
    return x, y


def diseq_dxdt(t, A, meas, diagram):
    """
    Return x given t along disequilibrium concordia curve.
    Used e.g. to compute dx/dt
    """
    assert diagram == 'tw'
    def conc_x(t):
        x, _ = diseq_xy(t, A, meas, diagram)
        return x
    h = abs(t) * np.sqrt(np.finfo(float).eps)
    dxdt = misc.cdiff(t, conc_x, h)
    return dxdt


def diseq_slope(t, A, meas, diagram):
    """
    Compute tangent to concordia at given t. I.e. dy/dx for given t.
    """
    h = np.sqrt(np.finfo(float).eps) * t
    x2, y2 = diseq_xy(t + h, A, meas, diagram)
    x1, y1 = diseq_xy(t - h, A, meas, diagram)
    return (y2 - y1) / (x2 - x1)


def diseq_age_ellipse(t, A, sA, meas, trials=1_000, pA=None, diagram='tw'):
    """
    Plot disequilibrium concordia marker as an "age ellipse" which provides
    a visual representation of uncertainty in x-y for given t value arising from
    uncertainties in activity ratio values.
    """
    assert diagram == 'tw', 'can only plot ellipses for Tera-Wasserburg diagram ' \
                            'at present'

    # do monte carlo simulation
    #TODO: should take in activity ratios?
    if pA is None:
        pA = cfg.rng.normal(A, sA, (trials, 4))
    else:
        assert pA.shape[1] == 4

    flags = np.zeros(trials)

    if meas[0]:
        a234_238_i = useries.aratio48i(t, pA[:, 0])
        flags = np.where(a234_238_i < 0, -1, 0)
    if meas[1]:
        a230_238_i = useries.aratio08i(t, pA[:, 0], pA[:, 1], init= not meas[0])
        flags = np.where((a230_238_i < 0) & (flags == 0), -2, 0)

    if sum(flags != 0) > (0.99 * trials):
        msg = f'{sum(flags != 0)}/{trials} negative activity ratio soln. ' \
              f'values in age ellipse t = {t:.3f} Ma'
        warnings.warn(msg)

    x, y = diseq_xy(t, A, meas, 'tw')    # centre point
    xpts, ypts = diseq_xy(t, np.transpose(pA), meas, 'tw')

    V_xy = np.cov(np.array([xpts, ypts]))
    sx, sy = np.sqrt(np.diag(V_xy))
    r_xy = V_xy[0, 1] / (sx * sy)

    if np.isclose(abs(r_xy), 1.0):
        #TODO: review this
        r_xy = (1. - 1e-08) * np.sign(r_xy)

    # reset r_xy for special cases:
    if sA[-1] == 0:
        if any(np.asarray(sA[:2]) != 0):
            r_xy = 1. - 1e-08
    else:
        if all((x == 0 for x in sA[:2])):
            sx = 0.
            r_xy = 0.

    return x, y, sx, sy, r_xy


def diseq_equi_points(t1, t2, xlim, ylim, A, meas, diagram, ngp=500_000, n=500):
    """
    Return ages that give equally spaced x, y points along disequilbrium
    concordia between upper and lower age limits.
    """
    # TODO: this needs further debugging, sometimes returns t of length n + 1,
    #        also occasionally returns duplicate t values.
    assert t2 > t1
    t = np.linspace(t1, t2, ngp)
    # suppress numpy warnings
    with np.errstate(all='ignore'):
        dr = diseq_velocity(t, xlim, ylim, A, meas, diagram)
    # Cumulative integrated area under eq_velocity curve (aka cumulative
    # "distance") at each t_j from t1 to t2:
    cum_r = integrate.cumtrapz(dr, t, initial=0)
    # Divide cumulative area under eq_velocity curve into equal portions.
    rj = np.arange(n + 1) * cum_r[-1] / n
    # Find t_j value at each r_j:
    idx = np.searchsorted(cum_r, rj, side="left")
    idx[-1] = ngp - 1 if idx[-1] >= ngp else idx[-1]
    t = t[idx]
    x, y = diseq_xy(t, A, meas, diagram)
    return t, x, y


def diseq_velocity(t, xlim, ylim, A, meas, diagram):
    """
    Estimate dr/dt, which is "eq_velocity" along a diseq concordia curve in
    x-y space. Uses axis coordinates to circumvent scaling issues.
    """
    h = 1e-08 * t
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    x2, y2 = diseq_xy(t + h, A, meas, diagram)
    x1, y1 = diseq_xy(t - h, A, meas, diagram)
    v = np.sqrt(((x2 - x1) / xspan) ** 2 + ((y2 - y1) / yspan) ** 2) / (2. * h)
    return v


#=================================
# Concordia age bound functions
#=================================

def diseq_age_limits(ax, A, meas, diagram='tw', tbounds=(0.010, 100.),
                     max_age=10.):
    """
    Find the age limits of a disequilibrium concordia curve segment
    that plots within the given axis limits.

    Uses a brute force method to find approximate limits, then refines these
    using Newton's method.

    """
    assert diagram == 'tw', 'Wetheril concordia not yet implemented'

    ax_xmin, ax_xmax = ax.get_xlim()
    ax_ymin, ax_ymax = ax.get_ylim()
    tbounds = np.asarray(tbounds, dtype='double')
    tlim = list(tbounds)

    # --- testing ----
    # tt = np.logspace(np.log(1e-3), np.log(1e3), num=1_000_000, base=np.exp(1))
    # xx, yy = concordia.diseq_xy(tt, A, ~np.asarray(meas), diagram)
    # ax.plot(xx, yy, 'ro')
    # ax.get_figure().show()
    # -----------

    t_min, t_max = None, None

    # Check if tlim[1] exceeds t_max
    if max_age is not None:
        if max_age < tlim[0]:
            raise ValueError('max_age cannot be less than the first element of tlim')
        if tlim[1] > max_age:
            tlim[1] = max_age

    # Check if hard limits are inside plot window.
    xc, yc = diseq_xy(np.asarray(tlim), A, meas, diagram)
    if all((xc > ax_xmin) & (xc < ax_xmax) & (yc > ax_ymin) & (yc < ax_ymax)):
        return 0, tlim, tbounds

    # Simulate log-spaced points and check if any are inside axis bounds.
    tc = np.logspace(np.log10(tlim[0]), np.log10(tlim[1]), num=1_000_000)
    xc, yc = diseq_xy(tc, A, meas, diagram)
    inside = ((ax_xmin < xc) & (xc < ax_xmax)) & ((ax_ymin < yc) & (yc < ax_ymax))

    # Verify that a point has been found, if not return error code.
    if np.sum(inside) < 1:
        return 1, (t_min, t_max), tbounds
    elif np.sum(inside) > 3 and any(meas):
        # If measured 234/238 or 230/238 given, do not allow concordia curve to
        # loop back over itself. This creates plotting difficulties. Also, the loooping
        # part is usually (always?) associated with physically implausible initial activity
        # ratio solutions.

        # check if dx/dt changes sign, if so, there are multiple y for x, and
        # therefore t needs to be truncated to plot envelope
        dxdt = diseq_dxdt(tc[inside], A, meas, diagram)
        ind = np.where(np.diff(np.sign(dxdt)) != 0)[0]
        if ind.shape[0] != 0:
            if len(ind) > 1:    # if multiple dx/dt changes, probably a numerical issue computing deriv.
                warnings.warn(f'multiple dx/dt sign changes in concordia found')
            else:
                tlim[1] = float(tc[inside][ind])
                tbounds[1] = tlim[1]
                warnings.warn(f'concordia truncated at t = {tlim[1]:.3f} because dx/dt changes sign')
            # re-do search for inside points
            tc = np.logspace(np.log10(tlim[0]), np.log10(tlim[1]), num=1_000_000)
            xc, yc = diseq_xy(tc, A, meas, diagram)
            inside = ((ax_xmin < xc) & (xc < ax_xmax)) & ((ax_ymin < yc) & (yc < ax_ymax))

    # Get indices of inside / outside change points for different cases:
    min_inside, max_inside = False, False
    idx = np.where(np.diff(inside) != 0)[0]
    cp = len(idx)
    if cp == 0:
        return 1, [t_min, t_max], tbounds
    elif cp == 1:
        # Either tlim[0] or tlim[1] is inside axis bounds.
        if inside[0]:
            min_inside = True
            t_min = tlim[0]
            t_max = (tc[idx[0]], tc[idx[0] + 1])
        else:
            max_inside = True
            t_min = (tc[idx[0]], tc[idx[0] + 1])
            t_max = tlim[1]
    elif cp == 2:
        if inside[0]:
            # must be more than one segment - only use first!
            t_min = tlim[0]
            t_max = (tc[idx[0]], tc[idx[0] + 1])
            tlim[1] = tc[idx[0] + 1]
        else:
            # Neither tlim[0] nor tlim[1] are in axis bounds.
            t_min = (tc[idx[0]], tc[idx[0] + 1])
            t_max = (tc[idx[1]], tc[idx[1] + 1])
    elif cp == 3 and inside[0]:
        # must be more than one segment - only use first!
        t_min = tlim[0]
        t_max = (tc[idx[0]], tc[idx[0] + 1])
        tlim[1] = tc[idx[0] + 1]
    elif cp in (3, 4):
        # Neither tlim[0] nor tlim[1] are in axis bounds.
        t_min = (tc[idx[0]], tc[idx[0] + 1])
        t_max = (tc[idx[1]], tc[idx[1] + 1])
        tlim[1] = tc[idx[1] + 1]

    elif cp > 4:
        raise RuntimeError('cannot have more than four boundary points')

    if not min_inside:
        try:
            t_min = refine_age_lim(ax.get_xlim(), ax.get_ylim(), *t_min, A,
                                   meas, which='lower')
        except ConvergenceError:
            warnings.warn('lower concordia age limit could not be refined')
            t_min = np.min(t_min)
    if not max_inside:
        try:
            t_max = refine_age_lim(ax.get_xlim(), ax.get_ylim(), *t_max, A,
                                   meas, which='upper')
        except ConvergenceError:
            warnings.warn('upper concordia age limit could not be refined')
            t_max = np.min(t_max)

    assert t_max > t_min, 'lower concordia age limit should be smaller than ' \
                          'upper limit'

    return 0, [t_min, t_max], tbounds


def refine_age_lim(xlim, ylim, t1, t2, A, meas, which='lower'):
    """
    Refine limits using newton
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    # refine lower
    x1, y1 = diseq_xy(t1, A, meas, 'tw')
    x2, y2 = diseq_xy(t2, A, meas, 'tw')
    t0 = np.mean((t1, t2))

    # Lower t limit. t2 inside axis bounds, t1 is outside.
    if x1 > x2:
        # check intersection with xmin
        if which == 'lower':
            fmin, dfmin = min_tax(xmax, meas, diagram='tw')
        else:
            fmin, dfmin = min_tax(xmin, meas, diagram='tw')
        with np.errstate(all='ignore'):
            r = optimize.newton(fmin, t0, dfmin, full_output=True, disp=False,
                                args=([A]))
        if r[1].converged:
            if ymin < diseq_xy(r[0], A, meas, 'tw')[1] < ymax:
                return r[0]

    if ((y2 > y1) and not (ymin < y1 < ymax)) or ((y2 < y1) and (ymin < y1 < ymax)):
        # check intersection with ymin
        fmin, dfmin = min_tay(ymin, meas, diagram='tw')
        with np.errstate(all='ignore'):
            r = optimize.newton(fmin, t0, dfmin, full_output=True, disp=False,
                                args=([A]))
        if r[1].converged:
            if xmin < diseq_xy(r[0], A, meas, 'tw')[0] < xmax:
                return r[0]
    else:
        # check intersection with ymax
        fmin, dfmin = min_tay(ymax, meas, diagram='tw')
        with np.errstate(all='ignore'):
            r = optimize.newton(fmin, t0, dfmin, full_output=True, disp=False,
                                args=([A]))
        if r[1].converged:
            if xmin < diseq_xy(r[0], A, meas, 'tw')[0] < xmax:
                return r[0]

    raise ConvergenceError('could not refine concordia age limits')


def min_tay(y, meas, diagram='tw'):
    """
    Minimisation function to solve concordia age for given y value.
    """
    def fmin(t, A):
        return diseq_xy(t, A, meas, diagram)[1] - y
    def dfmin(t, A):
        #TODO: replace with analytical derivative
        return misc.cdiff(t, fmin, 1e-08 * t, A)
    return fmin, dfmin


def min_tax(x, meas, diagram='tw'):
    """
    Minimisation function to solve concordia age for given x value.
    """
    def fmin(t, A):
        return diseq_xy(t, A, meas, diagram)[0] - x
    def dfmin(t, A):
        #TODO: replace with analytical derivative
        return misc.cdiff(t, fmin, 1e-08 * t, A)
    return fmin, dfmin


#====================
# Concordia markers
#====================

def generate_age_markers(ax, t1, t2, tbounds, diagram, eq=True,
            point_markers=True, ell=False, A=None, sA=None, meas=None,
            marker_ages=(), age_prefix='Ma', auto=True):
    """
    Generate appropriately spaced concordia age markers and label text.

    Parameters
    ----------
    t1 : float
        lower concordia age in plot,
    t2 : float
        upper concordia age
    ell : bool
        plot age ellipses

    """
    assert point_markers or ell, 'one of point_markers or ell must be True'
    assert t2 > t1, 'upper age limit must be greater than lower age limit'
    age_unit = 1. if age_prefix == 'Ma' else 1e-3
    dt = None

    if not auto:  # manual age markers

        if len(marker_ages) == 0:
            raise ValueError('ages cannot be empty if auto set to False')
        for i, x in enumerate(marker_ages):
            try:
                fx = float(x) * age_unit
            except ValueError:
                raise ValueError(f'could not convert marker age {x} to a'
                                 f'number')
            else:
                if not cfg.conc_age_bounds[0] < fx < cfg.conc_age_bounds[1]:
                    msg = f'marker age {x} {age_prefix} outside ' \
                          f'conc_age_bounds value set in config'
                    warnings.warn(msg)

        t_sorted = np.sort(np.array(marker_ages, dtype=np.double))
        t = t_sorted * age_unit
        t_sorted = t_sorted[(t1 < t_sorted) & (t_sorted < t2)]
        n_inside = len(t[np.logical_and(t1 < t, t2 > t)])
        # which markers to label:
        if n_inside > cfg.every_second_threshold:
            add_label = [True if i % 2 == 0 else False for i, t in enumerate(t)]
        else:
            add_label = [True for t in t]

    else:  # find auto marker locations

        max_markers = 8 if ell else 12
        dt = age_marker_spacing(ax, t1, t2, diagram, A=A, meas=meas, eq=eq,
                                max_markers=max_markers)

        # Get marker age points:
        t_start = misc.round_down(np.floor(t1 / dt) * dt, 5)
        t = np.arange(t_start, t2 + dt, dt)

        # Reset to 0 if sufficiently close in order to avoid labelling problems.
        t = [0. if abs(x) < 1e-9 else x for x in t]
        t = np.array([round(x, 10) for x in t])         # round ages to get around f.p. issues

        # If labelling every second, check which label to start with. Preference
        # starting on label with less significant digits, then preference starting
        # on label ending in 1, and finally preference starting on even number.
        n = len(t)

        start_idx = 0
        step = 1
        if n > cfg.every_second_threshold:
            step = 2
            t0 = misc.round_down(float(t[0]), 8)
            t1 = misc.round_down(float(t[1]), 8)
            s0 = str(t0).rstrip("0")  # first marker in sequence
            s1 = str(t1).rstrip("0")  # second marker in sequence

            if s0 == '':
                # if initial t is 0
                start_idx = 1
            elif s0[-1] == '.':
                start_idx = 0
            elif s1[-1] == '.':
                start_idx = 1
            elif len(s0) > len(s1):
                start_idx = 1
            elif float(s0[-1]) % 2 > 0 and float(s1[-1]) % 2 == 0:
                start_idx = 1

        # Add extra markers to ends, in case they are partly displayed in plot
        # window - but this should not affect label starting age.
        # if not ell:
        t = np.arange(t_start - dt, t2 + 2 * dt, dt)
        start_idx = 1 if start_idx == 0 else 0

        t = np.array([round(x, 10) for x in t])         # round ages again
        t = t[(t >= tbounds[0]) & (t <= tbounds[1])]   # double check bounds
        num_t = len(t)                                  # new number of markers

        # list of bools indicating which markers to add a label to:
        if n > cfg.every_second_threshold:
            add_label = [True if (i - start_idx) % step == 0
                         else False for i in range(num_t)]
        else:
            add_label = [True] * len(t)

    markers_dict = {'diagram': diagram,
                   't': t,
                   'dt': dt,
                   'ell': ell,
                   'point_markers': point_markers,
                   'eq': eq,
                   'A': A,
                   'sA': sA,
                   'meas': meas,
                   'add_label': add_label,
                   'age_prefix': age_prefix}

    return markers_dict


def estimate_marker_spacing(tspan):
    """
    Get initial estimate of appropriate concordia marker spacing.
    """
    dt = 10 ** misc.get_exponent(tspan) / 8
    while abs(tspan / dt) > 12:
        dt *= 2
    return misc.round_down(dt, 8)


def age_marker_spacing(ax, t1, t2, diagram, A=None, meas=None, eq=True,
                       max_markers=12):
    """
    Estimate reasonable concordia age marker spacing given upper and lower
    age marker limits.

    """
    tspan = t2 - t1
    t_ratio = t2 / t1
    # First estimate of spacing between markers, dt.
    dt = estimate_marker_spacing(tspan)
    if t_ratio > 5 and t2 > 1:
        dt = estimate_marker_spacing(tspan / 4)
    # Increase spacing if too many markers.
    while abs(tspan / dt) > max_markers:
        dt *= 2

    # Check dt after calculaing the fraction of x, y axis spanned by
    # concordia and refine if necessary...
    if eq:
        x_tmin, y_tmin = eq_xy(t1, diagram)
        x_tmax, y_tmax = eq_xy(t2, diagram)
    else:
        x_tmin, y_tmin = diseq_xy(t1, A, meas, diagram)
        x_tmax, y_tmax = diseq_xy(t2, A, meas, diagram)

    x_frac = abs((x_tmin - x_tmax) / (ax.get_xlim()[1] - ax.get_xlim()[0]))
    y_frac = abs((y_tmin - y_tmax) / (ax.get_ylim()[1] - ax.get_ylim()[0]))

    # Refine dt using some rules of thumb.
    k = 0.6
    for i in range(1, 5):
        k *= 0.5
        if x_frac < k and y_frac < k:
            dt *= 2

    return dt


def plot_age_markers(ax, markers_dict, p=0.95, pA=None):
    """
    Add age markers and/or age ellipses to plot.

    Parameters
    ----------
    markers : dict
        age marker properties, typically returned from calling get_age_markers

    """
    # unpack markers dict
    diagram = markers_dict['diagram']
    eq = markers_dict['eq']
    age_prefix = markers_dict['age_prefix']
    A = markers_dict['A']
    sA = markers_dict['sA']
    meas = markers_dict['meas']
    t = markers_dict['t']
    ell = markers_dict['ell']
    point_markers = markers_dict['point_markers']
    add_label = markers_dict['add_label']
    
    assert ell or point_markers, 'one of ell or point_markers must be True'
    n = len(t)

    # Plot markers / ellipses.
    if eq:
        x, y = eq_xy(t, diagram)
    else:
        x, y = diseq_xy(t, A, meas, diagram)

    if ell:    # plot age markers as ellipses
        # pre-allocate arrays to store ellipse params for labelling
        ell_obj = []
        bbox = []
        sx = np.empty(n)
        sy = np.empty(n)
        r_xy = np.empty(n)

        for i, age in enumerate(t):
            if eq:
                sx[i], sy[i], r_xy[i] = eq_age_ellipse(age, diagram)
            else:
                _, _, sx[i], sy[i], r_xy[i] = diseq_age_ellipse(age, A, sA, meas, pA=pA)
            ellipse = plotting.confidence_ellipse(ax, x[i], sx[i], y[i], sy[i],
                                 r_xy[i], p=p, mpl_label=f'age ellipse, {t[i]:.6f} Ma',
                                 ellipse_kw=cfg.conc_age_ellipse_kw,
                                 outline_alpha=False)
            ell_obj.append(ellipse)
            if point_markers:
                ax.plot(x[i], y[i], label='concordia marker', **cfg.conc_markers_kw)

    else:   # plot age markers only
        ax.plot(x, y, label='concordia marker', **cfg.conc_markers_kw)

    # Generate marker label text.
    age_unit = 1e-3 if age_prefix == 'ka' else 1.
    if sum(add_label) > 0:
        t_rounded = np.array([float(misc.round_down(age / age_unit, 8))
                              for age in t])
        n_dec = np.max([misc.num_dec_places(x) for x in t_rounded[add_label]
                        if add_label])

        if cfg.prefix_in_label:
            label_format = '{{:,.{}f}} {{}}'.format(n_dec)
            text = [label_format.format(x, age_prefix) for x in t_rounded]

        else:
            label_format = '{{:,.{}f}}'.format(n_dec)
            text = [label_format.format(x) for x in t_rounded]

        markers_dict['text'] = text

    markers_dict['x'] = x
    markers_dict['y'] = y
    markers_dict['add_label'] = add_label
    markers_dict['age_ellipses'] = ell

    if ell:
        markers_dict['bbox'] = bbox
        markers_dict['ell_obj'] = ell_obj
    
    return markers_dict


#=====================
# Concordia envelope
#=====================

def plot_envelope(ax, diagram, xc=None, npts=100):
    """
    Plot concordia uncertainty envelope which displays effect of decay constant
    errors.
    """
    if xc is None:
        xc = np.linspace(*ax.get_xlim(), num=100, endpoint=True)
    t = eq_age_x(xc, diagram)
    x, y = eq_xy(t, diagram)
    dy = 1.96 * eq_envelope(xc, diagram)
    ax.fill_between(xc, y + dy, y - dy, label='concordia envelope',
                    **cfg.conc_env_kw)
    ax.plot(xc, y - dy, **cfg.conc_env_line_kw, label='concordia envelope line')
    ax.plot(xc, y + dy, **cfg.conc_env_line_kw, label='concordia envelope line')


def plot_diseq_envelope(ax, ct, cx, cy, t0, t1, tbounds, A, sA, meas, diagram='tw',
                        trials=10_000, spaghetti=False):
    """
    Plot disequilibrium concordia envelope.
    """
    assert diagram == 'tw', 'concordia diagram must be in Tera-Wasserburg form'

    nx = cx.shape[0]

    # simulate activity ratios
    # TODO: should allow simulated activity ratios to be passed in (?)
    pA = cfg.rng.normal(A, sA, (trials, 4))

    # ---- testing -----
    if spaghetti:
        for i in range(trials):
            t = np.linspace(t0, t1, trials)
            xy = diseq_xy(t, pA[i, :], meas, 'tw')
            ax.plot(*xy, lw=0.5)
    # ------------------

    # get envelope limits
    y_upper = np.zeros(nx)
    y_lower = np.zeros(nx)
    # TODO: use vectorised approach
    for i in range(nx):
        y_upper[i], y_lower[i] = mc_concordia_envelope(cx[i], ct[i], pA, meas,
                                                       ax=ax)

    ok = ~np.isnan(y_lower) & ~np.isnan(y_upper)

    # plot envelope
    ax.plot(cx[ok], y_lower[ok], **cfg.conc_env_line_kw, label='concordia envelope line')
    ax.plot(cx[ok], y_upper[ok], **cfg.conc_env_line_kw, label='concordia envelope line')

    return pA


def mc_concordia_envelope(x, t0, pA, meas, ax=None):
    """
    t is on the concordia curve
    """
    trials = pA.shape[0]
    fmin, dfmin = min_tax(x, meas, diagram='tw')
    with np.errstate(all='ignore'):
        r = optimize.newton(fmin, np.full(trials, t0, dtype='double'), dfmin,
                        full_output=True, disp=False, args=([np.transpose(pA)]))
    if np.sum(r.converged) < (0.95 * trials):
        warnings.warn(f'less than 95% of Monte Carlo envelope trials succesful at x = {x:.3f}')
        y_upper, y_lower = np.nan, np.nan
    else:
        conv = r.converged & ~np.isnan(r.root)
        t = r.root[conv]
        # assert np.allclose(x, 1. / ludwig.f(t, pA[:, :3][conv]), meas=meas)
        y = ludwig.g(t, pA[:, -1][conv]) * x / cfg.U
        y_lower, y_upper = np.quantile(y, (0.025, 0.975))
    return y_upper, y_lower


#==============================================================================
# Concordia labels
#==============================================================================

def labels(ax, markers):
    """
    Add labels to concordia age markers. Uses the same offset and rotation
    for each marker.

    Parameters
    ----------
    markers : dict
        age marker properties

    """
    # Mask out values for markers that will not be labelled.
    add_label = np.array(markers['add_label'])
    x = np.array(markers['x'])[add_label]
    y = np.array(markers['y'])[add_label]
    txt = np.array(markers['text'])[add_label]
    n = sum(add_label)

    ann = []
    for i in range(n):
        an = ax.annotate(txt[i], (x[i], y[i]), **cfg.conc_text_kw,
                         label='concorida label')
        ann.append(an)
    markers['label_annotations'] = ann
    return markers


def individualised_labels(ax, markers_dict, diagram, eq=True, A=None,
                  meas=None, remove_overlaps=True):
    """
    Plot concordia age labels using individualised position and rotation.

    Notes
    -----
    This routine doesn't currently work well for disequilibrium concordia curves
    that curve back around.
    """
    assert diagram in ('tw', 'wc')
    fig = ax.get_figure()
    ell = markers_dict['age_ellipses']

    # Mask out values for markers_dict that will not be labelled.
    add_label = np.array(markers_dict['add_label'])
    if add_label.size == 0:
        warnings.warn('no labels to add within concordia age bounds')
        return
    x = np.array(markers_dict['x'])[add_label]
    y = np.array(markers_dict['y'])[add_label]
    t = np.array(markers_dict['t'])[add_label]
    txt = np.array(markers_dict['text'])[add_label]

    # Calculate some useful axes properties.
    xmin, xmax = ax.get_xlim()
    xspread = xmax - xmin
    ymin, ymax = ax.get_ylim()
    yspread = ymax - ymin

    # Get axis window extents in display points. Then calculate a "scale factor"
    # for converting slopes from data coordinates to display coordinates.
    ax_bbox = ax.get_window_extent()
    aspect_ratio = ax_bbox.height / ax_bbox.width
    scale_factor = yspread / xspread / aspect_ratio

    # Only pass in text properties to annotate(). We do not want to pass
    # in rotation or position properties for indidividualised labels.
    # TODO: consider if alignment properties could still be passed in?
    allowed_keys = ['alpha', 'backgroundcolor', 'color', 'c',
                    'fontfamily', 'family', 'fontsize', 'size',
                    'stretch', 'fontweight', 'weight', 'zorder',
                    'annotation_clip', 'clip_on']
    text_kw = {}
    for k, v in cfg.conc_text_kw.items():
        if k in allowed_keys:
            text_kw[k] = v

    # Create labels
    label_annotations = []

    n = sum(add_label)
    if ell:
        ell_obj = [b for (a, b) in zip(add_label, markers_dict['ell_obj']) if a]

    # Get annotation bbox properties for dummy annotation. Used to
    # calculate annotation box height (parallel to y-axis) in display
    # coordinates.
    an = ax.annotate(txt[0], (x[0], y[0]), va='center', ha='center',
             rotation=0, **text_kw)
    # fig.canvas.draw()
    anbbox = an.get_window_extent(renderer=fig.canvas.get_renderer())
    h = anbbox.height       # display coordinates
    an.remove()

    # TODO: alignment kwargs could be added back in here?

    text_outline = None
    # ------- testing only ------
    # text_outline = dict(ec='red', lw=0.5, pad=0.0, fc='none')
    # ----------------------------------

    # add each marker
    for i in range(n):
        # Add annotation to figure at arbitraty location (on top of marker),
        # so we can get it's bounding box dimensions.
        an = ax.annotate(txt[i], (x[i], y[i]), va='center', ha='center',
                         label='concordia label', rotation=0, bbox=text_outline,
                         **text_kw)
        # anbbox = an.get_window_extent()
        # fig.canvas.draw()
        an_bbox = an.get_window_extent(renderer=fig.canvas.get_renderer())
        w = an_bbox.width
        x_disp, y_disp = ax.transData.transform((x[i], y[i]))

        # ---testing---
        # ax.plot(x[i], y[i], 'bo', ms=2, zorder=100)
        # ----

        # Get concordia slope (i.e. tangent) at t.
        if eq:
            slope = eq_slope(t[i], diagram)
        else:
            slope = diseq_slope(t[i], A, meas, diagram)

        # angle of concordia at marker location in display coords:
        # angle = np.arctan(slope / scale_factor) * 180 / np.pi
        angle = np.arctan2(slope, 1.)
        angle_disp = np.arctan2(slope / scale_factor, 1.)

        # ------ testing only - comment out -------
        #plot line parallel to concordia slope
        # b = slope
        # a = y[i] - b * x[i]
        # ax.plot((xmin, xmax), (a + b * xmin, a + b * xmax)), 'b--', lw=0.5, zorder=100)
        # ---------------------------

        # theta is the angle from positive x axis to offset the text box (for
        # normal markers, this will be orthoganol to the slope of the
        # concordia):
        orth_angle = angle + np.pi / 2.
        orth_angle_disp = angle_disp + np.pi / 2.

        # Find slope and intercept of a line running orthogonal (on scaled plot)
        # to the concordia curve at x, y:
        # b = np.tan(theta * np.pi / 180)
        orth_slope = np.tan(orth_angle)           # true orthogonal line (data coords)
        b_disp = orth_slope * scale_factor   # apparent orthogonol line (display coords)
        b = orth_slope * scale_factor ** 2   # apparent orthogonal line (data coords)

        # x,y in display coords
        # x_disp, y_disp = ax.transData.transform((x[i], y[i]))
        # a_disp = y_disp - b_disp * x_disp

        # ------ testing only - comment out-------
        # plot line orthogonal to slope
        # a = y[i] - b * x[i]
        # xx = np.linspace(*ax.get_xlim())
        # yy = a + b * xx
        # ax.plot(xx, yy, 'm--', lw=0.5, zorder=100)
        # ---------------------------

        if cfg.rotate_conc_labels:
            if cfg.perpendicular_rotation:
                if diagram == 'tw':
                    an.set_rotation(orth_angle_disp * 180 / np.pi)
                elif diagram == 'wc':
                    an.set_rotation(orth_angle_disp * 180 / np.pi)
                d = w / 2.

            else:   # rotate parallel
                if diagram == 'tw':
                    an.set_rotation(angle_disp * 180 / np.pi)
                elif diagram == 'wc':
                     an.set_rotation(angle_disp * 180 / np.pi)
                d = h / 2.
        else:
            # Find intersection points of this orthogonal slope line and text
            # bbox:  # need to call this again after rotation
            a_disp = y_disp - b_disp * x_disp
            ints = plotting.box_line_intersection(an_bbox.xmin, an_bbox.xmax,
                        an_bbox.ymin, an_bbox.ymax, a_disp, b_disp)
            # offset text box by this extra amount (data coords)
            d = np.sqrt(np.sum((ints[1] - ints[0]) ** 2)) / 2.


        # Compute offsets
        f = cfg.offset_factor
        # d is the mimumum offset, f is extra user-defined offset (relative
        #   label height and thus textsize)

        # dx_disp = (d + f * h) * -np.abs(np.cos(orth_angle_disp))
        # dy_disp = (d + f * h) * -np.sign(orth_slope) * np.sin(orth_angle_disp)
        dx_disp = -np.sqrt((d + f * h) ** 2 / (b_disp ** 2  + 1.))
        if diagram == 'tw' and b_disp < 0:
            dx_disp *= -1.
        dy_disp = b_disp * dx_disp

        # If age ellipses, add extra offset label along ellipse axis:
        if ell:
            # ---- testing -----
            # plot bbox around ellipse
            # ellbb = ax.transData.inverted().transform(ell_obj[i].get_extents())
            # rec = Rectangle((np.min(ellbb[:, 0]), np.min(ellbb[:, 1])),
            #                 ell_obj[i].width, ell_obj[i].height,
            #                 fc='none', ec='red', lw=0.5, zorder=100)
            # ax.add_patch(rec)
            # ------------------

            # Get slope and y-int of line projected through ellipse major axis.

            # ---- old appraoch -----
            # WARNING !! This doesn't always work as expected !!
            # e_angle = ell_obj[i].get_angle() * np.pi / 180
            # eb = np.tan(e_angle)
            # eb_disp = eb / scale_factor
            # -------

            # get ellipse bbox from path:
            # path_xy  = ax.transData.inverted().transform(ell_obj[i].get_verts())
            path_xy = ell_obj[i].get_verts()
            ind_xmin = np.argmin(path_xy[:, 0])
            ind_xmax = np.argmax(path_xy[:, 0])
            ind_ymin = np.argmin(path_xy[:, 1])
            ind_ymax = np.argmax(path_xy[:, 1])

            xy_xmin = path_xy[ind_xmin, :]
            xy_xmax = path_xy[ind_xmax, :]
            xy_ymin = path_xy[ind_ymin, :]
            xy_ymax = path_xy[ind_ymax, :]

            # --- tesing ----
            # for xy in (xy_xmin, xy_xmax, xy_ymin, xy_ymax):
            #     xy = ax.transData.inverted().transform(xy)
            #     ax.plot(*xy, 'mo', ms=4, zorder=120)
            # -------

            # offset along line parralell / orthogonal to ellipse extreme values
            if diagram == 'tw':
                if np.isclose(xy_ymax[0],  xy_ymin[0]):
                    # eb_disp = 0.
                    # no correlation - offset along conc. slope:
                    eb_disp = b_disp
                else:
                    eb_disp = (xy_ymax[1] - xy_ymin[1]) / (xy_ymax[0] - xy_ymin[0])
            else:
                if np.isclose(xy_xmax[1], xy_xmin[1]):
                    # eb_disp = np.inf
                    # no correlation - offset along conc. slope:
                    eb_disp = b_disp
                else:
                    eb_disp = (xy_xmax[1] - xy_xmin[1]) / (xy_xmax[0] - xy_xmin[0])

            eb = eb_disp * scale_factor
            ea = y[i] - eb * x[i]
            ea_disp = y_disp - eb_disp * x_disp

            # ---- testing -----
            # ax.plot((xmin, xmax), (ea + eb * xmin, ea + eb * xmax),
            #         lw=0.5, c='red', ls='-')
            # xmin_disp = ax.get_window_extent().xmin
            # ymin_disp = ax.get_window_extent().ymin
            # xmax_disp = ax.get_window_extent().xmax
            # ymax_disp = ax.get_window_extent().ymax
            # y1 = ea_disp + eb_disp * xmin_disp
            # y2 = ea_disp + eb_disp * xmax_disp
            # x1, y1 = ax.transData.inverted().transform((xmin_disp, y1))
            # x2, y2 = ax.transData.inverted().transform((xmax_disp, y2))
            # ax.plot((x1, x2), (y1, y2), lw=0.5, c='blue', ls='-')
            # -------

            # find intercept points b/w conc slope line and bbox defined
            # by x, y min and max points on ellipse (calling get_window_extents
            # doesn't seem to work properly all the time?)
            ints = plotting.box_line_intersection(xy_xmin[0], xy_xmax[0],
                                xy_ymin[1], xy_ymax[1], ea_disp, eb_disp)

            if ints.size == 0:
                warnings.warn(f'individualised age ellipse label routine failed '
                              f'for {t[i]} Ma marker')
                np.array(markers_dict['add_label'])[i] = False
                an.remove()
                continue

            # ----- testing ----
            # ax.plot(*ax.transData.inverted().transform(ints[0, :]), 'bo', ms=2, zorder=100)
            # ax.plot(*ax.transData.inverted().transform(ints[1, :]), 'bo', ms=2, zorder=100)
            # dd = ints[0, :]  - np.array((x_disp, y_disp))
            # dx = dd[0]
            # dy = dd[1]
            # dist = np.sqrt(dx ** 2 + dy ** 2)
            # -----

            # offset text box by this extra amount (data coords)
            d2 = np.sqrt(np.sum((ints[1] - ints[0]) ** 2)) / 2.

            dx_disp2 = -np.sqrt(d2 ** 2 / (eb_disp ** 2 + 1.))
            dy_disp2 = eb_disp * dx_disp2

            # ---testing---
            # x2, y2 = ax.transData.inverted().transform((x_disp + dx_disp2, y_disp + dy_disp2))
            # ax.plot(x2, y2, 'go', ms=2, zorder=100)
            # ----

            dx_disp += dx_disp2
            dy_disp += dy_disp2

        # Transform x, y from display coords back to data coords and re-set
        # annotation position.
        # xy = ax.transData.inverted().transform((x_disp, y_disp))
        x_dat, y_dat = ax.transData.inverted().transform((x_disp + dx_disp,
                                                          y_disp + dy_disp))
        an.set_position((x_dat, y_dat))


        label_annotations.append(an)

    if remove_overlaps:
        remove_overlapping_labels(label_annotations)

    markers_dict['label_annotations'] = label_annotations
    return markers_dict


def remove_overlapping_labels(an):
    """
    Naive routine for removing overlapping concordia labels older than first
    overlap point.

    an : array-like
        list of concordia age marker / ellipse label annotations
    """
    n = len(an)
    overlap = np.full(n, False)

    # Check if annotation bbox fully (?) overlaps axis bbox,
    # and flag if not.
    for i in range(n - 1):
        b1 = an[i].get_window_extent()  # in display coords
        b2 = an[i + 1].get_window_extent()
        if b1.overlaps(b2):
            overlap[i + 1] = True
        else:
            overlap[i + 1] = False
    n_overlap = np.sum(overlap == True)

    # If overlaps found:
    # 1. If only youngest label overlaps, remove this and keep all others.
    # 2. If n > 5 and only youngest 2 overlap, remove these 2 and keep others.
    # 3. Otherwise remove from oldest down to first overlap.
    if n_overlap > 0:
        if n > 2 and n_overlap == 1 and overlap[0]:
            an[0].remove()
        elif n > 5 and n_overlap == 2 and overlap[0] and overlap[1]:
            an[0].remove()
            an[1].remove()
        else:
            for j in reversed(range(np.argwhere(overlap).min(), n)):
                an[j].remove()
