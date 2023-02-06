"""
Concordia plotting routines for (dis)equilibrium U-Pb datasets.

"""

import warnings
import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d

from . import plotting, ludwig
from . import cfg
from . import misc
from . import useries


exp = np.exp


#=================================
# Concordia plotting routines
#=================================

def plot_concordia(ax, diagram, plot_markers=True, env=False,
                   age_ellipses=False, marker_max=None, marker_ages=(),
                   auto_markers=True, remove_overlaps=True, age_prefix='Ma'):
    """
    Plot equilibrium U-Pb concordia curve on concordia diagram.
    
    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes object to plot concordia curve in.
    diagram : {'tw', 'wc'}
        Concordia diagram type.
    marker_max: float, optional
        User specified age marker max (Ma).

    Raises
    -------
    UserWarning: if concordia lies entirely outside the axis limits.
    
    """

    assert diagram in ('wc', 'tw')
    # get hard age bounds
    t_bounds = cfg.conc_age_bounds
    if auto_markers and (marker_max is not None):
        if not marker_max > t_bounds[0]:
            raise ValueError('marker_max value must be greater than the lower conc_age_bound value')

    # get t values at axis extremes
    t_limits, nsegs, code = age_limits(ax, diagram, t_bounds=t_bounds)
    if nsegs > 1:
        raise ValueError('cannot have more than 1 concordia segment '
                         'for equilibrium concordia')
    if code != 0:
        return  # outside axis limits

    t_limits = t_limits[0]

    # get equally spaced points
    xy = equi_points(*t_limits, ax.get_xlim(), ax.get_ylim(), diagram)[1:]
    # plot line
    ax.plot(*xy, **cfg.conc_line_kw, label='concordia line')
    # plot envelope
    if env:
        plot_envelope(ax, diagram)
    # plot markers
    if plot_markers:
        if auto_markers and (marker_max is not None) and (marker_max < t_limits[1]):
            t_limits[1] = marker_max
            if marker_max < t_limits[0]:
                warnings.warn('marker_max age is less than auto lower limit - no markers to plot')
                return
        markers = get_age_markers(ax, *t_limits, t_bounds, diagram, auto=auto_markers,
                                  marker_ages=marker_ages, ell=age_ellipses,
                                  age_prefix=age_prefix)
        markers = plot_age_markers(ax, markers)
        # label markers
        if cfg.individualised_labels:
            individualised_labels(ax, markers, diagram,
                                  remove_overlaps=remove_overlaps)
        else:
            labels(ax, markers)


def plot_diseq_concordia(ax, A, init, diagram, sA=None, age_ellipses=False,
                         plot_markers=True, marker_max=None, env=False, env_method='mc',
                         marker_ages=(), auto_markers=True, spaghetti=False, pA=None,
                         remove_overlaps=True, env_trials=1_000, negative_ratios=True,
                         age_prefix='Ma', ):
    """
    Plot disequilibrium U-Pb concordia curve on concordia diagram.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes object to plot concordia curve in.
    A : array-like
        one-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U]
    init : array-like
        two-element list of boolean values, the first is True if [234U/238U]
        is an initial value and False if a present-day value, the second is True
        if [230Th/238U] is an initial value and False if a present-day value
    sA : array-like, optional
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A
    diagram : {'tw', 'wc'}
        Concordia diagram type.
    marker_max: float, optional
        User specified age marker max (Ma).
    spaghetti : bool
        plot each simulated line using arbitrary colours

    Raises
    -------
    UserWarning: if concordia lies entirely outside the axis limits.

    """
    # get hard age limits depending on which (if any) activity ratios are
    # present values
    if not init[1]:
        t_bounds = cfg.diseq_conc_age_bounds[2]
    elif not init[0]:
        t_bounds = cfg.diseq_conc_age_bounds[1]
    else:
        t_bounds = cfg.diseq_conc_age_bounds[0]

    # get t-limits
    t_limits, nsegs, code = age_limits(ax, diagram, eq=False, A=A, init=init,
                                       t_bounds=t_bounds)

    if code != 0:
        return  # outside axis limits

    for s in range(nsegs):
        # get equally spaced x, y points
        t, x, y = diseq_equi_points(*t_limits[s], ax.get_xlim(), ax.get_ylim(),
                                    A, init, diagram)

        # plot line
        ax.plot(x, y, **cfg.conc_line_kw, label='concordia line')
        # plot envelope
        if env:
            if s > 0:
                warnings.warn('concordia envelope for first segment only will be plotted')
            else:
                if sA is None or all([x == 0 for x in sA]):
                    warnings.warn('cannot plot disequilibrium concordia age ellipses if no '
                              'uncertainity assigned to activity ratios')
                else:
                    assert env_method in ('mc', 'analytical')
                    if env_method == 'mc':
                        # only plot envelope for first segment for now
                        mc_diseq_envelope(ax, t_limits[s], t_bounds, A, sA,
                                  diagram='tw', init=init, trials=env_trials,
                                  spaghetti=spaghetti, negative_ratios=negative_ratios,
                                  pA=pA)
                    # else:
                        # analytical_diseq_envelope(ax, t, A, sA, 'tw', init=init)

    # plot markers
    if plot_markers:
        if auto_markers and marker_max is not None and marker_max < t_limits[0][1]:
            t_limits[0][1] = marker_max
            if marker_max < t_limits[0][0]:
                return
        if age_ellipses and (sA is None or all([x == 0 for x in sA])):
            warnings.warn('cannot plot disequilibrium concordia age ellipses if no '
                          'uncertainity assigned to activity ratios')
            return

        # plot markers for first segment...
        markers = get_age_markers(ax, *t_limits[0], t_bounds, diagram, A=A, sA=sA,
                                  init=init, eq=False, ell=age_ellipses, auto=auto_markers,
                                  marker_ages=marker_ages, negative_ratios=negative_ratios,
                                  age_prefix=age_prefix)
        markers = plot_age_markers(ax, markers)
        # plot markers for second segment...
        if nsegs > 1:
            if auto_markers and (marker_max is not None) and marker_max < t_limits[1][1]:
                t_limits[1][1] = marker_max
                if marker_max < t_limits[1][0]:
                    return
            markers = segment_2_markers(t_limits, markers)
            markers = plot_age_markers(ax, markers)
        # label markers
        if cfg.individualised_labels:
            individualised_labels(ax, markers, diagram, eq=False, A=A, init=init,
                                  remove_overlaps=remove_overlaps)
        else:
            labels(ax, markers)


def segment_2_markers(t_bounds, markers):
    """ Function for adding markers to the second concordia
    segment (uses spacing etc. from first segement).
    """
    t0 = markers['t'][0]
    dt = markers['dt']
    t1, t2 = t_bounds[1]
    t = np.arange(t0, t2 + dt, dt)
    if np.all(markers['add_label']):
        add_label = [True for x in range(len(t))]
    else:
        den = 2 if markers['add_label'] else 1
        add_label = [True if i % den == 0 else False for i, x in
                        enumerate(range(len(t)))]
    markers['t'] = t
    markers['add_label'] = add_label
    return markers


#==============================================================================
# Eq concordia functions
#==============================================================================

def conc_xy(t, diagram):
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


def conc_age_x(x, diagram):
    """
    Age of point on concordia at given x value.
    """
    assert diagram in ('tw', 'wc')
    if diagram == 'wc':
        t = 1. / cfg.lam235 * np.log(1. + x)
    else:
        t = 1. / cfg.lam238 * np.log(1. + 1. / x)
    return t


def conc_slope(t, diagram):
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


def conc_age_ellipse(t, diagram):
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
        x, y = conc_xy(t, diagram)
        sx = - x ** 2 * t * exp(lam238 * t) * s238
        sy = x * t * np.sqrt((exp(lam235 * t) * s235 / cfg.U) ** 2 + (
                (y * exp(lam238 * t) * s238) ** 2))
        cov_xy = x ** 3 * y * (t * exp(lam238 * t) * s238) ** 2
    return sx, sy, cov_xy


def conc_envelope(x, diagram):
    """
    Uncertainty in y for a given x value along the concordia to
    display the effects of decay constant errors on trajectory of the concordia
    curve. Requires computing uncertainty in y for a given x value using
    first-order error propagation.
    """
    assert diagram in ('tw', 'wc')
    lam238, lam235, s238, s235 = cfg.lam238, cfg.lam235, cfg.s238, cfg.s235
    t = conc_age_x(x, diagram)
    if diagram == 'wc':
        sy =  t * exp(lam238 * t) * np.sqrt(s238 ** 2
                + (lam238 / lam235 * s235) ** 2)
    else:
        sy = x * t * exp(lam235 * t) / cfg.U * np.sqrt(s235 ** 2
                + (lam235 / lam238 * s238) ** 2)
    return sy


def velocity(t, xlim, ylim, diagram):
    """
    Estimate dr/dt, which is "velocity" along an equilibrium concordia curve in
    x-y space. Uses axis coordinates to circumvent scaling issues.
    """
    # TODO: this could be calculated analytically for eq case ??
    h = 1e-08 * t
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    x2, y2 = conc_xy(t + h, diagram)
    x1, y1 = conc_xy(t - h, diagram)
    v = np.sqrt(((x2 - x1) / xspan) ** 2 + ((y2 - y1) / yspan) ** 2) / (2. * h)
    return v


def equi_points(t1, t2, xlim, ylim, diagram, ngp=500_000, n=500):
    """
    Uses numerical method to obtain age points that are approximately evenly
    spaced in x, y along equilibrium U-Pb concordia between age limits t1 and t2.
    """
    # TODO: could this be done analytically too?
    assert t2 > t1
    t = np.linspace(t1, t2, ngp)
    # suppress numpy warnings
    with np.errstate(all='ignore'):
        dr = velocity(t, xlim, ylim, diagram)
    # Cumulative integrated area under velocity curve (aka cumulative
    # "distance") at each t_j from t1 to t2:
    cum_r = integrate.cumtrapz(dr, t, initial=0)
    # Divide cumulative area under velocity curve into equal portions.
    rj = np.arange(n + 1) * cum_r[-1] / n
    # Find t_j value at each r_j:
    idx = np.searchsorted(cum_r, rj, side="left")
    idx[-1] = ngp - 1 if idx[-1] >= ngp else idx[-1]
    t = t[idx]
    x, y = conc_xy(t, diagram)
    return t, x, y


#==============================================================
# Diseq concordia functions
#==============================================================

def diseq_xy(t, A, init, diagram):
    """Return x, y for given t along disequilibrium concordia curve.
    """
    assert diagram in ('tw', 'wc')
    Lam238 = np.array((cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226))
    Lam235 = np.array((cfg.lam235, cfg.lam231))
    coef238 = ludwig.bateman(Lam238)
    coef235 = ludwig.bateman(Lam235, series='235U')
    if diagram == 'tw':
        x = 1. / ludwig.f(t, A[:-1], Lam238, coef238, init=init)
        y = ludwig.g(t, A[-1], Lam235, coef235) * x / cfg.U
    elif diagram == 'wc':
        y = ludwig.f(t, A[:-1], Lam238, coef238, init=init)
        x = ludwig.g(t, A[-1], Lam235, coef235)
    return x, y


def diseq_dxdt(t, A, init, diagram):
    """
    Return x given t along disequilibrium concordia curve.
    Used e.g. to compute dx/dt
    """
    assert diagram == 'tw'
    def conc_x(t):
        x, _ = diseq_xy(t, A, init, diagram)
        return x
    h = abs(t) * np.sqrt(np.finfo(float).eps)
    dxdt = misc.cdiff(t, conc_x, h)
    return dxdt


def diseq_slope(t, A, init, diagram):
    """
    Compute tangent to concordia at given t. I.e. dy/dx for given t.
    """
    h = np.sqrt(np.finfo(float).eps) * t
    x2, y2 = diseq_xy(t + h, A, init, diagram)
    x1, y1 = diseq_xy(t - h, A, init, diagram)
    return (y2 - y1) / (x2 - x1)


def diseq_age_ellipse(t, A, sA, init, diagram, pA=None, trials=1_000,
                     negative_ratios=True):
    """
    Plot disequilibrium concordia marker as an "age ellipse" which provides
    a visual representation of uncertainty in x-y for given t value arising from
    uncertainties in activity ratio values.
    """
    if pA is None:
        pA = cfg.rng.normal(A, sA, (trials, 4))
        if not negative_ratios:
            warnings.warn('concordia age ellipse plotting routine does not yet account '
                          'for rejected negative activity ratios')
    else:
        assert pA.shape[1] == 4

    flags = np.zeros(trials)
    # check for negative initial [234U/238U] and [230Th/238U] solutions and
    # raise warning if too many found.
    # TODO: in future negative ar solutions may be rejected?
    if not init[0]:
        A48i = useries.ar48i(t, pA[:, 0], cfg.lam238, cfg.lam234)
        flags = np.where(A48i < 0, -1, 0)
    if not init[1]:
        A08i = useries.ar08i(t, pA[:, 0], pA[:, 1], cfg.lam238, cfg.lam234,
                             cfg.lam230, init=init[0])
        flags = np.where((A08i < 0) & (flags == 0), -2, 0)

    if sum(flags != 0) > (0.99 * trials):
        msg = f'{sum(flags != 0)}/{trials} negative activity ratio soln. values in age ellipse t = {t} Ma'
        warnings.warn(msg)

    # Compute ellipse params
    pA = np.transpose(pA)
    xpts, ypts = diseq_xy(t, pA, init, diagram)
    xy = np.vstack((xpts, ypts))

    V = np.cov(xy)
    x, y = np.nanmean(xy, axis=1)
    sx, sy = np.sqrt(V.diagonal())
    cov_xy = V[0, 1]
    # r_xy = cov_xy / (sx * sy)

    # reset r_xy for special cases:
    if sA[-1] == 0:
        if any(np.asarray(sA[:2]) != 0):
            r_xy = 1. - 1e-09
            cov_xy = r_xy * sx * sy
        pass
    else:
        if all((x == 0 for x in sA[:2])):
            sx = 0.
            r_xy = 0.
            cov_xy = r_xy * sx * sy

    # cov = np.array([[sx ** 2, r_xy * sx * sy],
    #                 [r_xy * sx * sy, sy ** 2]])

    return x, y, sx, sy, cov_xy


def diseq_equi_points(t1, t2, xlim, ylim, A, init, diagram, ngp=500_000, n=500):
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
        dr = diseq_velocity(t, xlim, ylim, A, init, diagram)
    # Cumulative integrated area under velocity curve (aka cumulative
    # "distance") at each t_j from t1 to t2:
    cum_r = integrate.cumtrapz(dr, t, initial=0)
    # Divide cumulative area under velocity curve into equal portions.
    rj = np.arange(n + 1) * cum_r[-1] / n
    # Find t_j value at each r_j:
    idx = np.searchsorted(cum_r, rj, side="left")
    idx[-1] = ngp - 1 if idx[-1] >= ngp else idx[-1]
    t = t[idx]
    x, y = diseq_xy(t, A, init, diagram)
    return t, x, y


def diseq_velocity(t, xlim, ylim, A, init, diagram):
    """
    Estimate dr/dt, which is "velocity" along a diseq concordia curve in
    x-y space. Uses axis coordinates to circumvent scaling issues.
    """
    h = 1e-08 * t
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    x2, y2 = diseq_xy(t + h, A, init, diagram)
    x1, y1 = diseq_xy(t - h, A, init, diagram)
    v = np.sqrt(((x2 - x1) / xspan) ** 2 + ((y2 - y1) / yspan) ** 2) / (2. * h)
    return v


#=================================
# Concordia age bound functions
#=================================

def age_limits(ax, diagram, eq=True, A=None, init=(True, True),
               t_bounds=cfg.conc_age_bounds, max_tries=3):
    """
    Find concordia age limits at axis window boundaries. Will return age bound
    if it is inside the axis window.

    """
    if not eq:
        assert A is not None

    code = 0
    tmin, tmax = t_bounds           # maximum and miniumum plotted age
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Get starting points:
    # Generate evenly spaced points in time and check which are in axis window.
    npts = 100_000
    for i in range(max_tries):
        tv = np.linspace(tmin, tmax, npts)
        # suppress numpy warnings:
        with np.errstate(all='ignore'):
            if eq:
                x, y = conc_xy(tv, diagram)
            else:
                x, y = diseq_xy(tv, A, init, diagram)

        # check if points are inside plot window:
        inside = np.logical_and(np.logical_and(xmin < x, x < xmax),
                                np.logical_and(ymin < y, y < ymax))

        if np.sum(inside) < 10:
            if np.sum(inside) == 0 and i == max_tries - 1:
                warnings.warn('concordia curve appears to lie outside plot window')
                code = 1
                return None, 0, code
            else:
                npts *= 10  # increase number of points and try again
        else:
            break

    limits0, nsegs = t_limits_from_cp(tv, inside, eq=eq, tmin=tmin, tmax=tmax)

    limits = []     # both sets of limits (if 2 segments)

    # Refine age limits by starting at points in box and incrementally
    # moving out until the boundary is encountered:
    # for l in limits0:
    for l0 in limits0:

        l0 = np.sort(l0)
        l = np.zeros(2)

        # Lower age limit
        if l0[0] == t_bounds[0]:    # lower bound inside window - no need to refine
            l[0] = l0[0]
        else:
            l[0], code = refine_t_limit(l0, t_bounds, (xmin, xmax), (ymin, ymax),
                            eq=eq, increasing_t=False, diagram=diagram, ell=False,
                            ax=ax, A=A, init=init)
            if code != 0:
                raise ValueError('failed to find concordia envelope lower age limit')

        # Upper age limit
        if l0[1] == t_bounds[1]:  # upper bound inside window - no need to refine
            l[1] = l0[1]
        else:
            l[1], code = refine_t_limit(l0, t_bounds, (xmin, xmax), (ymin, ymax),
                            eq=eq, increasing_t=True, diagram=diagram, ell=False,
                            A=A, init=init, ax=ax)
            if code != 0:
                raise ValueError('failed to find concordia envelope upper age limit')

        # reset limit if outside bounds
        # TODO: shouldn't need this but just in case?
        if l[0] < t_bounds[0]:
            l[0] = t_bounds[0]
        if l[1] > t_bounds[1]:
            l[1] = t_bounds[1]

        limits.append(l)

    return limits, nsegs, code


def t_limits_from_cp(t, inside, tmin, tmax, eq=True):
    """
    Get estimated concordia age bounds from arbitrary age points within
    the plot window.

    """
    # Get indices of inside / outside change points for different cases:
    idx = np.where(np.diff(inside) != 0)[0]
    limits0 = []
    nsegs = 1

    if eq and len(idx) > 2:
        raise RuntimeError('cannot have more than two change points for '
                           'equilibrium concordia')

    if len(idx) == 0:
        limits0 = [[tmin, tmax]]

    elif len(idx) == 1:  # either tmin or tmax in box
        if inside[0]:
            limits0 = [[tmin, t[idx[0]]]]
        else:
            limits0 = [[t[idx[0] + 1], tmax]]

    elif len(idx) == 2:  # neither tmin nor tmax in box
        limits0 = [[t[idx[1]], t[idx[0] + 1]]]

    else:   # 2 segments
        nsegs = 2
        if len(idx) == 3:  # 3 points inside
            pass

        if len(idx) == 4:  # 4 points inside
            # deal with first segment:
            if idx[1] - idx[0] == 1:    # one point inside
                limits0.append([t[idx[0] + 1], t[idx[0] + 1]])
            else:
                limits0.append([t[idx[0] + 1], t[idx[1]]])

            # deal with second segment:
            if idx[3] - idx[2] == 1:    # one point inside
                limits0.append([t[idx[2] + 1], t[idx[2] + 1]])
            else:
                limits0.append([t[idx[2] + 1], t[idx[3]]])

    return limits0, nsegs


def refine_t_limit(limits0, t_bounds, xlim, ylim, increasing_t,
                    diagram='tw', eq=True, A=None, sA=None, init=(True, True), ell=False,
                    trials=1_000, ax=None):
    """
    Refine concordia age limits so they lie just outside axis window. Will
    return age bound if it is inside the axis window.

    Parameters
    ----------
    tmin : float
        lowder t-bound
    tmax : float
        upper t-bound
    """
    if ell:
        denom = 100.
    else:
        denom = 1000.

    xspread = xlim[1] - xlim[0]
    xtol = xspread / denom
    yspread = ylim[1] - ylim[0]
    ytol = yspread / denom

    t0, t1 = limits0

    if eq:
        # TODO: what is going on here?
        x0, y0 = conc_xy(t0, diagram)
        x1, y1 = conc_xy(t1, diagram)
        tpts, xpts, ypts = equi_points(t0, t1, xlim, ylim, diagram, ngp=20_000,
                                       n=20)
    else:
        x0, y0 = diseq_xy(t0, A, init, diagram)
        x1, y1 = diseq_xy(t1, A, init, diagram)
        tpts, xpts, ypts = diseq_equi_points(t0, t1, (x0, x1), (y0, y1), A,
                                 init, diagram, ngp=100_000, n=20)

    # --- testing ----
    # ax.plot(xpts, ypts, 'ro', ms=2)
    # --------

    if ell:
        pA = cfg.rng.normal(A, sA, (trials, 4))

    if increasing_t:
        dt = np.diff(tpts[-2:])[0]  # spacing between last two equi age points
        t_start = t1
    else:
        dt = -np.diff(tpts[:2])[0]  # spacing between first two equi age points
        t_start = t0

    for i in range(100):
        if ell:
            t, code = walk_ellipse_outward(t_start, dt, t_bounds, xlim, ylim, diagram,
                        xtol, ytol, trials, A=A, init=init, pA=pA,
                        ax=ax)
        else:
            t, code = walk_outward(t_start, dt, t_bounds, xlim, ylim, diagram, xtol,
                        ytol, eq=eq, A=A, init=init)
        if code == 0:
            return t, code


        # if walk_outward did not meet tol, reduce and try again
        t_start = t  # reset t0
        dt /= 2.

    # TODO: throw warning, but return t anyway?
    msg = 'age limits did not meet convergence criteria, but may be good enough?'
    warnings.warn(msg)

    return t0, 0


def walk_ellipse_outward(t0, dt, t_bounds, xlim, ylim, diagram, xtol,
                    ytol, trials, A=None, init=None, pA=None, ttol=1e-03,
                    max_steps=100, p=0.999, ax=None):
    """
    Walk outward from t0 until an axis boundary is found (i.e. ellipse is
    pretty much outside axis window, where 'pretty-muchness' is determined
    by p).
    """

    assert diagram == 'tw'
    xmin, xmax = xlim
    ymin, ymax = ylim
    x0, y0 = diseq_xy(t0, A, init, 'tw')    # centre point
    pA = np.transpose(pA)
    step = 0
    while step < max_steps:
        t1 = t0 + dt
        x1, y1 = diseq_xy(t1, A, init, 'tw')    # centre point
        xpts, ypts = diseq_xy(t1, pA, init, 'tw')   # simulated x-y
        # print(f'ell step {"+" if dt > 0 else "-"}: {step}')

        # TODO: deal with negative_ratios?

        if dt > 0 and t1 > t_bounds[1]: # if bounds exceeded while still in box..
            return t_bounds[1], 0
        elif dt < 0 and t0 < t_bounds[0]:
            return t_bounds[0], 0
        elif (np.sum(xpts < xmin) / trials > p) or \
                (np.sum(xpts > xmax) / trials > p) or \
                (np.sum(ypts > ymax) / trials > p) or \
                (np.sum(ypts < ymin) / trials > p):  # outside box
            # check tolerances are met - i.e. check estimate is accurate enough
            if (abs(x1 - x0) < xtol and abs(y1 - y0) < ytol):
                return t1, 0
            else:
                return t0, 1

        # keep walking
        t0, x0, y0 = t1, x1, y1
        step += 1

    code = 1
    return t0, code


def walk_outward(t0, dt, t_bounds, xlim, ylim, diagram, xtol, ytol, eq=True,
                 A=None, init=(True, True), ttol=1e-03, max_steps=1000,
                 ell=False, pA=None):
    """
    Walk outward from t0 until an axis boundary is found.
    """
    # x0, y0 = conc_xy(t0, diagram)
    if eq:
        x0, y0 = conc_xy(t0, diagram)
    else:
        x0, y0 = diseq_xy(t0, A, init, diagram)
    step = 0
    while step < max_steps:
        t1 = t0 + dt
        if eq:
            x1, y1 = conc_xy(t1, diagram)
        else:
            x1, y1 = diseq_xy(t1, A, init, diagram)

        if dt > 0 and t1 > t_bounds[1]: # if bounds exceeded while still in box..
            return t_bounds[1], 0
        elif dt < 0 and t0 < t_bounds[0]:
            return t_bounds[0], 0
        elif (x1 > xlim[1] or x1 < xlim[0]) or (y1 > ylim[1] or y1 < ylim[0]):  # outside box
            if (abs(x1 - x0) < xtol and abs(y1 - y0) < ytol or
                    abs(t1 - t0) / t0 < ttol):
                return t1, 0
            else:
                return t0, 1

        # keep walking
        t0, x0, y0 = t1, x1, y1
        step += 1
    code = 1
    return np.nan, code


#====================
# Concordia markers
#====================

def estimate_marker_spacing(tspan):
    """
    Get initial estimate of good concordia marker spacing.

    """
    dt = 10 ** misc.get_exponent(tspan) / 8
    while abs(tspan / dt) > 12:
        dt *= 2
    return misc.round_down(dt, 8)


def get_age_markers(ax, t1, t2, t_bounds, diagram, eq=True, ell=False, A=None,
                    sA=None, init=None, marker_ages=(), age_prefix='Ma',
                    auto=True, n_segments=1, negative_ratios=True):
    """
    Add concordia markers to concordia line and generate label text (but
    does not actually add marker labels).

    Parameters
    ----------
    t1 : float
        lower concordia age in plot,
    t2 : float
        upper concordia age
    ell : bool
        plot age ellipses

    """
    assert t2 > t1
    assert age_prefix in ('ka', 'Ma')
    age_unit = 1. if age_prefix == 'Ma' else 1e-3
    t_limits = (t1, t2)
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
        n_inside = len(t[np.logical_and(t1 < t, t2 > t)])
        # which markers to label:
        if n_inside > cfg.every_second_threshold:
            add_label = [True if i % 2 == 0 else False for i, t in enumerate(t)]
        else:
            add_label = [True for t in t]

    else:  # find auto marker locations

        dt = age_marker_spacing(ax, t1, t2, diagram, A=A, init=init, eq=eq)

        # Get marker age points:
        t_start = misc.round_down(np.floor(t1 / dt) * dt, 5)
        t = np.arange(t_start, t2 + dt, dt)

        # Check if there are too many diseq age ellipses in plot window and reduce
        # dt if necessary:
        if ell and not eq:
            t0, dt0 = t, dt
            t, t_start, dt, code = age_ellipse_marker_spacing(ax, t, dt, A, sA, init, t_limits)

            if code != 0:
                warnings.warn('age ellipse spacing routine failed')
                t = t0  # go back to normal marker spacing
                dt = dt0

        # Reset to 0 if sufficiently close in order to avoid labelling problems.
        t = [0. if abs(x) < 1e-9 else x for x in t]
        t = np.array([round(x, 10) for x in t])         # round ages to get around f.p. issues

        # If labelling every second, check which label to start with. Preference
        # starting on label with less significant digits, then preference starting
        # on label ending in 1, and finally preference starting on even number.
        n = len(t)
        if ell:
            n -= 1          # on average, at least one marker will be fully outside?
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
        if not ell:
            t = np.arange(t_start - dt, t2 + 2 * dt, dt)
            start_idx = 1 if start_idx == 0 else 0

        t = np.array([round(x, 10) for x in t])         # round ages again
        t = t[(t >= t_bounds[0]) & (t <= t_bounds[1])]  # double check bounds
        num_t = len(t)                                  # new number of markers

        # list of bools indicating which markers to add a label to:
        if n > cfg.every_second_threshold:
            add_label = [True if (i - start_idx) % step == 0
                         else False for i in range(num_t)]
        else:
            add_label = [True] * len(t)

    age_markers = {'diagram': diagram, 'eq': eq, 'A': A, 'sA': sA,
                   'init': init, 'ell': ell, 't': t, 'dt': dt,
                   'add_label': add_label, 'negative_ratios': negative_ratios,
                   'age_prefix': age_prefix}

    return age_markers


def age_ellipse_limits(ax, t0, dt, t_bounds, A, sA, init, p=0.998,
            trials=1_000, diagram='tw', maxiter=40, negative_ratios=True):
    """
    Find concordia age ellipse limits near axis window boundaries. Will return
    age bound if it is inside the axis window.

    Parameters
    ----------
    t0 : array-like
        Guess at limits (points must be inside plot window).
    dt : float
        appropriate step size

    """
    assert diagram == 'tw'
    tmin, tmax = t0
    assert tmax >= tmin
    tspan = tmax - tmin
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    t1, t2 = np.nan, np.nan
    code = 0

    pA = np.transpose(cfg.rng.normal(A, sA, (trials, 4)))

    # Left limit:
    # move outward until all endpoints are outside axis window
    t2 = tmax + dt
    i = 1
    while i <= maxiter:
        i += 1
        xpts, ypts = diseq_xy(t2, pA, init, 'tw')
        if (np.sum(xpts < xmin) / trials > p) or \
            (np.sum(ypts > ymax) / trials > p) or \
            (np.sum(ypts < ymin) / trials > p):
                break
        t2 += dt
        if i == maxiter:
            code = 1
            return code, t1, t2
        if t2 > t_bounds[1]:
            t2 = t_bounds[1]
            break

    # Right limit:
    t1 = tmin
    i = 1
    while i <= maxiter:
        i += 1
        xpts, ypts = diseq_xy(t1, pA, init, 'tw')
        if (np.sum(xpts > xmax) / trials > p) or \
            (np.sum(ypts > ymax) / trials > p) or \
            (np.sum(ypts < ymin) / trials > p):
            break
        t1 -= dt
        if t1 < t_bounds[0]:
            t1 = t_bounds[0]
            break
        if i == maxiter:
            code = 1
            return code, t1, t2

    return code, t1, t2


def age_marker_spacing(ax, t1, t2, diagram, A=None, init=None, eq=True,
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
        x_tmin, y_tmin = conc_xy(t1, diagram)
        x_tmax, y_tmax = conc_xy(t2, diagram)
    else:
        x_tmin, y_tmin = diseq_xy(t1, A, init, diagram)
        x_tmax, y_tmax = diseq_xy(t2, A, init, diagram)

    x_frac = abs((x_tmin - x_tmax) / (ax.get_xlim()[1] - ax.get_xlim()[0]))
    y_frac = abs((y_tmin - y_tmax) / (ax.get_ylim()[1] - ax.get_ylim()[0]))

    # Refine dt using some rules of thumb.
    k = 0.6
    for i in range(1, 5):
        k *= 0.5
        if x_frac < k and y_frac < k:
            dt *= 2

    return dt


def age_ellipse_marker_spacing(ax, t, dt, A, sA, init, t_bounds, diagram='tw',
                              maxiter=30, trials=1_000):
    """
    Starting with regular concordia age marker spacing and limits, check if
    these are also suitable for disequilibrium age ellipses.
    """
    assert diagram == 'tw'

    idx = [int(np.floor((len(t) - 1)/2)), int(np.ceil((len(t) - 1)/2))]
    t0 = [t[idx[0]], t[idx[1]]]
    code, t1, t2 = age_ellipse_limits(ax, t0, dt, t_bounds, A, sA, init,
                                      p=0.998, trials=1_000)

    if code != 0:
        return np.nan, np.nan, np.nan, code

    t1 = round(t1, 10)
    t2 = round(t2, 10)

    dt = age_marker_spacing(ax, t1, t2, diagram, A=A, init=init, eq=False,
                            max_markers=8)

    # Get marker age points:
    t_start = misc.round_down(np.floor(t1 / dt) * dt, 5)
    t = np.arange(t_start, t2 + dt, dt)

    return t, t_start, dt, code


def plot_age_markers(ax, markers, p=0.95):
    """
    Add age markers to plot.

    Parameters
    ----------
    markers : dict
        age marker properties, typically returned from calling get_age_markers

    """
    # unpack markers dict
    diagram = markers['diagram']
    eq = markers['eq']
    age_prefix = markers['age_prefix']

    A = markers['A']
    sA = markers['sA']
    init = markers['init']
    negative_ratios = markers['negative_ratios']

    t = markers['t']
    ell = markers['ell']
    add_label = markers['add_label']

    n = len(t)

    # Plot markers / ellipses.
    if eq:
        x, y = conc_xy(t, diagram)
    else:
        x, y = diseq_xy(t, A, init, diagram)

    if ell:    # plot age markers as ellipses
        ell_obj = []
        bbox = []
        xm = np.empty(n)    # use mean simulated x, y in case rejecting negative_ratios
        ym = np.empty(n)
        sx = np.empty(n)
        sy = np.empty(n)
        cov_xy = np.empty(n)

        for i, age in enumerate(t):
            if eq:
                xm[i], ym[i] = conc_xy(age, diagram)
                sx[i], sy[i], cov_xy[i] = conc_age_ellipse(age, diagram)
            else:
                xm[i], ym[i], sx[i], sy[i], cov_xy[i] = diseq_age_ellipse(
                    age, A, sA, init, diagram, negative_ratios=negative_ratios
                )

        for i in range(n):
            V = np.diag((sx[i], sy[i])) ** 2
            V[0, 1] = V[1, 0] = cov_xy[i]
            # --- choose either this ----
            # ellipse = plotting.confidence_ellipse2(ax, x[i], y[i], V,
            #                     n_std=2.0, **cfg.conc_ellipse_kw)
            # --- or this ----
            r_xy = cov_xy[i] / (sx[i] * sy[i])

            if abs(r_xy) > (1 - 1e-8):
                # msg = f'age ellipse correlation is greater than 1, r_xy = {r_xy} '
                # warnings.warn(msg)
                r_xy = np.sign(r_xy) * (1. - 1e-8)   # reset to ~1?

            ellipse = plotting.confidence_ellipse(ax, xm[i], sx[i], ym[i], sy[i],
                                 r_xy, p=p, mpl_label=f'age ellipse, {t[i]:.6f} Ma',
                                 ellipse_kw=cfg.conc_age_ellipse_kw,
                                 outline_alpha=False)
            # ----------------
            ell_obj.append(ellipse)

    else:   # plot age markers
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

        markers['text'] = text

    markers['x'] = x
    markers['y'] = y
    markers['add_label'] = add_label
    markers['age_ellipses'] = ell

    if ell:
        markers['bbox'] = bbox
        markers['ell_obj'] = ell_obj
    
    return markers


#=====================
# Concordia envelope
#=====================

def plot_envelope(ax, diagram, npts=100):
    """
    Plot concordia uncertainty envelope which displays effect of decay constant
    errors.
    """
    xx = np.linspace(*ax.get_xlim(), num=100, endpoint=True)
    t = conc_age_x(xx, diagram)
    x, y = conc_xy(t, diagram)
    dy = 1.96 * conc_envelope(xx, diagram)
    ax.fill_between(xx, y + dy, y - dy, label='concordia envelope',
                    **cfg.conc_env_kw)
    ax.plot(xx, y - dy, **cfg.conc_env_line_kw, label='concordia envelope line')
    ax.plot(xx, y + dy, **cfg.conc_env_line_kw, label='concordia envelope line')


def mc_diseq_envelope(ax, t_limits, t_bounds, A, sA, diagram='tw',
                init=(True, True), trials=1_000, limit_trials = 1_000,
                spaghetti=False, maxiter=50, negative_ratios=True, pA=None):
    """
    Plot uncertainty envelope about disequilibrium concordia based on Monte
    Carlo simulation. This displays uncertainty in trajectory of concordia
    arising from uncertainty in activity ratio values.

    Parameters
    ----------
    t_limits : array-like
        minimum and maximum concordia ages for given axis window
    t_bounds :
        minimum and maximum age bounds for full concordia curve

    """
    assert diagram == 'tw'
    tmin, tmax = t_limits      # actual age bounds for 'middle' curve only
    assert tmax > tmin
    tspan = tmax - tmin
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    points = 1000
    interp_points = 250

    # Concordia limits - use as initial guess at envelope t limits
    l0 = np.sort(t_limits)

    # Refine age limits by starting at points in box and incrementally
    # moving out until the boundary is encountered:
    # for l in limits0:
    l = np.zeros(2)

    # Lower age limit
    if l0[0] == t_bounds[0]:    # lower bound inside window - no need to refine
        l[0] = l0[0]
    else:
        l[0], code = refine_t_limit(l0, t_bounds, (xmin, xmax), (ymin, ymax),
                        eq=False, increasing_t=False, diagram=diagram,
                            A=A, sA=sA, init=init, ell=True, ax=ax)
        if code != 0:
            raise ValueError('failed to find concordia envelope lower age limit')

    # Upper age limit
    if l0[1] == t_bounds[1]:  # lower bound inside window - no need to refine
        l[1] = l0[1]
    else:
        l[1], code = refine_t_limit(l0, t_bounds, (xmin, xmax), (ymin, ymax),
                        eq=False, increasing_t=True, diagram=diagram,
                            A=A, sA=sA, init=init, ell=True, ax=ax)
        if code != 0:
            raise ValueError('failed to find concordia envelope upper age limit')

    # reset limit if outside bounds
    # TODO: shouldn't need this?
    if l[0] < t_bounds[0]:
        l[0] = t_bounds[0]
    if l[1] > t_bounds[1]:
        l[1] = t_bounds[1]

    t0, t1 = l
    t, x, y = diseq_equi_points(t0, t1, ax.get_xlim(), ax.get_ylim(), A, init,
                                diagram, n=points)
    t = t[:points]  # sometimes diseq_equi_points returns vector of length n + 1?

    # Remove duplicate t values
    # TODO: find a proper solution to this!
    t, ind = np.unique(t, return_index=True)
    x = x[ind]
    y = y[ind]
    points = len(t)
    # ----

    # reset xmin and xmax
    xmin = np.min(x)
    xmax = np.max(x)

    # Perturb activity ratios
    if pA is None:
        pA = cfg.rng.normal(A, sA, (trials, 4))
        if not negative_ratios:
            warnings.warn('concordia envelope plotting does not yet account for '
                          'rejected negative activity ratio trials')
    else:
        assert pA.shape[1] == 4

    # check if dx/dt changes sign, if so, there are multiple y for x, and
    # therefore t needs to be truncated to plot envelope
    dxdt = diseq_dxdt(t, A, init, diagram)
    ind = np.where(np.diff(np.sign(dxdt)) != 0)[0]
    if ind.shape[0] != 0:

        if ind.shape[0] > 1:    # if multiple dx/dt changes, probably a numerical issue computing deriv.
            warnings.warn(f'multiple dx/dt sign changes in concordia - no truncation')

        else:
            ind = np.min(ind)
            t = t[:ind]

            # refine change of sign x-limit....
            markers_dict = dict(
                diagram='tw', eq=False, A=A, sA=sA, init=init,
                negative_ratios=True, t=np.array([t[-1]]), ell=True,
                add_label=np.array([False]), age_prefix='Ma')
            plot_age_markers(ax, markers_dict, p=0.99)
            # get ellipse bounds...
            ell_obj = ax.patches[-1]
            ax.get_figure().canvas.draw()
            ell_bbox = ax.transData.inverted().transform(ell_obj.get_extents())
            xmin = np.max(ell_bbox[:, 0])
            ell_obj.remove()
            warnings.warn(f'concordia envelope truncated at t = {t[-1]} Ma because dx/dt changes sign')

            # Get evenly spaced x,y points over truncated t
            t, x, y = diseq_equi_points(t0, t[-1], ax.get_xlim(), ax.get_ylim(), A,
                                        init, diagram, n=points)
            t = t[:points]  # sometimes diseq_equi_points returns vector of length n + 1?
            # Remove duplicate t values
            # TODO: find a proper solution to this!
            t, ind = np.unique(t, return_index=True)
            x = x[ind]
            y = y[ind]
            # ----

            points = len(t)  # sometimes diseq_equi_points returns vector of length n + 1?

    # pre-allocate arrays to store interpolated points
    xv = np.linspace(xmin, xmax, interp_points)
    yv = np.zeros((trials, interp_points))

    # pre-allocate arrays to store simulated curves
    xc = np.zeros((trials, points))
    yc = np.zeros((trials, points))

    flags = np.zeros(points)
    for i in range(trials):
        # check for negative inital [234U/238U] and [230Th/238U] solutions and
        # reject these simulated pA curves - these really mess up envelope!
        if not init[0]:
            if any(useries.ar48i(t, pA[i, 0], cfg.lam238, cfg.lam234) < 0):
                flags[i] = -1
        if not init[1]:
            if any(useries.ar08i(t, *pA[i, :2], cfg.lam238, cfg.lam234,
                                 cfg.lam230, init=init[0]) < 0):
                if flags[i] != -1:
                    flags[i] = -2

        xc[i, :], yc[i, :] = diseq_xy(t, pA[i, :], init, 'tw')
        if spaghetti:
            ax.plot(xc[i, :], yc[i, :], ms=0, lw=0.50)
        else:
            # Sample at common x spacing
            f = interp1d(xc[i, :], yc[i, :], kind='cubic', bounds_error=False,
                         fill_value='extrapolate')
            yv[i, :] = f(xv)

    if sum(flags != 0) != 0:
        msg = f'{sum(flags != 0)} / {points} negative activity ratio soln. values in MC concordia env.'
        warnings.warn(msg)
        warnings.warn('MC concordia envelope may be unreliable for given activity ratio values over this age range')
        # yv = yv[flags == 0]

    # Estimate confidence interval from quantiles:
    y_low, y_hi = np.quantile(yv, (0.025, 0.975), axis=0)
    ax.plot(xv, y_low, **cfg.conc_env_line_kw, label='concordia envelope line')
    ax.plot(xv, y_hi, **cfg.conc_env_line_kw, label='concordia envelope line')
    ax.fill_between(xv, y_low, y2=y_hi, **cfg.conc_env_kw,
                    label='concordia envelope')


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
                  init=None, remove_overlaps=True):
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

    # Mask out values for markers_dict that will not be labelled.
    add_label = np.array(markers_dict['add_label'])
    x = np.array(markers_dict['x'])[add_label]
    y = np.array(markers_dict['y'])[add_label]
    t = np.array(markers_dict['t'])[add_label]
    txt = np.array(markers_dict['text'])[add_label]
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
            slope = conc_slope(t[i], diagram)
        else:
            slope = diseq_slope(t[i], A, init, diagram)

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
