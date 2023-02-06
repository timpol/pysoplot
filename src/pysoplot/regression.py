"""
Linear regression algorithms for 2-dimensional datasets.

"""

import numpy as np

# from matplotlib.offsetbox import AnchoredText

from . import cfg
from . import exceptions
from . import stats
from . import plotting


#=================================
# Regression fitting routines
#=================================

def classical_fit(x, sx, y, sy, r_xy, model='ca', plot=False, diagram=None,
        isochron=True, xlim=(None, None), ylim=(None, None),
        axis_labels=(None, None), norm_isotope=None, dp_labels=None):
    """
    Fit a classical linear regression line to 2-dimensional data and optionally
    create plot of the fit. If model is set to 'ca', then routine will emulate
    default the protocols of Isoplot Ludwig (2012).

    Parameters
    -----------
    x : np.ndarray
        x values (as 1-dimensional array)
    sx : np.ndarray
        analytical uncertainty on x at :math:`1\sigma` abs.
    y : np.ndarray
        y values
    sy : np.ndaray
        analytical uncertainty on y at :math:`1\sigma` abs.
    r_xy : np.ndarray
        x-y correlation coefficient
    isochron : bool, optional
        If true and model is 'ca' then fits model 3 for excess scatter data
        sets instead of model 2.
    model: {'ca', 'c1', 'c2', 'c3'}
        Regression model to fit to data.

    Notes
    -----
    `model` should be one of

    - 'ca' fit 'best' model depending on MSWD of york fit; emulates Isoplot behaviour
    - 'c1' standard York fit with analytical errors
    - 'c2' model of McSaveney in Faure (1977), equivalent to Isoplot model 2
    - 'c3' equivalent to Isoplot model 3

    References
    ..........

    Faure, G., 1977. Appendix 1: Fitting of isochrons for dating by the Rb-Sr
    method, in: Principles of Isotope Geology. John Wiley and Sons, pp. 1–17.

    Ludwig, K.R., 2012. Isoplot/Ex Version 3.75: A
    Geochronological Toolkit for Microsoft Excel, Special Publication 4.
    Berkeley Geochronology Center.

    """
    assert model in ('ca', 'c1', 'c2', 'c3')

    n = x.size
    if not n == sx.size == y.size == sy.size == r_xy.size:
        raise ValueError('incompatible number of elements in input data arrays')
    if n == 1:
        raise ValueError('cannot fit linear regression to a single data point')

    df = n - 2
    sy_excess = None

    # fit York model:
    theta, covtheta, mswd, xbar, ybar, r = york(x, sx, y, sy, r_xy)

    # get pr fit:
    p = stats.pr_fit(df, mswd)

    # get mswd_regression_thresholds:
    mswd_threshold_1 = stats.mswd_conf_limits(df, p=cfg.mswd_ci_thresholds[0])
    mswd_threshold_2 = stats.mswd_conf_limits(df, p=cfg.mswd_ci_thresholds[1])

    # equivalent p thresholds -
    p_threshold_1 = 1. - cfg.mswd_ci_thresholds[0]
    p_threshold_2 = 1. - cfg.mswd_ci_thresholds[1]

    if model == 'c1' or (model == 'ca' and mswd < mswd_threshold_2):
        if mswd < mswd_threshold_1:
            t_crit = 1.96
            theta_95ci = t_crit * np.sqrt(np.diag(covtheta))
            fitted_model = 'model 1'
            excess_scatter = False
        else:
            t_crit = stats.t_critical(df)
            covtheta *= mswd
            theta_95ci = t_crit * np.sqrt(np.diag(covtheta))
            fitted_model = 'model 1x'
            excess_scatter = True

    elif model == 'c2' or (model == 'ca' and not isochron):
        m2 = york(x, sx, y, sy, r_xy, model='2')
        theta, covtheta, mswd2, xbar, ybar, r = m2
        t_crit = stats.t_critical(df)
        covtheta *= mswd2
        theta_95ci = t_crit * np.sqrt(np.diag(covtheta))
        fitted_model = 'model 2'
        excess_scatter = True

    else:
        if mswd <= 1.:
            raise RuntimeError('cannot fit model 3 if mswd of York fit is less than 1')
        t_crit = stats.t_critical(df)
        sy_excess0 = np.sqrt(mswd) * np.sqrt(np.diag(covtheta))[0]  # use including scattter error
        fit_results = model_3(x, sx, y, sy, r_xy, sy_excess0, theta0=theta)
        (theta, covtheta, mswd3, xbar, ybar, r, sy_excess) = fit_results
        fitted_model = 'model 3'
        excess_scatter = True
        theta_95ci = t_crit * np.sqrt(np.diag(covtheta))

    fit = {
        'type': 'classical',
        'model': fitted_model,
        'excess_scatter': excess_scatter,
        'n': n,
        'sy_excess_1s': sy_excess,
        'theta': theta,
        'covtheta': covtheta,
        'theta_95ci': theta_95ci,
        't_crit': t_crit,
        'r_ab': covtheta[1, 0] / np.prod(np.sqrt(np.diag(covtheta))),
        'wtd_residuals': r,
        'mswd':  mswd,
        'mswd_regression_thresholds': [mswd_threshold_1, mswd_threshold_2],
        'p': p,
        'p_thresholds': [p_threshold_1, p_threshold_2],
        'xbar': xbar,
        'ybar': ybar
    }

    if plot:
        fig = plotting.plot_dp(x, sx, y, sy, r_xy, labels=dp_labels)
        plotting.apply_plot_settings(fig, plot_type='isochron', diagram=diagram,
                xlim=xlim, ylim=ylim, axis_labels=axis_labels,
                norm_isotope=norm_isotope)
        plotting.plot_rfit(fig.axes[0], fit)
        fit['fig'] = fig

    return fit


def robust_fit(x, sx, y, sy, r_xy, model='ra', plot=False,
        diagram=None, xlim=(None, None), ylim=(None, None),
        axis_labels=(None, None), norm_isotope='204Pb', dp_labels=None):
    """
    Fit a robust linear regression to 2-dimensional data and optionally create
    a plot of the fit.

    Parameters
    -----------
    x : np.ndarray
        x values (as 1-dimensional array)
    sx : np.ndarray
        analytical uncertainty on x at :math:`1\sigma` abs.
    y : np.ndarray
        y values
    sy : np.ndaray
        analytical uncertainty on y at :math:`1\sigma` abs.
    r_xy : np.ndarray
        x-y correlation coefficient
    model: str, optional
        Regression model to fit to data.

    Notes
    -----

    model should be one of

    - **ra** fit 'best' robust model depending on whether or no spine width is less than slim
    - **rs** spine fit of Powell et al. (2020)
    - **r2** robust model 2, a robust version of the Isoplot model 2
    - **rx** spine fit with expanded errors, for use if s is a little bit over slim (experimental feature only)

    Data should be input as 1-dimensional ndarrays.

    References
    ..........
    Powell, R., Green, E.C.R., Marillo Sialer, E., Woodhead, J., 2020. Robust
    isochron calculation. Geochronology 2, 325–342.
    https://doi.org/10.5194/gchron-2-325-2020

    """
    assert model in ('ra', 'rs', 'r2', 'rx')

    n = x.size
    if not n == sx.size == y.size == sy.size == r_xy.size:
        raise ValueError('incompatible number of elements in input data arrays')

    excess_scatter = False

    # fit spine model:
    if model in ('ra', 'rs', 'rx'):
        theta, covtheta, r, code = spine(x, sx, y, sy, r_xy, h=cfg.h)
        s = stats.nmad(r)
        s_upper95 = slim(n)
        excess_scatter = True if s > s_upper95 else False
        fitted_model = 'spine'

        if model == 'rx':    # !!! experimental only !!!
            if excess_scatter:
                covtheta *= (s / s_upper95) ** 2
                fitted_model = 'spine x'

    if model == 'r2' or (model == 'ra' and excess_scatter):
        try:
            theta, covtheta, r = robust_model_2(x, y,
                                    xy_err=np.asarray([sx, sy, r_xy]))
        except exceptions.ConvergenceError:
            raise RuntimeError('robust model 2 fit failed to converge')

        fitted_model = 'robust model 2'

        # If no spine fit, then set s to nan
        if model == 'r2':
            s = np.nan
            s_upper95 = np.nan
            excess_scatter = np.nan

    theta_95ci = 1.96 * np.sqrt(np.diag(covtheta))

    fit = {
        'type': 'robust',
        'model': fitted_model,
        'excess_scatter': excess_scatter,
        'n': n,
        'theta': theta,
        'covtheta': covtheta,
        'theta_95ci': theta_95ci,
        'r_ab': covtheta[1, 0] / np.prod(np.sqrt(np.diag(covtheta))),
        't_crit': 1.96,
        'wtd_residuals': r,
        's': s,
        'slim': s_upper95
    }

    if plot:
        fig = plotting.plot_dp(x, sx, y, sy, r_xy, labels=dp_labels)
        plotting.apply_plot_settings(fig, plot_type='isochron', diagram=diagram,
                xlim=xlim, ylim=ylim, axis_labels=axis_labels,
                norm_isotope=norm_isotope)
        plotting.plot_rfit(fig.axes[0], fit)
        fit['fig'] = fig

        # May add text boxes in future...
        # ax = fig.axes[0]
        #
        # text = f"slope: -0.4735 ± 0.079 e-05 " \
        #        f"\n y-int. 0.8137 ± 0.00077" \
        #        f"\n $s$: 1.14 (1.48, $n$ = {fit['n']})"
        # at = AnchoredText(text, prop=dict(size=7), frameon=True,
        #         loc='upper right')
        # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.5")
        # at.patch.set_linewidth(0.75)
        # ax.add_artist(at)

    return fit


#=================================
# Basic line fitting functions
#=================================

def lsq(x, y):
    """
    Fit ordinary least-squares model

    Notes
    ------
    Based on code by Roger Powell.

    """
    n = x.shape[0]
    x = np.column_stack((np.ones(n), x))
    y = y.reshape(n, 1)
    inv = np.linalg.inv(x.T @ x)
    theta = inv @ x.T @ y
    e = np.dot(x, theta) - y
    e = e.flatten()
    sigfit2 = (e.T @ e) / (n - 2)
    return theta.flatten(), sigfit2 * inv


def lad(x, y):
    """
    Fit least absolute deviation model of Sadovski (1974).

    Notes
    -----
    Based on code by Roger Powell.

    References
    ----------
    Sadovski, A.N., 1974. Algorithm AS 74: L1-norm Fit of a Straight Line. Journal
    of the Royal Statistical Society. Series C (Applied Statistics) 23,
    244–248. https://doi.org/10.2307/2347013

    """
    n = x.size
    rr = 1e-8 * cfg.rng.random(n - 1) # used for naive breaking of x ties
    bi = np.empty(n)
    bi.fill(False)
    k, i, i1, i2 = 0, 0, 0, 0

    while i != -1 and k < 12:
        i2 = i1
        i1 = i
        k += 1

        o = np.delete(np.arange(n), i)
        x1 = np.delete(x - x[i], i) + rr
        y1 = np.delete(y - y[i], i)
        oo = np.argsort(y1 / x1)
        x2 = np.abs(x1[oo])
        mid = sum(x2)/2
        sx = 0
        j = 0

        while sx < mid:
            sx += x2[j]
            j += 1
            i = o[oo][j-1]
            if i == i2:
                i = -1
            elif bi[i]:
                i = -1
            else:
                bi[i] = True

    theta = np.array(((x[i1] * y[i2] - x[i2] * y[i1])/(x[i1] - x[i2]),
                     (y[i1] - y[i2])/(x[i1] - x[i2])))
    return theta


def siegel(x, y):
    """
    Median of pairwise median slopes algorithm of Siegel (1982).

    Based on code by Roger Powell.

    References
    ----------
    Siegel, A. F.: Robust regression Using repeated medians, Biometrika,
    69, 242–244, 1982.

    """
    n = x.size
    x = x + 1e-8 * cfg.rng.random(n)  # naive breaking of x ties

    med = np.empty(n)
    for i in range(n):
        pmed = np.empty(n)
        for j in range(n):
            if i is not j:
                pmed[j] = (y[j] - y[i]) / (x[j] - x[i])
        med[i] = np.median(np.delete(pmed, i))

    b = np.median(med)
    theta = np.array((np.median(y - x * b), b))
    return theta


#=================================
# Classical stats algorithms
#=================================

def york(x, sx, y, sy, r_xy, sy_excess=None, model="1", itmax=50, rtol=1e-08,
         atol=1e-08, theta0=None):
    """
    Fit total weighted least squares regression using the algorithm
    of York (2004). Analytical errors on regression
    parameters are equivalent to the Maximum Likelihood approach of
    Titterington and Halliday (1979).

    Can also fit a version of the Isoplot model 2 and model 3 (but see note)
    by adjusting th weightings. Model 2 code is based on McSaveney in Faure (1977).
    Model 3 code emulates Isoplot (Ludwig, 2012).

    Parameters
    -----------
    x : np.ndarray, 1-D
        x values
    sx : np.ndarray, 1-D
        analytical uncertainty on x at :math:`1\sigma` abs.
    y : np.ndarray, 1-D
        y values
    sy : np.ndaray, 1-D
        analytical uncertainty on y at :math:`1\sigma` abs.
    r_xy : np.ndarray, 1-D
        x-y correlation coefficient
    model : {'1', '2', '3'}, optional
        model type to fit

    Notes
    -----
    Model 3 fits involve an iterative approach that typically requires calling
    this function several times. To perform a model 3 fit, call
    :func:`pysoplot.regression.model_3` instead.

    References
    ..........
    Faure, G., 1977. Appendix 1: Fitting of isochrons for dating by the Rb-Sr
    method, in: Principles of Isotope Geology. John Wiley and Sons, pp. 1–17.

    Ludwig, K.R., 2012. Isoplot/Ex Version 3.75: A
    Geochronological Toolkit for Microsoft Excel, Special Publication 4.
    Berkeley Geochronology Center.

    Titterington, D.M., Halliday, A.N., 1979. On the fitting of parallel
    isochrons and the method of maximum likelihood. Chemical Geology 26, 183–195.

    York, D., Evensen, N.M., Martinez, M.L., De Basabe Delgado, J., 2004. Unified
    equations for the slope, intercept, and standard errors of the best
    straight line. American Journal of Physics 72, 367–375.
    https://doi.org/10.1119/1.1632486

    """
    assert model in ("1", "2", "3")
    if model == '3' and sy_excess is None:
        raise ValueError('inital sy_excess value must provided for morel 3 fit')

    n = x.shape[0]

    # centre data
    avx = np.mean(x)
    avy = np.mean(y)
    div = np.array([1 / avy, avx / avy])
    x = x / avx
    sx = sx / avx
    y = y / avy
    sy = sy / avy

    if sy_excess is not None:
        sy_excess /= avy

    # break ties
    # TODO: is this necessary?
    x += 1e-9 * cfg.rng.random(n)

    if theta0 is None:
        theta0 = siegel(x, y)
    # b0 = theta0[1]
    b0 = (theta0 * div)[1]               # equivalent slope for centred data

    if model == "1":
        # Weight each point according to analytical uncertainties.
        wx = 1. / sx ** 2
        wy = 1. / sy ** 2
        cor = r_xy
    elif model == "2":
        # Weight each point equally with zero correlation - wy is assigned in
        # main for loop.
        wx = np.ones(n)
        cor = np.zeros(n)
    else:  # Model 3
        # analytical errors only for x
        wx = 1. / sx ** 2
        # combined analytical and initial ratio errors for y
        wy = 1. / (sy ** 2 + sy_excess ** 2)
        cor = r_xy * np.sqrt(sy ** 2 / (sy ** 2 + sy_excess ** 2))

    i = 0
    while i < itmax:
        i += 1

        if model == "2":
            wy = np.ones(n) / b0 ** 2

        # Evaluate w
        alpha = np.sqrt(wx * wy)
        w = wx * wy / (wx + wy * b0 ** 2 - 2 * b0 * cor * alpha)

        # Calculate xbar, ybar, U, V and beta
        xbar = np.sum(w * x) / np.sum(w)
        ybar = np.sum(w * y) / np.sum(w)
        u = x - xbar
        v = y - ybar
        beta = w * (u / wy + b0 * v / wx - (b0 * u + v) * cor / alpha)

        # Find improved b
        b = np.sum(w * beta * v) / np.sum(w * beta * u)

        # Check for convergence.
        if np.isclose(b0, b, atol=atol, rtol=rtol):

            # goodness of fit
            a = ybar - b * xbar
            theta = np.array((a, b))
            # s = np.sum(w * (y - b * x - a) ** 2)
            r = np.sqrt(w) * (y - b * x - a)
            s = np.sum(r ** 2)
            mswd = s / (n - 2) if n > 2 else np.nan

            # get analytical errors following York et al., (2004)
            xadj = xbar + beta
            xadj_bar = np.sum(w * xadj) / np.sum(w)
            uadj = xadj - xadj_bar
            sb = np.sqrt(1. / np.sum(w * uadj ** 2))
            sa = np.sqrt(1. / np.sum(w) + xadj_bar ** 2 * sb ** 2)
            r_ab = -xadj_bar / np.sqrt(np.sum(w * xadj ** 2) / np.sum(w))
            covtheta = np.array([[sa ** 2, r_ab * sb * sa],
                                 [r_ab * sa * sb, sb ** 2]])

            # # analytical errors following Powell et al., (2002; 2020)
            # #   - matrix approach
            # X = np.array((np.ones(n), x)).T
            # covtheta = np.linalg.inv(X.T @ np.diag(w) @ X)

            # 'de-centre' results
            theta /= div
            covtheta /= np.array([[div[0] ** 2, div[0] * div[1]],
                                  [div[0] * div[1], div[1] ** 2]])
            xbar *= avx
            ybar *= avy

            return theta, covtheta, mswd, xbar, ybar, r

        # Update estimate of b0
        b0 = b

    raise exceptions.ConvergenceError("york regression routine did not "
                "converge after maximum number of iterations")


def model_2(x, y):
    """
    Fit a model 2 regression line. The slope is the geometric mean of ordinary
    least squares of y on x and the reciprocal of the ordinary least squares
    of x on y (Powell et al., 2020).

    Equivalent to Isoplot Model 2 fit which is based on the routine of
    MacSaveney in Faure (1977) and uses the main York routine with weightings
    set to: wx = 0., wy = 1 / b^2, p_xy = 0, to achieve the same result
    iteratively.

    Notes
    -----
    Use the york function for now with model set to "2" as this function does
    not yet compute covtheta.

    """
    n = x.shape[0]
    xbar = (x @ np.ones(n)) / n
    ybar = (y @ np.ones(n)) / n
    u = x - xbar
    v = y - ybar
    b = np.sqrt(np.sum(v ** 2) / np.sum(u ** 2))

    # Check whether to take positive or negative root.
    sxy = np.sum(u * v)
    if sxy < 0:     # TODO: double check this
        b *= -1.

    a = ybar - b * xbar
    theta = np.array((a, b))

    return theta, xbar, ybar


def model_3(x, sx, y, sy, r_xy, sy_excess0, theta0, mswd_tol=1e-04,
            itmax=100, york_itmax=50, york_atol=1e-08, york_rtol=1e-08):
    """
    Fit a model 3 regression line that is equivalent to Isoplot model 3 Ludwig (2012).
    Weights each point according to analytical errors, plus an extra component
    of Gaussian distributed scatter in y (applied equally to each data point).
    For each iteration this excess error in y is inflated and a new York fit
    produced, until MSWD converges to 1.

    Parameters
    ----------
    x : np.ndarray
        x values (as 1-dimensional array)
    sx : np.ndarray
        analytical uncertainty on x at :math:`1\sigma` abs.
    y : np.ndarray
        y values
    sy : np.ndaray
        analytical uncertainty on y at :math:`1\sigma` abs.
    r_xy : np.ndarray
        x-y correlation coefficient

    Notes
    ------
    In Isoplot, this is only used for classical isochron ages

    When applying this algorithm,there should be a good justification for the
    assumption of **Guassian** distributed excess scatter.

    References
    -----------
    Ludwig, K.R., 2012. Isoplot/Ex Version 3.75: A Geochronological Toolkit for
    Microsoft Excel, Special Publication 4. Berkeley Geochronology Center.

    """
    sy_excess = sy_excess0     # starting guess of excess scatter in y
    i = 0
    mswd0 = -np.inf

    # check suitable starting sy_excess (i.e. starting mswd must be > 1)
    while i < itmax:
        i += 1
        try:
            theta, covtheta, mswd, xbar, ybar, r = york(x, sx, y, sy, r_xy,
                            theta0=theta0, sy_excess=sy_excess, model="3",
                            itmax=york_itmax, rtol=york_rtol, atol=york_atol)
            if mswd > 1.:
                break

        except exceptions.ConvergenceError:
            raise exceptions.ConvergenceError(f'model 3 routine failed')

        # update guess of excess scatter in y
        sy_excess *= 0.75
        theta0 = theta

    # TODO: implement a proper MLE routine here?
    i = 0
    while i < itmax:
        i += 1
        try:
            theta, covtheta, mswd, xbar, ybar, r = york(x, sx, y, sy, r_xy,
                            theta0=theta0, sy_excess=sy_excess, model="3",
                            itmax=york_itmax, rtol=york_rtol, atol=york_atol)
            if mswd < 1.:
                raise RuntimeError(f'model 3 fit failed - MSWD is < 1 after '
                                 f'iteration {i}')

        except exceptions.ConvergenceError:
            raise exceptions.ConvergenceError(f'york failed to converge for '
                       f'iteration {i} with mswd = {mswd}')

        if abs(mswd - 1) < mswd_tol:    # check convergence
            return theta, covtheta, mswd, xbar, ybar, r, sy_excess

        # update guess of excess scatter in y
        sy_excess *= np.sqrt(mswd)
        theta0 = theta

    raise exceptions.ConvergenceError(f' 3 routine failed - mswd (= {mswd}) '
              f'did not succesfully converge to 1')


#=================================
# Robust algorithms
#=================================

def slim(n):
    """Upper bound of 95% confidence interval on s (spine width) for Spine
    linear regression algorithm. Derived from simulation of Gaussian distributed
    datasets. See Powell et al., (2020).

    References
    ----------
    Powell, R., Green, E.C.R., Marillo Sialer, E., Woodhead, J., 2020. Robust
    isochron calculation. Geochronology 2, 325–342.
    https://doi.org/10.5194/gchron-2-325-2020

    """
    return 1.92 - 0.162 * np.log(10 + n)


def spine(x, sx, y, sy, rxy, h=1.4):
    """
    Iteratively re-weighted Huber line-fitting algorithm of Powell et al.
    (2020).

    Parameters
    ----------
    x : np.ndarray
        x values (as 1-dimensional array)
    sx : np.ndarray
        analytical uncertainty on x at :math:`1\sigma` abs.
    y : np.ndarray
        y values
    sy : np.ndaray
        analytical uncertainty on y at :math:`1\sigma` abs.
    rxy : np.ndarray
        x-y correlation coefficient

    Notes
    ------
    Updated to use normalised deltheta as convergence criteria for improved
    performance where slope and y-int are of very different order of
    magnitude.

    Code by Roger Powell with style modifications by TP.

    References
    ----------
    Powell, R., Green, E.C.R., Marillo Sialer, E., Woodhead, J., 2020. Robust
    isochron calculation. Geochronology 2, 325–342.
    https://doi.org/10.5194/gchron-2-325-2020

    """
    n = x.shape[0]
    itmax = 40
    mindel = 5e-7
    mincond = 1e-12

    # centre data
    avx = np.dot(x, np.ones(n)) / n
    avy = np.dot(y, np.ones(n)) / n
    div = np.array([1 / avy, avx/avy])
    x = x / avx
    sx = sx / avx
    y = y / avy
    sy = sy / avy
    cov = sx * sy * rxy

    theta = ntheta = siegel(x, y)

    i = 0
    code = 0
    ndeltheta = (1e10, 1e10)

    while i < itmax:
        i += 1
        a, b = oldtheta = theta
        e = a + b * x - y
        sde = np.sqrt(b ** 2 * sx ** 2 - 2 * b * cov + sy ** 2)
        r = e / sde
        wh = [1 if abs(rk) < h else np.sqrt(h/abs(rk)) for rk in r] / sde

        xp = x - r * (b * sx ** 2 - cov) / sde                 # x on the line
        ypp = (y - e - r * (b * cov - sy ** 2) / sde) * wh     # W^(1/2)(y'-e): y off the line
        c = np.transpose([wh, xp * wh])                        # W^(1/2) X'
        (u, s, v) = np.linalg.svd(c, full_matrices=False)

        if mincond * s[0] > s[1]:
            raise exceptions.ConvergenceError('spine algorithm did not converge - (nearly) '
                           'singular matrix')

        theta = np.dot(np.dot(np.dot(np.transpose(v), np.diag(1/s)),
                              np.transpose(u)), ypp)
        ndeltheta = (theta - oldtheta) / ntheta

        if np.sqrt(np.dot(ndeltheta, ndeltheta)) < mindel:
            dpsi = [1 if abs(rk) < h else 0 for rk in r]
            denom = np.array([[div[0] ** 2, div[0] * div[1]],
                              [div[0] * div[1], div[1] ** 2]])
            covtheta = np.linalg.inv(c.T @ np.diag(dpsi) @ c) / denom
            return theta / div, covtheta, r, code

    # not converged
    raise exceptions.ConvergenceError(f'spine linear regression routine did not '
               f'converge after maximum number of iterations - ndeltheta = '
               f'{ndeltheta}')


def huberu(x, y, xy_err=None, h=1.4, itmax=250):
    """
    Huber line-fitter with analytical errors discarded. Used for robust model
    2 fit. Can take many iterations to converge, so set itmax quite high.

    Original code by Roger Powell with some style modifications by TP.

    Parameters
    ----------
    xy_err : np.ndarray, optional
        3 x n array-like with elements sx, sy, r_xy (i.e. analytical
        errors on x and y, and their correlation coefficient). Used
        to ensure fit error is not less than average analytical error
        following approach of Powell et al. (2002).

    Notes
    -----
    Do not call this function directly, instead call robust_model_2.

    """
    n = x.shape[0]
    mindel = 1e-7
    mincond = 1e-12

    if xy_err is not None:
        sx = xy_err[0]
        sy = xy_err[1]
        r_xy = xy_err[2]

    i = 0
    ndeltheta = (1e10, 1e10)
    theta = ntheta = siegel(x, y)

    while i < itmax:
        i += 1
        (a, b) = oldtheta = theta
        e = a + b * x - y
        scat = stats.nmad(e)

        # Check if scat is smaller than average analytical uncertainties,
        # and if so, reset average analyutical uncertainties.
        # TODO: check these calcs
        if xy_err is not None:
            se = np.sqrt(sy ** 2 + (b * sx) ** 2 - 2 * b * sx * sy * r_xy)
            averr = np.median(se)
            scat = averr if averr > scat else scat

        r = e / scat
        wh = [1 if abs(rk) < h else np.sqrt(h / abs(rk)) for rk in r] / scat
        xp = x                                           # x on the line
        ypp = y * wh                                     # W^(1/2)(y'-e): y off the line
        c = np.transpose([wh, xp * wh])                  # W^(1/2) X'
        u, s, v = np.linalg.svd(c, full_matrices=False)

        if mincond * s[0] > s[1]:
            raise exceptions.ConvergenceError('huberu algorithm failed: '
                      '(nearly) singular matrix')

        theta = v.T @ np.diag(1 / s) @ u.T @ ypp
        ndeltheta = (theta - oldtheta) / ntheta

        if np.sqrt(np.dot(ndeltheta, ndeltheta)) < mindel:
            dpsi = [1 if abs(rk) < h else 0 for rk in r]
            covtheta = np.linalg.inv(c.T @ np.diag(dpsi) @ c)
            return theta, covtheta

    raise exceptions.ConvergenceError('huberu algorithm did not converge after '
              'maximum number of iterations')


def robust_model_2(x, y, xy_err=None, h=1.4, disp=True):
    """
    Fit a robust model 2 linear regression to 2-dimensional data.

    Analytical errors discarded and takes the geometric mean slope of the spine
    y on x fit, and the reciprocal of the spine x on y fit, using data scatter
    in place of analytical weights. Similar to the Isoplot model 2 approach
    but with robust properties.

    Parameters
    -----------
    x : np.ndarray
        x values (as 1-dimensional array)
    y : np.ndarray
        y values
    xy_err : np.ndarray, optional
        3 by n array with elements sx, sy, r_xy (i.e. analytical errors on x
        and y, and their correlation coefficient)

    Notes
    -----
    If xy_err is provided, nmad of the residuals will be checked against the
    median analytical uncertainty, and if larger, this value will be used to
    weight data points instead. Following Powell et al. (2002), this is to
    guard against a situation in which the scatter on the data is smaller than
    that expected with the analytical uncertainties.

    Code by Roger Powell with style modifications by TP.

    References
    ----------
    Powell, R., Hergt, J., Woodhead, J., 2002. Improving isochron calculations
    with robust statistics and the bootstrap. Chemical Geology 191–204.

    """
    # centre data
    avx = np.mean(x)
    avy = np.mean(y)
    div = np.array([1 / avy, avx / avy])
    x = x / avx
    y = y / avy

    # y on x regression
    if xy_err is not None:
        err = [xy_err[0] / avx, xy_err[1] / avy, xy_err[2]]
    else:
        err = None

    thyx, covthetayx, = huberu(x, y, xy_err=err, h=h)
    ayx, byx = thyx

    # x on y regression
    if xy_err is not None:
        err = [xy_err[1] / avy, xy_err[0] / avx, xy_err[2]]
    else:
        err = None

    thxy, covthxy = huberu(y, x, xy_err=err, h=h)
    axy, bxy = thetaxy = -thxy[0] / thxy[1], 1 / thxy[1]
    j = -bxy * np.array([[1, axy], [0, bxy]])
    covthetaxy = j @ covthxy @ np.transpose(j)

    # take geometric mean
    if byx > 0:
        avb = np.sqrt(byx * bxy)
    else:
        avb = -np.sqrt(byx * bxy)  # in general, care with signs
    ava = (axy * byx - ayx * bxy + avb * (ayx - axy)) / (byx - bxy)
    theta = (ava, avb)

    # residuals
    r = np.nan

    # This approximation is normally pretty good...
    # covavapprox = (covthetayx + covthetaxy) / 4

    jj = np.array([
        [bxy / (bxy + avb),
        -((-axy + ayx) * bxy * (bxy + byx - 2 * avb)) / (2 * avb * (-bxy + byx) ** 2),
         byx / (byx + avb),
         ((-axy + ayx) * byx * (bxy + byx - 2 * avb)) / (2 * avb * (-bxy + byx) ** 2)],
        [0, bxy / (2 * avb), 0, byx / (2 * avb)] ])

    cov = np.zeros((4, 4))
    cov[0, :2] = covthetayx[0]
    cov[1, :2] = covthetayx[1]
    cov[2, 2:] = covthetaxy[0]
    cov[3, 2:] = covthetaxy[1]

    covtheta = jj @ cov @ jj.T

    theta /= div
    covtheta /= np.array([[div[0] ** 2, div[0] * div[1]],
                          [div[0] * div[1], div[1] ** 2]])

    return theta, covtheta, r
