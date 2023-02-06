"""
Weighted average algorithms for 1- and 2-dimensional data

"""

import warnings
import numpy as np

from scipy.linalg import sqrtm


from . import cfg
from . import exceptions
from . import stats
from . import misc

"""
Simulated upper 95% confidence limit values on spine width for robust
spine weighted average. Note, MAD has an odd/even number bias for small n
(e.g. Hayes, 2014). Values greater than 30 are estimated based on curve fitting.

Hayes, K., 2014. "Finite-Sample Bias-Correction Factors for the Median
Absolute Deviation." Communications in Statistics - Simulation and
Computation 43, 2205–2212. https://doi.org/10.1080/03610918.2012.748913
"""
slim_even = {
    '4': 1.520,
    '6': 1.572,
    '8': 1.549,
    '10': 1.516,
    '12': 1.485,
    '14': 1.459,
    '16': 1.434,
    '18': 1.415,
    '20': 1.396,
    '22': 1.381,
    '24': 1.367,
    '26': 1.354,
    '28': 1.342,
    '30': 1.332,
}

slim_odd = {
    '3': 1.771,
    '5': 1.723,
    '7': 1.647,
    '9': 1.586,
    '11': 1.538,
    '13': 1.500,
    '15': 1.469,
    '17': 1.443,
    '19': 1.421,
    '21': 1.401,
    '23': 1.384,
    '25': 1.370,
    '27': 1.356,
    '29': 1.345,
}


#===========================================
# 1-D weighted average routines
#===========================================

def classical_wav(x, sx=None, V=None, method='ca'):
    """
    
    Compute a classical 1-dimensional weighted average accounting for assigned
    uncertainties (and optionally uncertainty correlations).

    Partly emulates the behaviour of Isoplot Ludwig (2012). If one-sided MSWD
    confidence limit is above lower threshold, uncertainty on weighted average is expanded
    according to data scatter (i.e., by sqrt(mswd)).

    Parameters
    -----------
    sx : np.ndarray, optional
        Analytical uncertainties at the 1 sigma level.
    V : np.ndarray, optional
        Uncertainty covariance matrix.
    method: {'ca'}
        Weighted average method to use (only one available currently)

    Notes
    -----
    Does not yet include external error component if MSWD confidence limit is
    above an upper threshold (as in the Isoplot MLE approach - e.g., Squid 2
    manual), although this feature may be added in future.

    References
    ..........
    Ludwig, K.R., 2012. Isoplot/Ex Version 3.75: A Geochronological Toolkit for
    Microsoft Excel, Special Publication 4. Berkeley Geochronology Center.

    """
    assert method in ('ca')                  # only one method at present
    assert (sx is not None) ^ (V is not None)
    assert cfg.mswd_ci_thresholds[1] >= cfg.mswd_ci_thresholds[0]

    n = x.shape[0]
    assert n > 1
    if V is None:
        if x.shape[0] != sx.shape[0]:
            raise ValueError('shape mismatch between inputs sx and x')
        xbar, sxbar, mswd, r = wav(x, sx)
    else:
        if V.shape != (n, n):
            raise ValueError('shape mismatch between inputs V and x')
        xbar, sxbar, mswd, r = wav_cor(x, V)

    # get pr fit:
    df = n - 1
    p = stats.pr_fit(df, mswd)

    # get mswd_regression_thresholds:
    mswd_threshold_1 = stats.mswd_conf_limits(df, p=cfg.mswd_wav_ci_thresholds[0])
    mswd_threshold_2 = stats.mswd_conf_limits(df, p=cfg.mswd_wav_ci_thresholds[1])

    # equivalent p thresholds -
    p_threshold_1 = 1. - cfg.mswd_wav_ci_thresholds[0]
    p_threshold_2 = 1. - cfg.mswd_wav_ci_thresholds[1]

    # If p is below limit, then expand errors to account for excess scatter.
    excess_scatter = True if mswd > mswd_threshold_1 else False
    t_crit = stats.t_critical(df) if excess_scatter else 1.96
    scatter_mult = np.sqrt(mswd) if excess_scatter else 1.

    results = {
        'type': 'classical',
        'n': n,
        'cov': True if V is not None else False,
        'ave': xbar,
        'ave_1s': sxbar,
        'ave_95pm': t_crit * scatter_mult * sxbar,
        'mswd':  mswd,
        'mswd_regression_thresholds': [mswd_threshold_1, mswd_threshold_2],
        'p': p,
        'p_thresholds': [p_threshold_1, p_threshold_2],
        't_crit': t_crit,
        'excess_scatter': excess_scatter,
        'wtd_residuals': r
    }
    return results


def robust_wav(x, sx=None, V=None, method='ra'):
    """

    Compute a robust 1-dimensional weighted average accounting for
    assigned uncertainties (and optionally uncertainty correlations).

    Parameters
    -----------
    x : np.ndarray
        Data points to average (as 1-d array).
    sx : np.ndarray, optional
        Analytical uncertainties at the 1 sigma level (as 1-d array).
    V : np.ndarray, optional
        Uncertainty covariance matrix.
    method: {'ra'}
        Weighted average method to use (only one available currently)

    Notes
    -----
    Only implements spine robust weighted average at present.

    """
    assert method in ('rs', 'ra')
    assert (sx is not None) ^ (V is not None)

    n = x.size
    assert n > 1
    if n < 3:
        raise ValueError('n must be greater than or equal to 3')

    model = 'spine'

    if V is None:
        if x.shape[0] != sx.shape[0]:
            raise ValueError('shape mismatch between inputs V and x')
        xbar, sxbar, s, r = spine_wav(x, sx, h=cfg.h)
    else:
        if V.shape != (n, n):
            raise ValueError('shape mismatch between inputs V and x')
        xbar, sxbar, s, r = spine_wav_cor(x, V, h=cfg.h)
    slim = slim_wav(n)

    # Check for excess scatter (note definition of excess scatter here
    # different from classical stats weighted average):
    if s > slim:
        excess_scatter = True
    else:
        excess_scatter = False

    results = {
        'type': 'robust',
        'model': model,
        'n': n,
        'cov': True if V is not None else False,
        'ave': xbar,
        'ave_1s': sxbar,
        'ave_95pm': 1.96 * sxbar,
        's': s,
        'slim': slim,
        'excess_scatter': excess_scatter,
        'wtd_residuals': r,
    }
    return results


#===========================================
# 1-D classical weighted average functions
#===========================================

def wav(x, sx):
    """
    Compute a 1-dimensional uncertainty weighted mean **without** accounting for
    uncertainty covariances.

    """
    w = 1. / sx ** 2
    xb = np.sum(w * x) / np.sum(w)                # weighted average x
    sxb = 1. / np.sqrt(np.sum(w))                 # std error on weighted average

    e = x - xb
    r = e / sx                                    # weighted residuals
    s = np.sum(r ** 2)
    df = x.shape[0] - 1
    mswd = s / df

    return xb, sxb, mswd, r


def wavx():
    """
    Classical weighted average with excess scatter parameterised as a uniform
    Gaussian source of error applied equally to all datapoints.
    """
    return


def wav_cor(x, V):
    """
    Compute a 1-dimensional uncertainty weighted mean, accounting for uncertainty
    correlations amongst individual data points.

    Based on equations given in, e.g., Powell1 and Holland (1988],
    Lyons et al. (1988), and McLean et al. (2011).

    Parameters
    ----------
    x : np.ndarray
        Data points to be averaged (should be a 1-dimensional array).
    V : np.ndarray
        covariance matrix (should be a 2-dimensional array).

    Notes
    ------
    See cautionary note in Lyons et al. (1988):

        "One point to beware of in practice is
        the situation in which the individual weights become numerically very
        large." ... this happens when [errors are very close to each other and
        highly correlated]. If we have slightly mis-estimated the elements of
        the error matrix, or if our measurements are slightly biassed, the effect
        of the large weights with different signs can be to drive our solution
        far away from the correct value."

    References
    ----------
    Lyons, L., Gibaut, D., Clifford, P., 1988. How to combine correlated
    estimates of a single physical quantity. Nuclear Instruments and Methods in
    Physics Research Section A: Accelerators, Spectrometers, Detectors and
    Associated Equipment 270, 110–117.

    McLean, N M, J F Bowring, and S A Bowring, 2011, An Algorithm for U-Pb Isotope
    Dilution Data Reduction and Uncertainty Propagation Geochemistry,
    Geophysics, Geosystems 12, no. 6.
    https://doi.org/10.1029/2010GC003478.

    Powell, R., Holland, T.J.B., 1988. An internally consistent dataset with
    uncertainties and correlations: 3. Applications to geobarometry, worked
    examples and a computer program. J Metamorph Geol 6, 173–204.
    https://doi.org/10.1111/j.1525-1314.1988.tb00415.x

    """
    if not misc.pos_def(V):
        raise ValueError('V must be a positive definite matrix')

    n = x.shape[0]
    avx = np.mean(x)
    x = x / avx
    V = V / avx ** 2

    x = x.reshape(n, 1)                # reshape to column vector
    ones = np.ones(x.shape)
    V_inv = np.linalg.inv(V)

    denom = ones.T @ V_inv @ ones
    alpha = V_inv @ ones / denom
    xb = float(alpha.T @ x)                     # wtd. average
    sxb = float(np.sqrt(1. / denom))            # standard error on wtd. average

    # goodness of fit
    e = x - xb
    s = e.T @ np.linalg.inv(V) @ e
    df = n - 1.
    mswd = float(s / df)

    # weighted residuals
    V_inv2 = sqrtm(V_inv)                           # V^(-1.2) - analytical weights
    r = V_inv2 @ e
    assert np.isclose(sum(r ** 2) / df, mswd)       # sanity check

    x *= avx
    xb *= avx
    sxb *= avx

    return xb, sxb, mswd, r.flatten()


#==============================
# 1-D robust weighted average
#==============================

def slim_wav(n):
    """
    Returns Upper 95% confidence limit on spine width (s) for weighted averages.
    Values are derived via simulation of Gaussian distributed datasets.
    """
    if n < 4:
        return np.nan

    elif n <= 30:                           # lookup value in table
        if n % 2 == 0:
            return slim_even[str(n)]
        else:
            return slim_odd[str(n)]
    else:                                   # use curve fit estimation
        if n > 150:
            warnings.warn('95% confidence limit on s may be underestimated for '
                          'n > 150')
        if n % 2 == 0:
            return 1.52825321 - 0.07689812 * np.log(-17.63329986 + n)
        else:
            return 1.53489602 - 0.07829274 * np.log(-18.25797641 + n)


def spine_wav(x, sx, xb0=None, maxiter=50, h=1.4):
    """
    Compute a 1-dimensional robust uncertainty weighted average using the spine
    algorithm, **without** accounting for error covariance.

    """
    avx = np.mean(x)
    x = x / avx
    sx = sx / avx

    # solve for xbar using iterative re-weighting (e.g. sect. 2.8 of Maronna)
    xb0 = np.median(x) if xb0 is None else xb0
    i = 0

    while i < maxiter:
        i += 1
        e = x - xb0
        r = e / sx                              # weighted residuals
        wh = [1 if abs(rk) < h else np.sqrt(h / abs(rk))
              for rk in r.flatten()]
        w = wh / sx ** 2                        # see e.g. B12 of Powell (2020)
        xb = sum(w * x) / sum(w)

        if np.isclose(xb, xb0, atol=0, rtol=1e-10):

            dpsi = [1 if abs(rk) < h else 0 for rk in r]
            var = 1. / sum(dpsi / sx ** 2)
            sxb = np.sqrt(var)  # standard error on wtd average
            s = stats.nmad(r)  # "spine width"

            xb *= avx
            sxb *= avx

            return xb, sxb, s, r

        xb0 = xb

    raise exceptions.ConvergenceError('Spine weighted average routine failed to converge '
            'after maximum number of iterations')


def spine_wav_cor(x, V, xb0=None, maxiter=50, h=1.4, atol=1e-08, rtol=1e-08):
    """
    Compute a spine robust uncertainty weighted average accounting for
    uncertainty correlations amongst data points. Equivalent to a 1-dimensional
    version of the spine linear regression algorithm of Powell et al. (2020).
    Note, this reduces to classical statistics uncertainty weighted average for "well-behaved"
    datasets.

    Parameters
    ----------
    x : np.ndarray
        Data points to be averaged (1-dimensional array).
    V : np.ndarray
        covariance matrix (2-dimensional array).


    References
    ----------
    Powell, R., Green, E.C.R., Marillo Sialer, E., Woodhead, J., 2020. Robust
    isochron calculation. Geochronology 2, 325–342.
    https://doi.org/10.5194/gchron-2-325-2020

    """
    n = x.shape[0]

    if not misc.pos_def(V):
        raise ValueError("V must be a positive definite matrix")

    avx = np.mean(x)
    x = x / avx
    V = V / avx ** 2

    x = x.reshape(n, 1)                         # reshape to column vector
    ones = np.ones(x.shape)
    V_inv = np.linalg.inv(V)
    V_inv2 = sqrtm(V_inv)               # V^(-1/2) - analytical weights
                                        # without sorting eigenv(alues)ectors

    # Double check that eigenvalues and vectors are in correct order:
    assert np.allclose(V_inv2 @ V_inv2, V_inv)

    xb0 = np.median(x) if xb0 is None else xb0
    i = 0

    # solve for x-bar using iterative re-weighting (e.g. sect. 2.8 of Maronna)
    while i < maxiter:
        i += 1
        e = x - xb0
        r = V_inv2 @ e                              # uncorrelated weighted residuals
        wh = [1 if abs(rk) < h else np.sqrt(h / abs(rk))
              for rk in r.flatten()]
        w = V_inv @ np.diag(wh)                     # e.g. B12 of Powell (2020)
        xb = ones.T @ w @ x / (ones.T @ w @ ones)   # e.g. B13 of Powell (2020)

        if np.isclose(xb, xb0, atol=atol, rtol=rtol):

            dpsi = [1 if abs(rk) < h else 0 for rk in r]
            var = 1. / (ones.T @ V_inv @ np.diag(dpsi) @ ones)
            sxb = np.sqrt(var)                     # standard error on wtd average
            s = stats.nmad(r)                      # spine width

            xb = float(xb) * avx
            sxb = float(sxb) * avx
            r = r.reshape(n, )

            return xb, sxb, s, r

        xb0 = xb

    raise exceptions.ConvergenceError('Spine weighted average routine failed '
            'to converge after maximum number of iterations')


#=================================
# 2-D weighted average
#=================================

# def wav_2d():
#     """Two-dimensional weighted average based on classical stats.
#     """
#     return


# def spine_wav_2d():
#     """
#     Would this be useful?
#     It would require a robust version of MSWD equivalence, then perhaps if
#     data points are deemed equivalent, the classical stats approach could still
#     be used to assess MSWD concordance and compute age uncertainty?
#     """
#     return
