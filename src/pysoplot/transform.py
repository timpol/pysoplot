"""
Transform data points, regression fits, etc.

"""


import numpy as np

from . import cfg


def dp_errors(dp, in_error_type, append_corr=True, row_wise=True, dim=2,
              tol=1e-10):
    """
    Convert data point uncertainties to 1 :math:`\sigma` absolute.

    Parameters
    ----------
    in_error_type : str
        Error type and sigma level, use: {abs/per/rel}{1s/2s}
    append_corr : bool, optional
        Assume error correlations are zero by appending a column of zeros to the end
        of n x 4 data point array.
    row_wise : bool, optional
        True if each row is a data point and each column is a variable. False if
        vice versa.
    dim : int
        1 for univariate data (e.g. Pb*/U data), 2 for multivariate (2-D) data.

    """
    assert in_error_type in ('abs1s', 'abs2s', 'per1s', 'per2s', 'rel1s', 'rel2s')
    dp = np.atleast_2d(dp)
    if dp.ndim > 2:
        raise ValueError('dp must a 1 or 2 dimensional array')

    # TODO: test this
    nan_idx = np.argwhere(np.isnan(dp))
    if nan_idx.size > 0:
        raise ValueError("input data contains empty cells or non-allowed "
                         "values in row(s): %s" % [x // dp.ndim for x in nan_idx])

    # convert to one row per variable
    if not row_wise:
        dp = np.transpose(dp)

    npts, nvar = dp.shape  # number of vars and data points

    a = 0.01 if 'per' in in_error_type else 1.
    b = 0.5 if '2' in in_error_type else 1.

    if nvar not in (2, 4, 5):
        raise ValueError('unexptected number of columns in input data')
    if dim == 1 and nvar != 2:
        raise ValueError('incompatible number of dimensions in input data')
    if dim == 2 and nvar not in (4, 5):
        raise ValueError('incompatible number of dimensions in input data')

    x = dp[:, 0]

    # Convert errors to 1 sigma absolute
    c = x if ('per' in in_error_type or 'rel' in in_error_type) else 1.
    sx = dp[:, 1] * a * b * c

    if nvar == 2:
        return np.array((x, sx))

    if nvar in (4, 5):
        y = dp[:, 2]
        c = y if ('per' in in_error_type or 'rel' in in_error_type) else 1.
        sy = dp[:, 3] * a * b * c

    if nvar == 4 and append_corr:
        cor_xy = np.full(npts, 0.)
    elif nvar == 5:
        if dp.shape[1] == 5:
            # TODO: values within some tolerance of +/- 1 should be reset to min/max values
            if np.any(np.abs(dp[:, -1]) > 1.):
                raise ValueError('data point correlation coefficients must '
                                 'be > -1 and < 1')
        cor_xy = dp[:, 4]

    return np.array((x, sx, y, sy, cor_xy))


def transform_fit(fit, transform_to='wc'):
    """
    Transform regression theta and covtheta from Tera-Wasserburg
    coordinates to Wetheril coordinates (or vice versa)
    in one go.

    Parameters
    ----------
    fit : dict
        Regression fit parameters.
    transform_to : {'wc', 'tw'}, optional
        Diagram coordinates to output.

    """
    theta = transform_theta(fit['theta'], transform_to=transform_to)
    covtheta = transform_covtheta(fit['theta'], fit['covtheta'],
                   transform_to=transform_to)
    return theta, covtheta


def transform_theta(theta, transform_to='wc'):
    """
    Transform theta from Tera-Wasserburg (tw) coordinates to Wetheril
    (conventional concordia, wc) coordinates or vice versa.

    Parameters
    ----------
    theta : array-like, 1-D
        Linear regression y-intercept and slope values.
    transform_to : {'wc', 'tw'}, optional
        Diagram coordinates to output.

    """
    # WC int / slope are denoted A / B.
    # T-W int / slope are denoted a / b.

    assert transform_to in ('wc', 'tw')
    if transform_to == 'wc':
        a, b = theta
        A = -b / a
        B = 1. / (cfg.U * a)
        return np.array((A, B))
    elif transform_to == 'tw':
        A, B = theta
        a = 1. / (cfg.U * B)
        b = - A / (cfg.U * B)
        return np.array((a, b))


def transform_covtheta(theta0, covtheta0, transform_to='wc'):
    """
    Transform regression fit covtheta from Tera-Wasserburg coordinates to
    Wetheril (conventional discordia) coordinates or vice versa.
    See, e.g., Ludwig (2000).

    Parameters
    ----------
    theta0: array-like, 1-D
        Linear regression y-intercept and slope parameters.
    covtheta0 : np.ndarray, 2 x 2
        Covariance matrix of regression fit parameters.
    transform_to : {'wc', 'tw'}, optional
        Diagram coordinates to output.

    References
    ----------
    Ludwig, K.R., 2000. Decay constant errors in U–Pb concordia-intercept ages.
    Chemical Geology 166, 315–318.
    https://doi.org/10.1016/S0009-2541(99)00219-3

    """
    assert transform_to in ('tw', 'wc')
    a, b = theta0
    A, B = transform_theta(theta0, transform_to=transform_to)
    if transform_to == 'wc':
        jac = np.array([[b / a ** 2, -1. / a], [1. / (cfg.U * a ** 2), 0. ]])
    else:
        raise ValueError('not yet coded')
    covtheta = jac @ covtheta0 @ jac.T
    #TODO: there is an error here causing incorrect sign for covariance!?!
    covtheta[1, 0] *= -1.
    covtheta[0, 1] *= -1.
    return covtheta


def transform_centroid(xbar, ybar, transform_to='wc'):
    """
    Transform x-bar and y-bar for classical regression fit from Terra-Wasserburg
    to Wetheril or vice versa.
    """
    # WC variables are denoted X, Y
    # T-W variables are denoted x, y

    assert transform_to in ('wc', 'tw')
    if transform_to == 'wc':
        Xbar = ybar * cfg.U / xbar
        Ybar = 1. / xbar
        return Xbar, Ybar
    elif transform_to == 'tw':
        Xbar, Ybar = xbar, ybar
        xbar = 1 / Ybar
        return Ybar * Xbar / cfg.U


def transform_dp(x0, ox0, y0, oy0, r_xy0, to='wc'):
    """
    Transform concordia diagram data points from Tera-Wasserburg to Wetheril
    concordia or vice versa.

    Notes
    -----
    E.g., see Ludwig (1998) and Noda (2017).

    References
    ----------
    Ludwig, K.R., 1998. On the treatment of concordant uranium-lead ages.
    Geochimica et Cosmochimica Acta 62, 665–676.
    https://doi.org/10.1016/S0016-7037(98)00059-3

    Noda, A., 2017. A new tool for calculation and visualization of U.
    Bulletin of the Geological Survey of Japan 1–10.
    """

    # WC --> X, Y
    # T-W --> x, y

    assert to in ('wc', 'tw')
    #TODO: use matrix multiplication instead ??
    if to == 'wc':
        x, y = x0, y0
        ox, oy, r_xy = ox0, oy0, r_xy0
        # convert to relative errors
        sx = ox / x
        sy = oy / y
        # do transformation
        Y = 1 / x
        X = cfg.U * y / x
        sX = np.sqrt(sx ** 2 + sy ** 2 - 2 * sx * sy * r_xy)
        sY = sx
        r_XY = (sx ** 2 - sx * sy * r_xy) / (sx * sy)
        # back to abs. errors
        oX = abs(sX * X)
        oY = abs(sY * Y)
        return np.array((X, oX, Y, oY, r_XY))

    else:
        raise ValueError('not yet coded')

