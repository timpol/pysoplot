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


def fit(fit, transform_to='wc'):
    """
    Transform regression parameters (y-int and slope) from Tera-Wasserburg
    coordinates to Wetheril coordinates (or vice versa)
    in one go.

    Parameters
    ----------
    fit : dict
        Regression fit parameters.
    transform_to : {'wc', 'tw'}, optional
        Diagram coordinates to output.

    """
    th = theta(fit['theta'], transform_to=transform_to)
    covth = covtheta(fit['theta'], fit['covtheta'], transform_to=transform_to)
    return th, covth


def theta(theta, transform_to='wc'):
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


def covtheta(theta0, covtheta0, transform_to='wc'):
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
    # A, B = theta(theta0, transform_to=transform_to)

    if transform_to == 'wc':
        jac = np.array([[b / a ** 2, -1. / a], [1. / (cfg.U * a ** 2), 0. ]])
    else:
        raise ValueError('not yet coded')

    V_th = jac @ covtheta0 @ jac.T
    #TODO: there is an error here causing incorrect sign for covariance!?!
    V_th[1, 0] *= -1.
    V_th[0, 1] *= -1.
    return V_th


def centroid(xbar, ybar, transform_to='wc'):
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

def concordia_dp(dp, to='tw', d=2):
    """
    Transform concordia diagram data points from Tera-Wasserburg to Wetheril
    concordia or vice versa.

    Notes
    ------
    Wetheril (conventional) coordinates are x=207/235, y=206/238, (z=204/238).
    T-W coordinates are u = 238/206, v = 207/206, (w=204/206).
    See e.g. Eq. (63) in McLean et al. (2011).
    """
    dp = np.asarray(dp)
    n = dp.shape[1]

    if d == 2:
        x, sx, y, sy, r_xy = dp
    elif d == 3:
        x, sx, y, sy, z, sz, r_xy, r_xz, r_yz = dp

    if to == 'tw':

        u = 1. / y
        v = x / (cfg.U * y)
        dudx = 0. * x                   # this is 0s vector of len(x)
        dvdx = 1. / (cfg.U * y)
        dudy = -1. / y ** 2
        dvdy = -x / (cfg.U * y ** 2)
        cov_xy = r_xy * sx * sy
        su = np.zeros(n)
        sv = np.zeros(n)
        r_uv = np.zeros(n)

        if d == 3:
            w = z / y
            dwdx = 0. * x
            dwdy = -z / y ** 2
            dudz = 0. * x              # dbl check # this is 0s vector of len(x)
            dvdz = 0. * x              # this is 0s vector of len(x)
            dwdz = 1. / y
            cov_xz = r_xz * sx * sz
            cov_yz = r_yz * sy * sz
            sw = np.zeros(n)
            r_uw = np.zeros(n)
            r_vw = np.zeros(n)

        if d == 2:
            for i in range(n):
                J = np.array([[dudx[i], dvdx[i]],
                              [dudy[i], dvdy[i]]])
                V_xy = np.array([[sx[i] ** 2, cov_xy[i]],
                              [cov_xy[i], sy[i] ** 2]])
                V_uv = J.T @ V_xy @ J
                su[i] = np.sqrt(V_uv[0, 0])
                sv[i] = np.sqrt(V_uv[1, 1])
                r_uv[i] = V_uv[1, 0] / (su[i] * sv[i])

            return u, su, v, sv, r_uv

        elif d == 3:
            for i in range(n):
                J = np.array([[dudx[i], dvdx[i], dwdx[i]],
                              [dudy[i], dvdy[i], dwdy[i]],
                              [dudz[i], dvdz[i], dwdz[i]]])
                V_xyz = np.array([[sx[i] ** 2, cov_xy[i], cov_xz[i]],
                                  [cov_xy[i], sy[i] ** 2, cov_yz[i]],
                                  [cov_xy[i], cov_yz[i], sz[i] ** 2]])
                V_uvw = J.T @ V_xyz @ J
                su[i] = np.sqrt(V_uvw[0, 0])
                sv[i] = np.sqrt(V_uvw[1, 1])
                sw[i] = np.sqrt(V_uvw[2, 2])
                r_uv[i] = V_uvw[1, 0] / (su[i] * sv[i])
                r_uw[i] = V_uvw[2, 0] / (su[i] * sw[i])
                r_vw[i] = V_uvw[2, 1] / (sv[i] * sw[i])

            return u, su, v, sv, w, sw, r_uv, r_uw, r_vw

    elif to == 'wc':

        if d == 2:
            u, su, v, sv, r_uv = dp
        elif d == 3:
            u, su, v, sv, w, sw, r_uv, r_uw, r_vw = dp

        y = 1. / u
        x = cfg.U * v / u
        dudx = 0. * x                   # this is 0s vector of len(x)
        dvdx = 1. / (cfg.U * y)
        dudy = -1. / y ** 2
        dvdy = -x / (cfg.U * y ** 2)
        cov_xy = r_uv * su * sv
        su = np.zeros(n)
        sv = np.zeros(n)
        r_uv = np.zeros(n)

        if d == 3:
            z = w / u
            dwdx = 0. * x
            dwdy = -z / y ** 2
            dudz = 0. * x              # dbl check # this is 0s vector of len(x)
            dvdz = 0. * x              # this is 0s vector of len(x)
            dwdz = 1. / y
            cov_xz = r_xz * sx * sz
            cov_yz = r_yz * sy * sz
            sw = np.zeros(n)
            r_uw = np.zeros(n)
            r_vw = np.zeros(n)

        if d == 2:
            for i in range(n):
                J = np.array([[dudx[i], dvdx[i]],
                              [dudy[i], dvdy[i]]])
                V_xy = np.array([[sx[i] ** 2, cov_xy[i]],
                              [cov_xy[i], sy[i] ** 2]])
                V_uv = J.T @ V_xy @ J
                su[i] = np.sqrt(V_uv[0, 0])
                sv[i] = np.sqrt(V_uv[1, 1])
                r_uv[i] = V_uv[1, 0] / (su[i] * sv[i])

            return u, su, v, sv, r_uv

        elif d == 3:
            for i in range(n):
                J = np.array([[dudx[i], dvdx[i], dwdx[i]],
                              [dudy[i], dvdy[i], dwdy[i]],
                              [dudz[i], dvdz[i], dwdz[i]]])
                V_xyz = np.array([[sx[i] ** 2, cov_xy[i], cov_xz[i]],
                                  [cov_xy[i], sy[i] ** 2, cov_yz[i]],
                                  [cov_xy[i], cov_yz[i], sz[i] ** 2]])
                V_uvw = J.T @ V_xyz @ J
                su[i] = np.sqrt(V_uvw[0, 0])
                sv[i] = np.sqrt(V_uvw[1, 1])
                sw[i] = np.sqrt(V_uvw[2, 2])
                r_uv[i] = V_uvw[1, 0] / (su[i] * sv[i])
                r_uw[i] = V_uvw[2, 0] / (su[i] * sw[i])
                r_vw[i] = V_uvw[2, 1] / (sv[i] * sw[i])

            return u, su, v, sv, w, sw, r_uv, r_uw, r_vw

