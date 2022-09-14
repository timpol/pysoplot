"""
Miscellaneous functions

"""

import numpy as np

#=================================
# Basic geochronology functions
#=================================

# def dc2hl():
#     """Convert decay constant to half life.
#     """
#     return
#
#
# def hl2dc():
#     """ Convert half life to decay constant.
#     """
#     return


#========
# Maths
#========

def pos_def(x, tol=1e-8):
    """
    Verify that matrix is positive semi-definite.
    Based on numpy multivariate_normal code, e.g.:
    https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx
    """
    x = np.asarray(x, dtype=np.double)
    (u, s, v) = np.linalg.svd(x)
    psd = np.allclose(np.dot(v.T * s, v), x, rtol=tol, atol=tol)
    return psd


def cdiff(x, f, h, *args, **kwargs):
    """
    Estimate derivative using central difference method
    """
    return (f(x + h, *args, **kwargs) - f(x - h, *args, **kwargs)) / (2.0 * h)


def eigsorted(cov):
    """
    Return eigenvalues and vectors sorted from largest to smallest.
    Based on code by Joe Kington:
    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def covmat_to_cormat(cov):
    """
    Compute correlation matrix from covariance matrix. Based on:
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    cor = cov / outer_v
    cor[cov == 0] = 0
    return cor


#=========================
# Number formatting etc.
#=========================

def get_exponent(n):
    """Return order of magnitude of n
    """
    if n == 0:
        return 0. * n
    return int(np.floor(np.log10(abs(n))))


def num_dec_places(n):
    """Return number of decimal digits in string representation of a decimal
    number.
    """
    s = str(n).rstrip("0")
    try:
        nd = s.split('.')[1]
    except IndexError:
        return 0
    return len(nd)


def round_down(n, sf=2):
    """ Round float down to n significant figures.
    """
    if n == 0:
        return 0. * n   # for array inputs
    return round(n, sf - int(np.floor(np.log10(abs(n)))) - 1)


#=========================
# Very basic I/O
#=========================

def print_result(dic, title='Result'):
    print(title)
    print('--------------------')
    for k, v in dic.items():
        print(f'{k}: {v}')
    print()


def print_table(cols, title='Table'):
    n = len(cols[0])
    assert [len(x) == n for x in cols]
    print(title)
    print('--------------------')
    for row in range(n):
        out = []
        for x in cols:
            out.append(f'{x[row]}')
        print(out)