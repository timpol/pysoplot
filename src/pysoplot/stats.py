"""
Basic stats functions.

"""

import numpy as np
from scipy.stats import norm, t, chi2


def nmad(x):
    """
    Median absolute deviation about the median (MAD) normalised such that
    statistic will be equal to the standard deviation for a standard normal
    random variable.
    """
    return 1.4826 * np.median(np.absolute(x - np.median(x)))


def t_critical(df, alpha=0.05):
    """
    Critical Student's t value for df degrees of freedom.
    """
    q = 1. - alpha / 2
    return t.ppf(q, df)


def two_sample_p(x, ox, y, oy):
    """
    Return p value for test that two measurements with **Gaussian** distributed
    uncertainties and known sigma are the same. I.e. is x - y compatible with
    0 given uncertainties?
    """
    d = abs(x - y)
    sd = np.sqrt(ox ** 2 + oy ** 2)
    p = 1. - norm.cdf(d, loc=0., scale=sd)
    return p


def pr_fit(df, mswd):
    """
    Compute probability of fit following Ludwig (2012).

    References
    ----------
    Ludwig, K.R., 2012. Isoplot/Ex Version 3.75: A Geochronological Toolkit for
    Microsoft Excel, Special Publication 4. Berkeley Geochronology Center.

    """
    return 1. - chi2.cdf(mswd * df, df)


def mswd_conf_limits(df, p=0.95, two_tailed=False):
    """
    Returns confidence limits on MSWD for given df.

    """
    assert 0 < p < 1
    if two_tailed:
        p_low = (1. - p) / 2
        p_high = 1. - p_low
        lower = chi2.ppf(p_low, df) / df
        upper = chi2.ppf(p_high, df) / df
        return (lower, upper)
    else:
        # note: Wendt and Carl (1991) approximation is 1 + 2 * np.sqrt(2/df) ?
        upper = chi2.ppf(p, df) / df
        return upper

