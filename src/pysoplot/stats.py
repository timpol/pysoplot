"""
Basic stats functions.

References
----------
.. [Ludwig2012]
    Ludwig, K.R., 2012. Isoplot/Ex Version 3.75: A Geochronological Toolkit for
    Microsoft Excel, Special Publication 4. Berkeley Geochronology Center.
.. [Wendt1991]
    Wendt, I., Carl, C., 1991. The statistical distribution of the mean squared
    weighted deviation. Chemical Geology: Isotope Geoscience section 86,
    275–285. https://doi.org/10.1016/0168-9622(91)90010-T

"""

import numpy as np
from scipy import stats


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
    return stats.t.ppf(q, df)


def pr_fit(df, mswd):
    """
    Probability of fit following Ludwig (2012).
    """
    return 1. - stats.chi2.cdf(mswd * df, df)


def mswd_conf_limits(df, p=0.95, two_tailed=False):
    """
    Returns confidence limits on MSWD.
    """
    assert 0 < p < 1
    if two_tailed:
        p_low = (1. - p) / 2
        p_high = 1. - p_low
        lower = stats.distributions.chi2.ppf(p_low, df) / df
        upper = stats.distributions.chi2.ppf(p_high, df) / df
        return (lower, upper)
    else:
        # note: Wendt and Carl (1991) approximation is 1 + 2 * np.sqrt(2/df) ?
        upper = stats.distributions.chi2.ppf(p, df) / df
        return upper

