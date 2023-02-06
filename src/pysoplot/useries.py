"""
Uranium-series functions and routines.

"""

import numpy as np

from . import cfg


exp = np.exp


#=====================================
# 234U age equaions
#=====================================
#TODO: sec_eq should be a kwarg for these functions?, and dqpb functions should
# set this to cfg.sec_eq.

def ar48i(t, a234_238, lam238, lam234):
    """
    Calculate initial [234U/238U] activity ratio as a function of present
    ratio and age (Ma).

    Notes
    ------
    Non-secular equilibrium equation is derived from Bateman (1910) and does
    not assume negligible decay of 238U (e.g., Ivanovich and Harmon
    (1992)). This is an experimental feature.

    References
    ----------
    Ivanovich, M. and Harmon, R. S.: Uranium-Series Disequilibrium:
    Applications to Earth, Marine, and Environmental Sciences., Clarendon
    Press, United Kingdom, second edn., 1992.

    """
    if cfg.secular_eq:
        return 1. + (a234_238 - 1.) * exp(lam234 * t)
    else:
        return lam234/(lam234-lam238) + (a234_238 - lam234/(lam234-lam238)) \
               * exp(-(lam238-lam234)*t)


def ar48(t, a234_238_i, lam238, lam234):
    """
    Calculate initial [234U/238U] activity ratio as a function of initial
    ratio and age (Ma).

    Notes
    ------
    Non-secular equilibrium equation is derived from Bateman (1910) and does
    not assume negligible decay of 238U (e.g., Ivanovich and Harmon
    (1992)). This is an experimental feature.

    References
    ----------
    Ivanovich, M. and Harmon, R. S.: Uranium-Series Disequilibrium:
    Applications to Earth, Marine, and Environmental Sciences., Clarendon
    Press, United Kingdom, second edn., 1992.

    """
    if cfg.secular_eq:
        return 1. + (a234_238_i - 1.) * exp(-lam234 * t)
    else:
        return lam234/(lam234-lam238) + (a234_238_i - lam234/(lam234-lam238)) \
               * exp((lam238-lam234)*t)


#=====================================
# Th230 equations
#=====================================

def Th230_age():
    pass


def Th230_age_uncert():
    pass


def ar08i(t, a234_238, a230_238, lam238, lam234, lam230, init=True):
    """
    Calculate initial [230Th/238U] as a function of present ratio and
    time.

    Parameters
    ----------
    init : bool
        True if `a234_238` is as an initial activity ratio.

    Notes
    ------
    Secular equilibrium equations are given by, e.g., Cheng et al., (2000) and
    Hellstrom (2006).

    Non-secular equilibrium equation is derived from Bateman (1910) and does
    not assume negligible decay of 238U (e.g., Ivanovich and Harmon
    (1992)). This is an experimental feature.

    References
    ----------
    Cheng, H., Adkins, J., Edwards, R.L., Boyle, E.A., 2000.
    U-Th dating of deep-sea corals. Geochimica et Cosmochimica Acta 64, 2401–2416.
    https://doi.org/10.1016/S0016-7037(99)00422-6

    Hellstrom, J., 2006. U–Th dating of speleothems with
    high initial 230Th using stratigraphical constraint. Quaternary
    Geochronology 1, 289–295. https://doi.org/10.1016/j.quageo.2007.01.004

    """
    if cfg.secular_eq:
        if init:
            a234_238_i = a234_238
            a230_238_i = 1. - (a234_238_i - 1.) * (lam230/(lam230-lam234)) \
                   * (exp((lam230-lam234) * t) - 1) + (a230_238 - 1.) * exp(lam230*t)
        else:
            a230_238_i = 1. - (a234_238 - 1.) * (lam230/(lam230-lam234)) \
                   * (exp(lam230*t) - exp(lam234*t)) + (a230_238-1.) * exp(lam230 * t)
        return a230_238_i

    else:
        a234_238_i = ar48i(t, a234_238, lam238, lam234) if init else a234_238

        c1 = lam230*lam234 / ((lam234-lam238) * (lam230-lam238))
        c2 = lam230*lam234 / ((lam238-lam234) * (lam230-lam234))
        c3 = lam230*lam234 / ((lam238-lam230) * (lam234-lam230))

        A08i = (a230_238 - (c1 + c2*exp((lam238-lam234)*t) + c3*exp((lam238-lam230)*t)
                + a234_238_i * (lam230/(lam230-lam234)) * (exp((lam238-lam234)*t)
                   - exp((lam238-lam230) * t)))) * exp(-(lam238-lam230) * t)
        return A08i


def ar08(t, a234_238, a230_238_i, lam238, lam234, lam230, init=True):
    """
    Calculate present [230Th/238U] as a function of present ratio and
    time.

    Parameters
    ----------
    init : bool
        True if `a234_238` is as an initial activity ratio.

    Notes
    ------
    Secular equilibrium equations are given by, e.g., Cheng et al. (2000) and
    Hellstrom (2006).

    Non-secular equilibrium equation is derived from Bateman (1910) and does
    not assume negligible decay of 238U (e.g., Ivanovich and Harmon
    (1992)). This is an experimental feature.

    References
    ----------
    Cheng, H., Adkins, J., Edwards, R.L., Boyle, E.A., 2000.
    U-Th dating of deep-sea corals. Geochimica et Cosmochimica Acta 64, 2401–2416.
    https://doi.org/10.1016/S0016-7037(99)00422-6

    Hellstrom, J., 2006. U–Th dating of speleothems with
    high initial 230Th using stratigraphical constraint. Quaternary
    Geochronology 1, 289–295. https://doi.org/10.1016/j.quageo.2007.01.004

    """
    if not cfg.secular_eq:
        raise ValueError('non-secular equations not yet implemented here')

    if init:
        a234_238_i = a234_238
        a230_238 = 1. - exp(-lam230*t) * (1. - a230_238_i) + (a234_238_i - 1.) \
                    * (lam230/(lam230-lam234)) * (exp(-lam234*t) - exp(-lam230*t))
    else:
        a230_238 = 1. - exp(-lam230*t) * (1. - a230_238_i) + (a234_238-1.) \
                   * (lam230/(lam230-lam234)) * (1.-exp((lam234-lam230)*t))
    return a230_238


#=====================================
# U/Th isochron age equations
#=====================================

def Th230_isochron():
    pass



#==================================================
# Back-calculate initial activity ratio solutions
#===================================================

def init_ratio_solutions(t, A, init, Lam):
    """
    Compute initial activity ratio solutions from present-day values and U-Pb age
    solution. In principle these values are computed iteratively along with the
    age solution, but practically it is more convenient to compute them separately
    
    """
    a234_238_i = None
    a230_238_i = None
    if not init[0]:        # present [234U/238U]
        a234_238_i = ar48i(t, A[0], Lam[0], Lam[1])
        if not init[1]:    # present [234U/238U] and [230Th/238U]
            a230_238_i = ar08i(t, ar48i, A[1], Lam[0], Lam[1], Lam[2])
    elif not init[1]:      # only present [230Th/238U]
        a230_238_i = ar08i(t, A[0], A[1], Lam[0], Lam[1], Lam[2], init=init[0])
    return a234_238_i, a230_238_i