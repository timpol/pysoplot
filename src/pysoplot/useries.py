"""
Uranium-series functions and routines.

References
----------
.. [Cheng2000]
    Cheng, H., Adkins, J., Edwards, R.L., Boyle, E.A., 2000. U-Th dating of
    deep-sea corals. Geochimica et Cosmochimica Acta 64, 2401–2416.
    https://doi.org/10.1016/S0016-7037(99)00422-6
.. [Hellstrom2006]
    Hellstrom, J., 2006. U–Th dating of speleothems with high initial 230Th using
    stratigraphical constraint. Quaternary Geochronology 1, 289–295.
    https://doi.org/10.1016/j.quageo.2007.01.004
.. [Ludwig1977]
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium on
    U-Pb isotope apparent ages of young minerals. Journal of Research of the US
    Geological Survey 5, 663–667.
"""

import numpy as np

from . import cfg


exp = np.exp


#=====================================
# 234U age equaions
#=====================================
def ar48i(t, A48, l8, l4):
    """
    Initial 234U/238U activity ratio from present ratio.

    Notes
    ------
    Non-secular equilibrium equation is derived following approach of Ludwig
    (1977). This is an experimental feature only!

    """
    if cfg.secular_eq:
        return 1. + (A48 - 1.) * exp(l4 * t)
    else:
        return l4/(l4-l8) + (A48 - l4/(l4-l8)) *exp(-(l8-l4)*t)


def ar48(t, A48i, l8, l4):
    """
    234U/238U activity ratio from initial ratio and time.

    Notes
    ------
    Non-secular equilibrium equation is derived following approach of Ludwig
    (1977). This is an experimental feature only!

    """
    if cfg.secular_eq:
        return (A48i - 1.) * exp(-l4 * t) + 1.
    else:
        return l4/(l4-l8) + (A48i - l4/(l4-l8)) * exp((l8-l4)*t)


#=====================================
# Th230 equations
#=====================================

def Th230_age():
    pass


def Th230_age_uncert():
    pass


def ar08i(t, A48, A08, l8, l4, l0, init=True):
    """
    Initial [230Th/238U] as a function of present-day ratio and
    time. E.g. Cheng et al. (2000; GCA) and Hellstrom (2006; Q. Geochon.).

    Parameters
    ----------
    init : bool
        If True, [234U/238] is treated as an initial value, otherwise it's
        treated as a present-day value.

    Notes
    ------
    Non-secular equilibrium equation is derived following approach of Ludwig
    (1977). This is an experimental feature only!

    """
    if cfg.secular_eq:
        if init:
            A48i = A48
            A08i = 1. - (A48i-1.) * (l0/(l0-l4)) * (exp((l0-l4) * t) - 1) \
                    + (A08-1.) * exp(l0*t)
        else:
            A08i = 1. - (A48-1.) * (l0/(l0-l4)) * (exp(l0*t) - exp(l4*t)) \
                   + (A08-1.) * exp(l0 * t)
        return A08i
    else:
        a48i = ar48i(t, A48, l8, l4) if init else A48

        c1 = l0*l4 / ((l4-l8) * (l0-l8))
        c2 = l0*l4 / ((l8-l4) * (l0-l4))
        c3 = l0*l4 / ((l8-l0) * (l4-l0))

        A08i = (A08 - (c1 + c2*exp((l8-l4)*t) + c3*exp((l8-l0)*t)
                + a48i*(l0/(l0-l4)) * (exp((l8-l4)*t) - exp((l8-l0) * t)))) \
                * exp(-(l8-l0) * t)
        return A08i


def A08(t, A48, A08i, l8, l4, l0, init=True):
    """
    Present-day [230Th/238U] activity ratio as a function of initial ratio
    and time. Based on equations in, e.g., Cheng et al. (2000; GCA) and
    Hellstrom (2006; Q. Geochon.).

    Parameters
    ----------
    init : bool
        If True, [234U/238] is treated as an initial value, otherwise it's
        treated as a present-day value.

    Notes
    ------
    Non-secular equilibrium equation is derived following approach of Ludwig
    (1977). This is an experimental feature only!

    """
    if not cfg.secular_eq:
        raise ValueError('non-secular equations not yet implemented here')

    if init:
        A48i = A48
        A08 = 1. - exp(-l0*t) * (1. - A08i) + (A48i-1.)*(l0/(l0-l4)) \
              * (exp(-l4*t) - exp(-l0*t))
    else:
        A08 = 1. - exp(-l0*t) * (1. - A08i) + (A48-1.) * (l0/(l0-l4)) \
              * (1.-exp((l4-l0)*t))
    return A08


#=====================================
# U/Th isochron age equations
#=====================================

def Th230_isochron():
    pass



#==================================================
# Back-calculate initial activity ratio solutions
#===================================================

def Ai_solutions(t, A, init, DC):
    """
    Compute initial activity ratio solutions from present-day values and U-Pb age
    solution. In principle these values are computed iteratively along with the
    age solution, but practically it is more convenient to compute them after
    the fact.
    """
    a48i = None
    a08i = None
    if not init[0]:        # present [234U/238U]
        a48i = ar48i(t, A[0], DC[0], DC[1])
        if not init[1]:    # present [234U/238U] and [230Th/238U]
            a08i = ar08i(t, ar48i, A[1], DC[0], DC[1], DC[2])
    elif not init[1]:      # only present [230Th/238U]
        a08i = ar08i(t, A[0], A[1], DC[0], DC[1], DC[2], init=init[0])
    return a48i, a08i