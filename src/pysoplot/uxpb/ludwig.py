"""
Disequilibrium Pb/U equations based on Ludwig (1977)

References
----------
.. [Ludwig1977]
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.

"""

import numpy as np

from .. import cfg
from .. import useries


exp = np.exp
log = np.log
nan = np.nan


# =================================
# Diseq Pb/U ratios
# =================================

def f(t, A, DC, BC, init=(True, True)):
    """
    206Pb*/238U ratio (where * denotes radiogenic Pb) as a function of t and
    activity ratio values following Ludwig (1977). Note there is a small typo
    in the original article.
    """
    # Get initial activity ratios if present value given.
    A48i = A[0] if init[0] else useries.ar48i(t, A[0], DC[0], DC[1])
    A08i = A[1] if init[1] else useries.ar08i(t, A[0], A[1], DC[0], DC[1], DC[2], init=True)
    
    # Components of F from 238U and each intermediate daughter
    # ignoring 210Pb
    F = exp(DC[0] * t) * (BC[0] * exp(-DC[0] * t) + BC[1] * exp(-DC[1] * t)
            + BC[2] * exp(-DC[2] * t) + BC[3] * exp(-DC[3] * t) + 1.)
    F += A48i*DC[0]/DC[1]*exp(DC[0]*t) * (BC[4] * exp(-DC[1] * t)
            + BC[5] * exp(-DC[2] * t) + BC[6] * exp(-DC[3] * t) + 1.)
    F += A08i*DC[0]/DC[2]*exp(DC[0]*t) * (BC[7] * exp(-DC[2] * t)
            + BC[8] * exp(-DC[3] * t) + 1.)
    F += A[2]*DC[0]/DC[3]*exp(DC[0]*t) * (1. - exp(-DC[3] * t))

    return F


def g(t, A15i, DC, BC):
    """
    207Pb*/235U (where * denotes radiogenic Pb) ratio as a function of t and
    activity ratio values following Ludwig (1977)
    """
    # ignoring 227Ac
    G = exp(DC[0] * t) * (BC[0] * exp(-DC[0] * t) + BC[1] * exp(-DC[1] * t) + 1.)
    G += A15i * DC[0] / DC[1] * exp(DC[0] * t) * (1. - exp(-DC[1] * t))
    return G


def f_comp(t, A, DC, BC, init=(True, True)):
    """
    Return individual components of f.
    """
    # Get initial activity ratios if present value given.
    A48i = A[0] if init[0] else useries.ar48i(t, A[0], DC[0], DC[1])
    A08i = A[1] if init[1] else useries.ar08i(t, A48i, A[1], DC[0], DC[1],
                                              DC[2], init=True)

    # Components of F from 238U and each intermediate daughter
        # ignoring 210Pb
    F1 = exp(DC[0] * t) * (BC[0] * exp(-DC[0] * t) + BC[1] * exp(-DC[1] * t)
            + BC[2] * exp(-DC[2] * t) + BC[3] * exp(-DC[3] * t) + 1.)
    F2 = A48i*DC[0]/DC[1]*exp(DC[0]*t) * (BC[4] * exp(-DC[1] * t)
            + BC[5] * exp(-DC[2] * t) + BC[6] * exp(-DC[3] * t) + 1.)
    F3 = A08i*DC[0]/DC[2]*exp(DC[0]*t) * (BC[7] * exp(-DC[2] * t)
            + BC[8] * exp(-DC[3] * t) + 1.)
    F4 = A[2]*DC[0]/DC[3]*exp(DC[0]*t) * (1. - exp(-DC[3] * t))

    return F1, F2, F3, F4


def g_comp(t, A15i, DC, BC):
    """
    Return individual components of g.
    """
    # ignoring 227Ac
    G1 = exp(DC[0] * t) * (BC[0] * exp(-DC[0] * t) + BC[1] * exp(-DC[1] * t) + 1.)
    G2 = A15i * DC[0] / DC[1] * exp(DC[0] * t) * (1. - exp(-DC[1] * t))
    return G1, G2


# =================================
# Bateman coefficients
# =================================

def bateman(DC, series='238U'):
    """
    Bateman coefficients for the 238U or 235U decay series as defined in
    Ludwig (1977) and elsewhere.

    DC : array-like
        array of relevant decay constants (see below)

    """
    assert series in ('238U', '235U')

    if series == '238U':

        # DC[0] = lam238
        # DC[1] = lam234
        # DC[2] = lam230
        # DC[3] = lam226
        # DC[4] = lam210

        # ignoring 210Pb
        c1 = DC[1] * DC[2] * DC[3] / ((DC[0] - DC[1]) * (DC[0] - DC[2]) * (DC[0] - DC[3]))
        c2 = DC[0] * DC[2] * DC[3] / ((DC[1] - DC[0]) * (DC[1] - DC[2]) * (DC[1] - DC[3]))
        c3 = DC[0] * DC[1] * DC[3] / ((DC[2] - DC[0]) * (DC[2] - DC[1]) * (DC[2] - DC[3]))
        c4 = DC[0] * DC[1] * DC[2] / ((DC[3] - DC[0]) * (DC[3] - DC[1]) * (DC[3] - DC[2]))

        h1 = -DC[2] * DC[3] / ((DC[1] - DC[2]) * (DC[1] - DC[3]))
        h2 = -DC[1] * DC[3] / ((DC[2] - DC[1]) * (DC[2] - DC[3]))
        h3 = -DC[1] * DC[2] / ((DC[3] - DC[1]) * (DC[3] - DC[2]))

        p1 = DC[3] / (DC[2] - DC[3])
        p2 = DC[2] / (DC[3] - DC[2])
        return np.array((c1, c2, c3, c4, h1, h2, h3, p1, p2))

    else:
        # DC[0] = lam235
        # DC[1] = lam231
        # DC[2] = lam227
        
        # ignoring 227Ac
        d1 = DC[1] / (DC[0] - DC[1])
        d2 = DC[0] / (DC[1] - DC[0])
        return np.array((d1, d2))



# =================================
# Derivatives of F and G
# =================================

def dfdt(t, A, init=(True, True), comp=False):
    """
    !!! Experimental function !!!
    Derivative of f with respect to t.

    Parameters
    ----------
    comp : bool
        if True, then returns individual components.

    """
    l8, l4, l0, l6 = cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226
    c1, c2, c3, c4, h1, h2, h3, p1, p2 = bateman((l8, l4, l0, l6))

    v2 = l8/l4 * (h1*exp((l8-l4)*t) + h2*exp((l8-l0)*t) + h3*exp((l8-l6)*t)
                  + exp(l8*t))
    dv2 = l8/l4 * (h1*(l8-l4)*exp((l8-l4)*t) + h2*(l8-l0)*exp((l8-l0)*t)
                     + h3*(l8-l6)*exp((l8-l6)*t) + l8*exp(l8*t))
    v3 = l8/l0 * (p1*exp((l8-l0)*t) + p2*exp((l8-l6)*t) + exp(l8*t))
    dv3 = l8/l0 * (p1*(l8-l0)*exp((l8-l0)*t) + p2*(l8-l6)*exp((l8-l6)*t)
                     + l8*exp(l8*t))

    if init[0]:  # iniital [234U/238]
        u2 = A[0]
        du2 = 0.
    else:  # present [234U/238U]
        u2 = (A[0] - 1.) * exp(l4 * t) + 1.
        du2 = l4 * (A[0] - 1.) * exp(l4 * t)

    if init[1]:  # initial [230Th/238U]
        u3 = A[1]
        du3 = 0.
    else:
        if init[0]:  # pres. [230Th/238U], initial [234U/238U]
            u3 = (A[0] - 1.)*exp(l0*t) - (A[0] - 1.) * (l0/(l0-l4)) \
                    * (exp((l0 - l4)*t) - 1.) + 1.
            du3 = (A[0] - 1.)*l0*exp(l0*t) - (A[0] - 1.) * (l0/(l0-l4)) \
                    * (l0-l4)*exp((l0-l4)*t)
        else:  # pres. [230Th/238U], pres. [234U/238U]
            u3 = 1. - (A[0] - 1.) * (l0/(l0-l4))*(exp(l0*t) - exp(l4*t)) \
                    + (A[1] - 1.)*exp(l0*t)
            du3 = l0 * (A[0] - 1.) * exp(l0 * t) - (A[0] - 1.) * (l0/(l0-l4)) \
                    * (l0 * exp(l0*t) - l4 * exp(l4*t))

    dF1 = l8 * exp(l8 * t) * (c1 * exp(-l8 * t) + c2 * exp(-l4 * t) + c3 * exp(-l0 * t)
                + c4*exp(-l6 * t) + 1) + exp(l8 * t) * (-c1 * l8 * exp(-l8 * t)
                - c2*l4*exp(-l4 * t) - c3*l0*exp(-l0 * t) - c4*l6*exp(-l6 * t))

    dF2 = du2 * v2 + u2 * dv2
    dF3 = du3 * v3 + u3 * dv3
    dF4 = l8 / l6 * A[2] * exp(l8 * t) * (l8 * (1 - exp(-l6 * t)) + (l6 * exp(-l6 * t)))

    if comp:
        return dF1, dF2, dF3, dF4
    return dF1 + dF2 + dF3 + dF4


def dgdt(t, A15i, comp=False):
    """
    !!! Experimental function !!!
    Derivative of g with respect to t.

    Parameters
    ----------
    comp : bool
        if True, then returns individual components

    """
    l5, l1 = cfg.lam235, cfg.lam231
    d1, d2 = bateman((l5, l1), series='235U')
    dG1 = l5 * exp(l5 * t) * (d1 * exp(-l5 * t) + d2 * exp(-l1 * t) + 1.) \
          + exp(l5 * t) * (-d1 * l5 * exp(-l5 * t) - l1 * d2 * exp(-l1 * t))
    dG2 = l5 / l1 * A15i * exp(l5 * t) * (l5 * (1 - exp(-l1 * t)) + (l1 * exp(-l1 * t)))
    if comp:
        return dG1, dG2
    return dG1 + dG2
