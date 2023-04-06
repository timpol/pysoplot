"""
Disequilibrium U-Pb equations based on Ludwig (1977).

"""

import numpy as np

from pysoplot import cfg
from pysoplot import useries


exp = np.exp
log = np.log
nan = np.nan


def f(t, A, meas=(False, False), Lam=None, coef=None, comp=False):
    """
    206Pb*/238U ratio (where * denotes radiogenic Pb) as a function of t and
    activity ratio values following Ludwig (1977). Note there is a small typo
    in the original Ludwig article.

    Parameters
    ----------
    t : float or array-like
        Age (Ma)
    A : array-like
        One-dimensional array of activity ratio values arranged as follows
        [234U/238U], [230Th/238U], [226Ra/238U],
    meas : array-like, optional
        Two element array of bools. First element is True if [234U/238U] is
        a present-day value. Second element is True if [230Th/238U] is a present-
        day value.
    Lam : array-like, optional
        Decay constants ordered as [lam238, lam234, lam230, lam226].
    coef : array-like, optional
        Bateman coefficients correspaning to decay constants in Lam.
    comp : bool, optional
        Return 206Pb*/238U ratio as array of components.

    References
    -----------
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.
       
    """
    if Lam is None:
        Lam = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    if coef is None:
        coef = bateman(Lam)

    # Get initial activity ratios if present value given.
    a234_238_i = A[0] if not meas[0] else useries.aratio48i(t, A[0], Lam=Lam)
    a230_238_i = A[1] if not meas[1] else useries.aratio08i(t, A[0], A[1], Lam=Lam, init=True)
    
    # Components of F from 238U and each intermediate daughter
    # ignoring 210Pb
    F1 = exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) + coef[1] * exp(-Lam[1] * t)
            + coef[2] * exp(-Lam[2] * t) + coef[3] * exp(-Lam[3] * t) + 1.)
    F2 = a234_238_i*Lam[0]/Lam[1]*exp(Lam[0]*t) * (coef[4] * exp(-Lam[1] * t)
            + coef[5] * exp(-Lam[2] * t) + coef[6] * exp(-Lam[3] * t) + 1.)
    F3 = a230_238_i*Lam[0]/Lam[2]*exp(Lam[0]*t) * (coef[7] * exp(-Lam[2] * t)
            + coef[8] * exp(-Lam[3] * t) + 1.)
    F4 = A[2]*Lam[0]/Lam[3]*exp(Lam[0]*t) * (1. - exp(-Lam[3] * t))

    if comp:
        return F1, F2, F3, F4

    return F1 + F2 + F3 + F4


def g(t, a231_235_i, Lam=None, coef=None, comp=False):
    """
    207Pb*/235U  ratio (where * denotes radiogenic Pb) as a function of t and
    activity ratio values following Ludwig (1977).

    Parameters
    ----------
    t : float or array-like
        Age (Ma)
    a231_235_i : float or array-like
        [231Pa/235U] activity ratio.
    Lam : array-like, optional
        One-dimensional array of decay constants (Ma^-1) ordered as
        [lam235, lam231]
    coef : array-like, optional
        One-dimensional array of Bateman coefficients corresponding to decay
        constants in Lam.
    comp : bool, optional
        Return 206Pb*/238U ratio as array of components.

    References
    -----------
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.

    """
    if Lam is None:
        Lam = (cfg.lam235, cfg.lam231)
    if coef is None:
        coef = bateman(Lam, series='235U')

    # ignoring 227Ac
    G1 = exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) + coef[1] * exp(-Lam[1] * t) + 1.)
    G2 = a231_235_i * Lam[0] / Lam[1] * exp(Lam[0] * t) * (1. - exp(-Lam[1] * t))

    if comp:
        return G1, G2

    return G1 + G2


def bateman(Lam, series='238U'):
    """
    Return Bateman coefficients for the 238U or 235U decay series as defined in
    Ludwig (1977).

    Parameters
    -----------
    Lam : array-like
        array of relevant decay constants (see below)
    series : {'238U', '235U'}, optional
        Uranium series to compute coefficients for.

    References
    -----------
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.

    """
    assert series in ('238U', '235U')

    if series == '238U':

        # Lam[0] = lam238
        # Lam[1] = lam234
        # Lam[2] = lam230
        # Lam[3] = lam226
        # Lam[4] = lam210 <- not currently used

        # ignoring 210Pb
        c1 = -Lam[1]*Lam[2]*Lam[3] / ((Lam[1]-Lam[0]) * (Lam[2]-Lam[0]) * (Lam[3]-Lam[0]))
        c2 = -Lam[0]*Lam[2]*Lam[3] / ((Lam[0]-Lam[1]) * (Lam[2]-Lam[1]) * (Lam[3]-Lam[1]))
        c3 = -Lam[0]*Lam[1]*Lam[3] / ((Lam[0]-Lam[2]) * (Lam[1]-Lam[2]) * (Lam[3]-Lam[2]))
        c4 = -Lam[0]*Lam[1]*Lam[2] / ((Lam[0]-Lam[3]) * (Lam[1]-Lam[3]) * (Lam[2]-Lam[3]))

        h1 = -Lam[2]*Lam[3] / ((Lam[2]-Lam[1]) * (Lam[3]-Lam[1]))
        h2 = -Lam[1]*Lam[3] / ((Lam[1]-Lam[2]) * (Lam[3]-Lam[2]))
        h3 = -Lam[1]*Lam[2] / ((Lam[1]-Lam[3]) * (Lam[2]-Lam[3]))

        p1 = -Lam[3] / (Lam[3]-Lam[2])
        p2 = -Lam[2] / (Lam[2]-Lam[3])
        
        return np.array((c1, c2, c3, c4, h1, h2, h3, p1, p2))

    else:
        # Lam[0] = lam235
        # Lam[1] = lam231
        # Lam[2] = lam227 <- not currently used
        
        # ignoring 227Ac
        d1 = -Lam[1] / (Lam[1]-Lam[0])
        d2 = -Lam[0] / (Lam[0]-Lam[1])

        return np.array((d1, d2))


# =================================
# Derivatives of F and G
# =================================


def dfdt(t, A, meas=(False, False), Lam=None, coef=None, comp=False):
    """
    Derivative of 206Pb*/238U ratio (where * denotes radiogenic Pb) with 
    respect to t.

    Notes
    ------
    These equations are based on secular equilibrium U-series equations only at
    present.

    """

    if Lam is None:
        Lam = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    if coef is None:
        coef = bateman(Lam)

    dF1 = Lam[0] * exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) 
            + coef[1] * exp(-Lam[1] * t) + coef[2] * exp(-Lam[2] * t) 
            + coef[3] * exp(-Lam[3] * t) + 1) + exp(Lam[0] * t) \
          * (-coef[0] * Lam[0] * exp(-Lam[0] * t) 
             - coef[1] * Lam[1] * exp(-Lam[1] * t) 
             - coef[2] * Lam[2] * exp(-Lam[2] * t)
            - coef[3] * Lam[3] * exp(-Lam[3] * t))

    # initial [234U/238]
    if not meas[0]:
        dF2 = A[0] * Lam[0] / Lam[1] * (
                coef[4] * (Lam[0] - Lam[1]) * exp((Lam[0] - Lam[1]) * t) 
                + coef[5] * (Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
                + coef[6] * (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t) 
                + Lam[0] * exp(Lam[0] * t))
    
    # present [234U/238U]
    else:
        dF2 = Lam[1] * (A[0] - 1.) * exp(Lam[1] * t) * (
                Lam[0] / Lam[1] * (coef[4] * exp((Lam[0] - Lam[1]) * t)
                + coef[5] * exp((Lam[0] - Lam[2]) * t) 
                + coef[6] * exp((Lam[0] - Lam[3]) * t) + exp(Lam[0] * t))) \
              + ((A[0] - 1.) * exp(Lam[1] * t) + 1.) * (
                Lam[0] / Lam[1] * (
                    coef[4] * (Lam[0] - Lam[1]) * exp((Lam[0] - Lam[1]) * t)
                    + coef[5] * (Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
                    + coef[6] * (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t)
                    + Lam[0] * exp(Lam[0] * t)))

    # initial [230Th/238U]
    if not meas[1]:
        dF3 = A[1] * Lam[0] / Lam[2] * (coef[7] * (
                Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
                + coef[8] * (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t)
                + Lam[0] * exp(Lam[0] * t))
    # present [230Th/238U]
    else:
        dF3 = ((A[0] - 1.) * Lam[2] * exp(Lam[2] * t) - (A[0] - 1.) * (
                Lam[2] / (Lam[2] - Lam[1])) * (Lam[2] - Lam[1]) 
                * exp((Lam[2] - Lam[1]) * t)) \
                * Lam[0] / Lam[2] * (coef[7] * exp((Lam[0] - Lam[2]) * t)
                    + coef[8] * exp((Lam[0] - Lam[3]) * t) + exp(Lam[0] * t)) \
               + ((A[0] - 1.) * exp(Lam[2] * t) - (A[0] - 1.) 
                  * (Lam[2] / (Lam[2] - Lam[1])) * (exp((Lam[2] - Lam[1]) * t) 
                    - 1.) + 1.) * Lam[0] / Lam[2] \
                  * (coef[7] * (Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
                     + coef[8] * (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t) 
                     + Lam[0] * exp(Lam[0] * t))

    dF4 = A[2]* Lam[0]/Lam[3] * (Lam[0] * exp(Lam[0]*t)
                - (Lam[0]-Lam[3]) * exp((Lam[0]-Lam[3]) * t))

    if comp:
       return dF1, dF2, dF3, dF4

    return dF1 + dF2 + dF3 + dF4


def dgdt(t, a231_235_i, Lam=None, coef=None, comp=False):
    """
    Derivative of 207Pb*/235U (where * denotes radiogenic Pb) with respect to
    t.
    """
    if Lam is None:
        Lam = (cfg.lam235, cfg.lam231)
    if coef is None:
        coef = bateman(Lam, series='235U')

    dG1 = Lam[0] * exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t)
            + coef[1] * exp(-Lam[1] * t) + 1.) + exp(Lam[0] * t) * (
            -coef[0] * Lam[0] * exp(-Lam[0] * t)
            - Lam[1] * coef[1] * exp(-Lam[1] * t))
    dG2 = a231_235_i * Lam[0] / Lam[1] * (Lam[0] * exp(Lam[0] * t)
            - (Lam[0]-Lam[1]) * exp((Lam[0]-Lam[1]) * t))

    if comp:
        return dG1, dG2

    return dG1 + dG2


