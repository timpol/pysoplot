"""
Disequilibrium U-Pb equations based on Ludwig (1977).

"""

import numpy as np

from pysoplot import useries


exp = np.exp
log = np.log
nan = np.nan


def f(t, A, Lam, coef, init=(True, True)):
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
    Lam : array-like
        One-dimensional array of decay constants (Ma^-1) arragend as follows
        [lam238, lam234, lam230, lam226]
    coef : array-like
        One-dimensional array of Bateman coefficients
        [c1, c2, c3, c4, c5, p1, p2, p3, h1, h2]
    init : array-like, optional
        Two element array of bools. First element is True if [234U/238U] is
        an initial value. Second element is True if [230Th/238U] is an initial
        value.

    References
    -----------
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.
       
    """
    # Get initial activity ratios if present value given.
    a234_238_i = A[0] if init[0] else useries.ar48i(t, A[0], Lam[0], Lam[1])
    a230_238_i = A[1] if init[1] else useries.ar08i(t, A[0], A[1], Lam[0], Lam[1], Lam[2], init=True)
    
    # Components of F from 238U and each intermediate daughter
    # ignoring 210Pb
    F = exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) + coef[1] * exp(-Lam[1] * t)
            + coef[2] * exp(-Lam[2] * t) + coef[3] * exp(-Lam[3] * t) + 1.)
    F += a234_238_i*Lam[0]/Lam[1]*exp(Lam[0]*t) * (coef[4] * exp(-Lam[1] * t)
            + coef[5] * exp(-Lam[2] * t) + coef[6] * exp(-Lam[3] * t) + 1.)
    F += a230_238_i*Lam[0]/Lam[2]*exp(Lam[0]*t) * (coef[7] * exp(-Lam[2] * t)
            + coef[8] * exp(-Lam[3] * t) + 1.)
    F += A[2]*Lam[0]/Lam[3]*exp(Lam[0]*t) * (1. - exp(-Lam[3] * t))

    return F


def g(t, a231_235_i, Lam, coef):
    """
    207Pb*/235U  ratio (where * denotes radiogenic Pb) as a function of t and
    activity ratio values following Ludwig (1977).

    Parameters
    ----------
    t : float or array-like
        Age (Ma)
    a231_235_i : float or array-like
        [231Pa/235U] activity ratio.
    Lam : array-like
        One-dimensional array of decay constants (Ma^-1) arragend as follows
        [lam235, lam231]
    coef : array-like
        One-dimensional array of Bateman coefficients
        [d1, d2]

    References
    -----------
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.

    """
    # ignoring 227Ac
    G = exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) + coef[1] * exp(-Lam[1] * t) + 1.)
    G += a231_235_i * Lam[0] / Lam[1] * exp(Lam[0] * t) * (1. - exp(-Lam[1] * t))
    return G


def f_comp(t, A, Lam, coef, init=(True, True)):
    """
    Return individual components of f.
    """
    # Get initial activity ratios if present value given.
    a234_238_i = A[0] if init[0] else useries.ar48i(t, A[0], Lam[0], Lam[1])
    a230_238_i = A[1] if init[1] else useries.ar08i(t, a234_238_i, A[1], Lam[0], Lam[1],
                                              Lam[2], init=True)

    # Components of F from 238U and each intermediate daughter
        # ignoring 210Pb
    F1 = exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) + coef[1] * exp(-Lam[1] * t)
            + coef[2] * exp(-Lam[2] * t) + coef[3] * exp(-Lam[3] * t) + 1.)
    F2 = a234_238_i*Lam[0]/Lam[1]*exp(Lam[0]*t) * (coef[4] * exp(-Lam[1] * t)
            + coef[5] * exp(-Lam[2] * t) + coef[6] * exp(-Lam[3] * t) + 1.)
    F3 = a230_238_i*Lam[0]/Lam[2]*exp(Lam[0]*t) * (coef[7] * exp(-Lam[2] * t)
            + coef[8] * exp(-Lam[3] * t) + 1.)
    F4 = A[2]*Lam[0]/Lam[3]*exp(Lam[0]*t) * (1. - exp(-Lam[3] * t))

    return F1, F2, F3, F4


def g_comp(t, a231_235_i, Lam, coef):
    """
    Return individual components of g.
    """
    # ignoring 227Ac
    G1 = exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) + coef[1] * exp(-Lam[1] * t) + 1.)
    G2 = a231_235_i * Lam[0] / Lam[1] * exp(Lam[0] * t) * (1. - exp(-Lam[1] * t))
    return G1, G2


def bateman(Lam, series='238U'):
    """
    Return Bateman coefficients for the 238U or 235U decay series as defined in
    Ludwig (1977).

    Lam : array-like
        array of relevant decay constants (see below)

    Notes
    ------


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


def dfdt(t, A, Lam, coef, init=(True, True)):
    """
    Derivative of 206Pb*/238U ratio (where * denotes radiogenic Pb) with 
    respect to t.

    Notes
    ------
    These equations are based on secular equilibrium U-series equations only at
    present.

    """
    dF1 = Lam[0] * exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t) 
            + coef[1] * exp(-Lam[1] * t) + coef[2] * exp(-Lam[2] * t) 
            + coef[3] * exp(-Lam[3] * t) + 1) + exp(Lam[0] * t) \
          * (-coef[0] * Lam[0] * exp(-Lam[0] * t) 
             - coef[1] * Lam[1] * exp(-Lam[1] * t) 
             - coef[2] * Lam[2] * exp(-Lam[2] * t)
            - coef[3] * Lam[3] * exp(-Lam[3] * t))

    # initial [234U/238]
    if init[0]:
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
    if init[1]:
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

    return dF1 + dF2 + dF3 + dF4


def dfdt_comp(t, A, Lam, coef, init=(True, True)):
    """
    Individual components of derivative of 206Pb*/238U ratio (where * denotes 
    radiogenic Pb) with respect to t.
    
    Notes
    ------
    These equations are based on secular equilibrium U-series equations only at
    present.
    
    """
    dF1 = Lam[0] * exp(Lam[0] * t) * (
            coef[0] * exp(-Lam[0] * t)
            + coef[1] * exp(-Lam[1] * t) + coef[2] * exp(-Lam[2] * t)
            + coef[3] * exp(-Lam[3] * t) + 1) + exp(Lam[0] * t) \
           * (-coef[0] * Lam[0] * exp(-Lam[0] * t)
              - coef[1] * Lam[1] * exp(-Lam[1] * t)
              - coef[2] * Lam[2] * exp(-Lam[2] * t)
              - coef[3] * Lam[3] * exp(-Lam[3] * t))

    # initial [234U/238]
    if init[0]:
        dF2 = A[0] * Lam[0] / Lam[1] * (
                coef[4] * (Lam[0] - Lam[1]) * exp((Lam[0] - Lam[1]) * t)
                + coef[5] * (Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
                + coef[6] * (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t)
                + Lam[0] * exp(Lam[0] * t))

    # present [234U/238U]
    else:
        dF2 = Lam[1] * (A[0] - 1.) * exp(Lam[1] * t) * (
                Lam[0] / Lam[1] * (
                    coef[4] * exp((Lam[0] - Lam[1]) * t)
                    + coef[5] * exp((Lam[0] - Lam[2]) * t)
                    + coef[6] * exp((Lam[0] - Lam[3]) * t) + exp(Lam[0] * t))) \
               + ((A[0] - 1.) * exp(Lam[1] * t) + 1.) * (
                       Lam[0] / Lam[1] * (
                       coef[4] * (Lam[0] - Lam[1]) * exp((Lam[0] - Lam[1]) * t)
                       + coef[5] * (Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
                       + coef[6] * (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t)
                       + Lam[0] * exp(Lam[0] * t)))

    # initial [230Th/238U]
    if init[1]:
        dF3 = A[1] * Lam[0] / Lam[2] * (
                coef[7] * (Lam[0] - Lam[2]) * exp((Lam[0] - Lam[2]) * t)
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

    dF4 = A[2] * Lam[0] / Lam[3] * (Lam[0] * exp(Lam[0] * t)
             - (Lam[0] - Lam[3]) * exp((Lam[0] - Lam[3]) * t))

    return dF1, dF2, dF3, dF4


def dgdt(t, a231_235_i, Lam, coef):
    """
    Derivative of 207Pb*/235U (where * denotes radiogenic Pb) with respect to
    t.
    """
    dG1 = Lam[0] * exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t)
            + coef[1] * exp(-Lam[1] * t) + 1.) + exp(Lam[0] * t) * (
            -coef[0] * Lam[0] * exp(-Lam[0] * t)
            - Lam[1] * coef[1] * exp(-Lam[1] * t))
    dG2 = a231_235_i * Lam[0] / Lam[1] * (Lam[0] * exp(Lam[0] * t)
            - (Lam[0]-Lam[1]) * exp((Lam[0]-Lam[1]) * t))
    return dG1 + dG2


def dgdt_comp(t, a231_235_i, Lam, coef):
    """
    Individual components of derivative of 207Pb*/235U (where * denotes 
    radiogenic Pb) with respect to t.
    """
    dG1 = Lam[0] * exp(Lam[0] * t) * (coef[0] * exp(-Lam[0] * t)
            + coef[1] * exp(-Lam[1] * t) + 1.) + exp(Lam[0] * t) * (
            -coef[0] * Lam[0] * exp(-Lam[0] * t)
            - Lam[1] * coef[1] * exp(-Lam[1] * t))
    dG2 = a231_235_i * Lam[0] / Lam[1] * (Lam[0] * exp(Lam[0] * t)
            - (Lam[0]-Lam[1]) * exp((Lam[0]-Lam[1]) * t))
    return dG1, dG2

