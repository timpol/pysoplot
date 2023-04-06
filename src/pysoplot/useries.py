"""
Uranium-series functions and routines.

"""

import numpy as np

from . import cfg


exp = np.exp


#=====================================
# 234U age equaions
#=====================================

def aratio48i(t, a234_238, Lam=None):
    """
    Calculate initial 234U/238U activity ratio as a function of present
    ratio and age (Ma).

    Notes
    ------
    Non-secular equilibrium equation is derived from Bateman (1910) and avoids
    the assumption of negligible decay of 238U (e.g., Ivanovich and Harmon
    (1992)). This is an experimental feature.

    References
    ----------
    Ivanovich, M. and Harmon, R. S.: Uranium-Series Disequilibrium:
    Applications to Earth, Marine, and Environmental Sciences., Clarendon
    Press, United Kingdom, second edn., 1992.

    """
    if Lam is None:
        Lam = (cfg.lam238, cfg.lam234)
    if cfg.secular_eq:
        return 1. + (a234_238 - 1.) * exp(Lam[1] * t)
    else:
        return Lam[1] / (Lam[1]-Lam[0]) + (a234_238 - Lam[1] / (Lam[1]-Lam[0])) \
               * exp(-(Lam[0]-Lam[1])*t)



def aratio48(t, a234_238_i, Lam=None):
    """
    Calculate initial 234U/238U activity ratio as a function of initial
    ratio and age (Ma).

    Notes
    ------
    Non-secular equilibrium equation is derived from Bateman (1910) and avoids
    the assumption of negligible decay of 238U (e.g., Ivanovich and Harmon
    (1992)). This is an experimental feature.

    References
    ----------
    Ivanovich, M. and Harmon, R. S.: Uranium-Series Disequilibrium:
    Applications to Earth, Marine, and Environmental Sciences., Clarendon
    Press, United Kingdom, second edn., 1992.

    """
    if Lam is None:
        Lam = (cfg.lam238, cfg.lam234)
    if cfg.secular_eq:
        return 1. + (a234_238_i - 1.) * exp(-Lam[1] * t)
    else:
        return Lam[1]/(Lam[1]-Lam[0]) + (a234_238_i - Lam[1]/(Lam[1]-Lam[0])) \
               * exp((Lam[0]-Lam[1])*t)
    

#=====================================
# Th230 equations
#=====================================

def Th230_age():
    pass


def Th230_age_uncert():
    pass


def aratio08i(t, a234_238, a230_238, init=True, Lam=None):
    """
    Calculate initial 230Th/238U as a function of present ratio and
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
    if Lam is None:
        Lam = (cfg.lam238, cfg.lam234, cfg.lam230)
    
    if cfg.secular_eq:
        if init:
            a234_238_i = a234_238
            a230_238_i = 1. - (a234_238_i - 1.) * (Lam[2]/(Lam[2]-Lam[1])) \
                   * (exp((Lam[2]-Lam[1]) * t) - 1) + (a230_238 - 1.) * exp(Lam[2]*t)
        else:
            a230_238_i = 1. - (a234_238 - 1.) * (Lam[2]/(Lam[2]-Lam[1])) \
                   * (exp(Lam[2]*t) - exp(Lam[1]*t)) + (a230_238-1.) * exp(Lam[2] * t)
        return a230_238_i

    else:
        a234_238_i = aratio48i(t, a234_238) if init else a234_238

        c1 = Lam[2]*Lam[1] / ((Lam[1]-Lam[0]) * (Lam[2]-Lam[0]))
        c2 = Lam[2]*Lam[1] / ((Lam[0]-Lam[1]) * (Lam[2]-Lam[1]))
        c3 = Lam[2]*Lam[1] / ((Lam[0]-Lam[2]) * (Lam[1]-Lam[2]))

        A08i = (a230_238 - (c1 + c2*exp((Lam[0]-Lam[1])*t) + c3*exp((Lam[0]-Lam[2])*t)
                + a234_238_i * (Lam[2]/(Lam[2]-Lam[1])) * (exp((Lam[0]-Lam[1])*t)
                   - exp((Lam[0]-Lam[2]) * t)))) * exp(-(Lam[0]-Lam[2]) * t)
        return A08i


def aratio08(t, a234_238, a230_238_i, init=True, Lam=None):
    """
    Calculate present 230Th/238U as a function of present ratio and
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
        raise ValueError('non-secular equation not yet implemented')

    if Lam is None:
        Lam = (cfg.lam238, cfg.lam234, cfg.lam230)

    if init:
        a234_238_i = a234_238
        a230_238 = 1. - exp(-Lam[2]*t) * (1. - a230_238_i) + (a234_238_i - 1.) \
                    * (Lam[2]/(Lam[2]-Lam[1])) * (exp(-Lam[1]*t) - exp(-Lam[2]*t))
    else:
        a230_238 = 1. - exp(-Lam[2]*t) * (1. - a230_238_i) + (a234_238-1.) \
                   * (Lam[2]/(Lam[2]-Lam[1])) * (1.-exp((Lam[1]-Lam[2])*t))
    return a230_238


#=====================================
# U/Th isochron age equations
#=====================================

def Th230_isochron():
    pass



#==================================================
# Back-calculate initial activity ratio solutions
#===================================================

def init_ratio_solutions(t, A, meas, Lam=None):
    """
    Compute initial activity ratio solutions from present-day values and U-Pb age
    solution. In principle these values are computed iteratively along with the
    age solution, but practically it is more convenient to compute them separately
    
    """
    a234_238_i = None
    a230_238_i = None
    if meas[0]:        # present [234U/238U]
        a234_238_i = aratio48i(t, A[0], Lam=Lam)
        if meas[1]:    # present [234U/238U] and [230Th/238U]
            a230_238_i = aratio08i(t, aratio48i, A[1], Lam=Lam)
    elif meas[1]:      # only present [230Th/238U]
        a230_238_i = aratio08i(t, A[0], A[1], init=not meas[0], Lam=Lam)
    return a234_238_i, a230_238_i
