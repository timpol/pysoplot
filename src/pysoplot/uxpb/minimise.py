"""
Minimisation functions for numerical disequilibrium age solutions.

References
----------
.. [Guillong2014]
    Guillong, M., von Quadt, A., Sakata, S., Peytcheva, I., Bachmann, O., 2014.
    LA-ICP-MS Pb-U dating of young zircons from the Kos–Nisyros volcanic centre,
    SE aegean arc. Journal of Analytical Atomic Spectrometry 29, 963–970.
    https://doi.org/10.1039/C4JA00009A
.. [Sakata2017]
    Sakata, S., Hirakawa, S., Iwano, H., Danhara, T., Guillong, M., Hirata, T.,
    2017. A new approach for constraining the magnitude of initial
    disequilibrium in Quaternary zircons by coupled uranium and thorium decay
    series dating. Quaternary Geochronology 38, 1–12.
    https://doi.org/10.1016/j.quageo.2016.11.002

"""

import numpy as np

from . import ludwig
from .. import misc


#=================================
# Disequilibrium Age minimisation functions
#=================================

def concage_x(diagram, t0=1.0, init=(False, False)):
    """ """
    assert diagram == 'tw'
    h = abs(t0) * np.sqrt(np.finfo(float).eps)
    def fmin(t, x, A, DC, BC):
        return x - 1. / ludwig.f(t, A, DC, BC, init=init)
    def dfmin(t, x, A, DC, BC):
        return misc.cdiff(t, fmin, h, x, A, DC, BC)
    return fmin, dfmin


def concint(t0=1.0, diagram='tw', init=(True, True)):
    """
    Disequilbrium concordia-intercept age minimisation function.
    """
    assert diagram in ('tw', 'wc')
    h = abs(t0) * np.sqrt(np.finfo(float).eps)
    if diagram == 'tw':
        def fmin(t, a, b, A238, A235, DC8, DC5, BC8, BC5, U):
            return b + a * ludwig.f(t, A238, DC8, BC8, init=init) \
                   - ludwig.g(t, A235, DC5, BC5) / U
        def dfmin(t, a, b, A238, A235, DC8, DC5, BC8, BC5, U):
            return misc.cdiff(t, fmin, h, a, b, A238, A235, DC8, DC5,
                    BC8, BC5, U)
    else:
        raise ValueError('not yet coded')
    return fmin, dfmin


def isochron(t0=1.0, age_type='iso-Pb6U8', init=(True, True)):
    """
    Disequilibrium isochron age minimisation function.
    """
    assert age_type in ('iso-Pb6U8', 'iso-Pb7U5')
    h = abs(t0) * np.sqrt(np.finfo(float).eps)
    if age_type == 'iso-Pb6U8':
        def fmin(t, b, A, DC, BC):
            return b - ludwig.f(t, A, DC, BC, init=init)
    elif age_type == 'iso-Pb7U5':
        def fmin(t, b, A, DC, BC):
            return b - ludwig.g(t, A, DC, BC)
    def dfmin(t, b, A, DC, BC):
        return misc.cdiff(t, fmin, h, b, A, DC, BC)
    return fmin, dfmin


def concordant_A48():
    """
    Minimisation function for computing initial U234/U238 activity ratio that
    forces concordance between 238U and 235U isochron ages.

    Minimises function: f(t75, A) - slope_86, where t75 is
    the 207Pb/x-235U/x isochron age.
    """
    h = 1.
    def fmin(A48i, t57, slope_86, A, DC, BC):
        return ludwig.f(t57, [A48i, A[1], A[2]], DC, BC) - slope_86
    def dfmin(A48i, t57, slope_86, A, DC, BC):
        return misc.cdiff(A48i, fmin, h, t57, slope_86, A, DC, BC)
        # return DC[0] / DC[1] * np.exp(DC[0] * t57) * (BC[3] * np.exp(-DC[1] * t57)
        #        + (BC[4] * np.exp(-DC[1] * t57) + 1.))
    return fmin, dfmin


def pbu(age_type='Pb6U8', t0=1., init=(True, True)):
    """
    Disequilbrium Pb/U age functions.
    """
    h = abs(t0) * np.sqrt(np.finfo(float).eps)
    if age_type == 'Pb6U8':
        # x is 206Pb/238U or 207Pb/235U
        def fmin(t, x, A, DC, BC):
            return x - ludwig.f(t, A, DC, BC, init=init)
        def dfmin(t, x, A, DC, BC):
            return misc.cdiff(t, fmin, h, x, A, DC, BC)
    elif age_type == 'Pb7U5':
        def fmin(t, x, A, DC, BC):
            return x - ludwig.g(t, A, DC, BC)
        def dfmin(t, x, A, DC, BC):
            return misc.cdiff(t, fmin, h, x, A, DC, BC)
    return fmin, dfmin


def guillong(t0=1.):
    """
    Minimisation function for disequilibrium Pb/U ages using the equations of
    Guillong et al., (2014) - eith 206Pb/238U or 207Pb/235U ages.
    fXU is either fThU or fPaU
    lamU is either lam238 or lam235
    lamX is either lam230 or lam231
    """
    def fmin(t, x, fXU, lamU, lamX):
        """ Age minisation functions using equations of Guillong (2014).
        """
        return (np.exp(lamU * t) - 1.) + lamU / lamX \
               * (fXU - 1.) * (1. - np.exp(-lamX * t)) \
               * np.exp(lamU * t) - x
    def dfmin(t, x, fXU, lamU, lamX):
        """x is dummy argument
        """
        return lamU * np.exp(lamU * t) + lamU / lamX * (fXU - 1.) \
               * (lamU - (lamU - lamX) * np.exp(- lamX * t)) \
               * np.exp(lamU * t)
    return fmin, dfmin


def mod207(t0=1., init=(True, True)):
    """
    Age minimisation function for modified 207Pb ages.
    """
    h = abs(t0) * np.sqrt(np.finfo(float).eps)
    def fmin(t, x, y, A, DC8, DC5, BC8, BC5, U, Pb76):
        """
        x : measured 206Pb*/238U
        y : measured 207Pb*/206Pb
        """
        num = ludwig.g(t, A[-1], DC5, BC5) / U - (y - Pb76) / x
        denom = ludwig.f(t, A[:-1], DC8, BC8, init=init)
        return num / denom - Pb76
    def dfmin(t, x, y, A, DC8, DC5, BC8, BC5, U, Pb76):
        out = misc.cdiff(t, fmin, h, x, y, A, DC8, DC5, BC8, BC5, U, Pb76)
        return out
    return fmin, dfmin


def sakata(t0=1.):
    """
    Pb/U age minimisation functions using equaitons of Sakata (2017).
    """
    h = abs(t0) * np.sqrt(np.finfo(float).eps)
    def fmin(t, x, y, fThU, fPaU, lam238, lam230, lam235, lam231, U, Pb76):
        F = np.exp(lam238 * t) - 1. \
              + lam238 / lam230 * (fThU - 1) * (1 - np.exp(-lam230 * t)) \
              * np.exp(lam238 * t)
        G = (np.exp(lam235 * t) - 1.) \
              + lam235 / lam231 * (fPaU - 1) * (1 - np.exp(-lam231 * t)) \
              * np.exp(lam235 * t)
        num = (1 / U) * G - (y - Pb76) / x
        return num / F - Pb76
    def dfmin(t, x, y, fThU, fPaU, lam238, lam230, lam235, lam231, U, Pb76):
        return misc.cdiff(t, fmin, h, x, y, fThU, fPaU, lam238, lam230, lam235,
                          lam231, U, Pb76)
    return fmin, dfmin

