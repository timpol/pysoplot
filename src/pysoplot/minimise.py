"""
Minimisation functions for numerical disequilibrium age solutions.

"""

import numpy as np

from pysoplot import misc
from pysoplot import ludwig
from pysoplot import cfg

exp = np.exp


#=================================
# Disequilibrium Age minimisation functions
#=================================

def concage_x(diagram, init=(False, False)):
    """ """
    assert diagram == 'tw'
    def fmin(t, x, A, Lam, coef):
        return x - 1. / ludwig.f(t, A, Lam, coef, init=init)
    def dfmin(t, x, A, Lam, coef):
        return - ludwig.f(t, A, Lam, coef, init=init) / (
                ludwig.f(t, A, Lam, coef, init=init) ** 2)
    return fmin, dfmin


def concint(diagram='tw', init=(True, True)):
    """
    Disequilbrium concordia-intercept age minimisation function.
    """
    assert diagram in ('tw', 'wc')
    # TODO: dfdt doesn't yet handle non-secular eq u-series equations, so this
    #  analytical derivative may fail if cfg.sec_eq is True .
    if diagram == 'tw':
        def fmin(t, a, b, A238, A235, Lam238, Lam235, coef238, coef235, U):
            return b + a * ludwig.f(t, A238, Lam238, coef238, init=init) \
                   - ludwig.g(t, A235, Lam235, coef235) / U
        def dfmin(t, a, b, A238, A235, Lam238, Lam235, coef238, coef235, U):
            return a * ludwig.dfdt(t, A238, Lam238, coef238, init=init) \
                   - ludwig.dgdt(t, A235, Lam235, coef235) / U
    else:
        raise ValueError('not yet coded')
    return fmin, dfmin


def isochron(age_type='iso-206Pb', init=(True, True)):
    """
    Disequilibrium isochron age minimisation function.

    """
    assert age_type in ('iso-206Pb', 'iso-207Pb')
    # TODO: dfdt doesn't yet handle non-secular eq u-series equations, so this
    #  analytical derivative may fail if cfg.sec_eq is True .
    if age_type == 'iso-206Pb':
        def fmin(t, b, A, Lam, coef):
            return b - ludwig.f(t, A, Lam, coef, init=init)
        def dfmin(t, b, A, Lam, coef):
            return -ludwig.dfdt(t, A, Lam, coef, init=init)
    elif age_type == 'iso-207Pb':
        def fmin(t, b, A, Lam, coef):
            return b - ludwig.g(t, A, Lam, coef)
        def dfmin(t, b, A, Lam, coef):
            return  -ludwig.dgdt(t, A, Lam, coef)
    return fmin, dfmin


def concordant_A48():
    """
    Minimisation function for computing initial U234/U238 activity ratio that
    forces concordance between 238U and 235U isochron ages.

    Minimises function: f(t75, A) - slope_86, where t75 is
    the 207Pb/x-235U/x isochron age.
    """
    def fmin(A48i, t57, slope_86, A, Lam, coef):
        return ludwig.f(t57, [A48i, A[1], A[2]], Lam, coef) - slope_86
    def dfmin(A48i, t57, slope_86, A, Lam, coef):
        return ludwig.dfdt(t57, [A48i, A[1], A[2]], Lam, coef)
    return fmin, dfmin


def pbu(age_type='206Pb*', init=(True, True)):
    """
    Age minimisation function for disequilibrium Pb*/U and
    207Pb-corrected ages.
    """
    assert age_type in ('206Pb*', '207Pb*', 'cor207Pb')

    if age_type == '206Pb*':

        def fmin(t, x, A, Lam, coef):
            return x - ludwig.f(t, A, Lam, coef, init=init)

        def dfmin(t, x, A, Lam, coef):
            return - ludwig.dfdt(t, A, Lam, coef, init=init)

    elif age_type == '207Pb*':

        def fmin(t, x, A, Lam, coef):
            return x - ludwig.g(t, A, Lam, coef)

        def dfmin(t, x, A, Lam, coef):
            return -ludwig.dgdt(t, A, Lam, coef)

    else:
        def fmin(t, x, y, A, alpha, U, Lam238, Lam235, coef238, coef235):
            """
            x : measured 206Pb*/238U
            y : measured 207Pb*/206Pb
            alpha : common 207Pb/206Pb
            """
            num = ludwig.g(t, A[-1], Lam235, coef235) / U - (y - alpha) / x
            denom = ludwig.f(t, A[:-1], Lam238, coef238, init=init)
            return num / denom - alpha

        def dfmin(t, x, y, A, alpha, U, Lam238, Lam235, coef238, coef235):
            num = ludwig.g(t, A[-1], Lam235, coef235) / U - (y - alpha) / x
            dnum = ludwig.dgdt(t, A[-1], Lam235, coef235) / U
            den = ludwig.f(t, A[:-1], Lam238, coef238, init=init)
            dden = ludwig.dfdt(t, A[:-1], Lam238, coef238, init=init)
            return (dnum * den - num * dden) / den ** 2

    return fmin, dfmin


def pbu_iterative(age_type='206Pb*', meas_232Th_238U=True):
    """
    Age functions for disequilibrium 206Pb*/U238 ages computed
    iteratively. Either (232Th/238U)i is inferred from measured 232Th/238U, or
    from radiogenic 208Pb/206Pb and age.

    """
    if age_type == '206Pb*':

        if meas_232Th_238U:
            def fmin(t, x, Th232_U238, ThU_melt, Lam238, coef238):

                ThU_min = Th232_U238 * (exp(cfg.lam232 * t)
                        / (exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U))
                fThU = ThU_min / ThU_melt

                return x - ludwig.f(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                                    Lam238, coef238)

            def dfmin(t, x, Th232_U238, ThU_melt, Lam238, coef238):

                ThU_min = Th232_U238 * (exp(cfg.lam232 * t)
                        / (exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U))
                fThU = ThU_min / ThU_melt
                den = np.exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U
                dfThU = Th232_U238 / ThU_melt * np.exp(cfg.lam232 * t) * (
                        cfg.lam232 * den - cfg.lam238 * (cfg.lam238 * t)
                        + np.exp(cfg.lam235 * t) / cfg.U) / den ** 2

                df1, df2, _, df4 = ludwig.dfdt_comp(t, [cfg.a234_238_eq, np.nan,
                                 cfg.a226_238_eq], Lam238, coef238)
                df3 = dfThU * Lam238[0]/Lam238[2] * (coef238[7] * exp((Lam238[0]-Lam238[2]) * t)
                        + coef238[8] * exp((Lam238[0]-Lam238[3]) * t) + exp(Lam238[0]*t)) \
                     + fThU * Lam238[0] / Lam238[2] * (
                        coef238[7] * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                        + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                        + Lam238[0] * exp(Lam238[0] * t))

                return -(df1 + df2 + df3 + df4)

        else:
            def fmin(t, x, Pb208_206, ThU_melt, Lam238, coef238):

                Th232_U238 = x * Pb208_206 * (1. / (np.exp(cfg.lam232 * t) - 1.))
                ThU_min = Th232_U238 * (exp(cfg.lam232 * t)
                        / (exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U))
                fThU = ThU_min / ThU_melt

                return x - ludwig.f(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                                    Lam238, coef238)

            def dfmin(t, x, Pb208_206, ThU_melt, Lam238, coef238):
                """

                """
                num = np.exp(cfg.lam232 * t) / (np.exp(cfg.lam232 * t) - 1.)
                den = np.exp(cfg.lam238 * t) +  np.exp(cfg.lam235 * t) / cfg.U
                fThU = x * Pb208_206 * (num / den) / ThU_melt

                dnum = -cfg.lam232 * np.exp(cfg.lam232 * t) / (
                        np.exp(cfg.lam232 * t) - 1.) ** 2
                dden = cfg.lam238 * np.exp(cfg.lam238 * t) \
                       + cfg.lam235 * np.exp(cfg.lam235 * t) / cfg.U
                dfThU = x * Pb208_206 / ThU_melt * (dnum * den
                            - num * dden) / den ** 2

                df1, df2, _, df4 = ludwig.dfdt_comp(t, [cfg.a234_238_eq, np.nan,
                                 cfg.a226_238_eq], Lam238, coef238)
                df3 = dfThU * Lam238[0]/Lam238[2] * (coef238[7] * exp((Lam238[0]-Lam238[2]) * t)
                        + coef238[8] * exp((Lam238[0]-Lam238[3]) * t) + exp(Lam238[0]*t)) \
                     + fThU * Lam238[0] / Lam238[2] * (
                        coef238[7] * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                        + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                        + Lam238[0] * exp(Lam238[0] * t))

                return -(df1 + df2 + df3 + df4)

    elif age_type == 'cor207Pb':

        if meas_232Th_238U:
            def fmin(t, x, y, Th232_U238, ThU_melt, fPaU, alpha, Lam238, Lam235,
                     coef238, coef235):
                """
                x : measured 206Pb/238U
                y : measured 207Pb/206Pb
                alpha : common 207Pb/206Pb
                """
                Th232_U238_init = Th232_U238 * np.exp((cfg.lam238 - cfg.lam232) * t)
                ThU_min = Th232_U238_init * (cfg.U / (cfg.U + 1.))
                fThU = ThU_min / ThU_melt

                num = ludwig.g(t, fPaU, Lam235, coef235) / cfg.U - (y - alpha) / x
                den = ludwig.f(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                               Lam238, coef238)
                return num / den - alpha

            def dfmin(t, x, y, Th232_U238, ThU_melt, fPaU, alpha, Lam238, Lam235,
                     coef238, coef235):
                # TODO: code analytical derivative
                h = t * 1e-08
                args = (x, y, Th232_U238, ThU_melt, fPaU, alpha, Lam238, Lam235,
                     coef238, coef235)
                return misc.cdiff(t, fmin, h, *args)

        else:
            def fmin(t, x, y, Pb208_206, ThU_melt, fPaU, alpha, Lam238, Lam235,
                     coef238, coef235):
                """
                x : measured 206Pb/238U
                y : measured 207Pb/206Pb
                alpha : common 207Pb/206Pb
                """

                Th232_U238 = (1. / x) * Pb208_206 * (1. / (np.exp(cfg.lam232 * t) - 1.))
                ThU_min = Th232_U238 * (exp(cfg.lam232 * t)
                        / (exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U))
                fThU = ThU_min / ThU_melt

                num = ludwig.g(t, fPaU, Lam235, coef235) / cfg.U - (y - alpha) / x
                den = ludwig.f(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                                    Lam238, coef238)
                return num / den - alpha

            def dfmin(t, x, y, Pb208_206, ThU_melt, fPaU, alpha, Lam238, Lam235,
                      coef238, coef235):
                """

                """
                # TODO: double chek this
                # num = np.exp(cfg.lam232 * t) / (np.exp(cfg.lam232 * t) - 1.)
                # den = np.exp(cfg.lam238 * t) +  np.exp(cfg.lam235 * t) / cfg.U
                # fThU = x * Pb208_206 * (num / den) / ThU_melt
                #
                # dnum = -cfg.lam232 * np.exp(cfg.lam232 * t) / (
                #         np.exp(cfg.lam232 * t) - 1.) ** 2
                # dden = cfg.lam238 * np.exp(cfg.lam238 * t) \
                #        + cfg.lam235 * np.exp(cfg.lam235 * t) / cfg.U
                # dfThU = x * Pb208_206 / ThU_melt * (dnum * den
                #             - num * dden) / den ** 2
                #
                # num = ludwig.g(t, fPaU, Lam235, coef235) / cfg.U - (y - alpha) / x
                # dnum = ludwig.dgdt(t, fPaU, Lam235, coef235) / cfg.U
                #
                # den = ludwig.f(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                #                     Lam238, coef238)
                # df1, df2, _, df4 = ludwig.dfdt_comp(t, [cfg.a234_238_eq, np.nan,
                #                      cfg.a226_238_eq], Lam238, coef238)
                # df3 = dfThU * Lam238[0] / Lam238[2] * (
                #         coef238[7] * exp((Lam238[0] - Lam238[2]) * t)
                #         + coef238[8] * exp((Lam238[0] - Lam238[3]) * t) + exp(Lam238[0] * t)) \
                #       + fThU * Lam238[0] / Lam238[2] * (
                #         coef238[7] * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                #         + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                #         + Lam238[0] * exp(Lam238[0] * t))
                # dden = df1 + df2 + df3 + df4
                #
                # return (dnum * den - num * dden) / den ** 2
                h = t * 1e-08
                args = (x, y, Pb208_206, ThU_melt, fPaU, alpha, Lam238, Lam235,
                        coef238, coef235)
                return misc.cdiff(t, fmin, h, *args)

    return fmin, dfmin


def guillong(t0=1.):
    """
    Minimisation function for disequilibrium Pb/U ages using the equations of
    Guillong et al., (2014) - eith 206Pb/238U or 207Pb/235U ages.
    fXU is either fThU or fPaU

    Parameters
    -----------
    lamU is either lam238 or lam235
    lamX is either lam230 or lam231

    References
    -----------
    Guillong, M., von Quadt, A., Sakata, S., Peytcheva, I., Bachmann, O., 2014.
    LA-ICP-MS Pb-U dating of young zircons from the Kos–Nisyros volcanic centre,
    SE aegean arc. Journal of Analytical Atomic Spectrometry 29, 963–970.
    https://doi.org/10.1039/C4JA00009A

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


def sakata(t0=1.):
    """
    Pb/U age minimisation functions using equaitons of Sakata (2017).

    References
    ----------
    Sakata, S., Hirakawa, S., Iwano, H., Danhara, T., Guillong, M., Hirata, T.,
    2017. A new approach for constraining the magnitude of initial
    disequilibrium in Quaternary zircons by coupled uranium and thorium decay
    series dating. Quaternary Geochronology 38, 1–12.
    https://doi.org/10.1016/j.quageo.2016.11.002

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
        # TODO: implement anlytical derivative
        return misc.cdiff(t, fmin, h, x, y, fThU, fPaU, lam238, lam230, lam235,
                          lam231, U, Pb76)
    return fmin, dfmin

