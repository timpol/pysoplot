"""
Functions and routines for calculating U-Pb ages **corrected** for
initial disequilibrium under the assumption that radioactive equilibrium
has been established at the time of analysis.

!!! This module contains experimental functions and routines that haven't
been properly tested or debugged !!!

Notes
-------
See dqpb module for disequilibrium U-Pb age functions - to be used if the sample
cannot be assumed to have attained radioactive equilbrium at the time
of analysis.

References
----------
.. [Mclean2011]
    McLean, N.M., Bowring, J.F., Bowring, S.A., 2011. An algorithm for U-Pb
    isotope dilution data reduction and uncertainty propagation. Geochemistry,
    Geophysics, Geosystems 12. https://doi.org/10.1029/2010GC003478
.. [Scharer1984]
    Schärer, U., 1984. The effect of initial $^{230}$Th disequilibrium on young
    U-Pb ages: the Makalu case, Himalaya. Earth and Planetary Science Letters
    67, 191–204.

"""

import numpy as np

from .. import cfg
from .. import plotting
from .. import wtd_average
from .. import monte_carlo as mc
from .. import misc


#=================================
# Single analysis Pb/U ages
#=================================

def pb206_age(x, sx, ThU_type=1, DThU=None, DThU_se=None, ThU_min=None,
        ThU_melt=None, ThU_melt_se=None, wav=False, dc_errors=False,
        sx_external=0., cov=True, wav_method='ca'):

    """
    !!! Experimental function !!!
    Compute 206Pb/238U ages corrected for initial 230Th/238U disequilbrium,
    under the assumption that the 238U-206Pb decay series has established
    radioactive equilbrium at the time of analysis (e.g. Schärer, 1984).
    """
    assert ThU_type in (1, 2)
    lam238, lam230 = cfg.lam238, cfg.lam230
    s238, s230 = cfg.s238, cfg.s230
    cov80 = cfg.cor_238_230 * s238 * s230
    n = x.shape[0]

    # rename parameters
    D = DThU
    sD = DThU_se
    rmin = ThU_min
    rmelt = ThU_melt
    srmelt = ThU_melt_se

    if ThU_type == 1:
        f = np.broadcast_to(D, (n,))          # note, f = f_ThU = (Th/U_min) / (Th/U_mag)
    else:
        f = rmin / rmelt

    t = -1. / lam238 * np.log(1. - x + lam238 / lam230 * (f - 1.))

    # compute age uncertainty
    dtx = np.exp(lam238 * t) / lam238            # dt/dx etc.

    if dc_errors:
        dt8 = - 1. / lam238 * (t + (f - 1) * np.exp(lam238 * t) / lam230)
        dt0 = (f - 1.) * np.exp(lam238 * t) / lam230 ** 2
    else:
        dt8 = np.zeros(n)
        dt0 = np.zeros(n)

    # Compile age covariance matrix following, e.g., McLean et al., (2011).
    # random errors:
    cov_tr = np.diag((dtx * sx) ** 2)

    # systematic errors:
    cov_s = np.zeros((4, 4))
    cov_s[-2:, -2:] = np.array([[s238 ** 2, cov80], [cov80, s230 ** 2]])
    cov_s[0, 0] = sx_external ** 2

    if ThU_type == 1:
        cov_s[1, 1] = sD ** 2
        dtD = - np.exp(lam238 * t) / lam230
        jac = np.stack((dtx, dtD, dt8, dt0), axis=-1)
    else:
        cov_s[1, 1] = srmelt ** 2
        dtrm = rmin * np.exp(lam238 * t) / (lam230 * rmelt ** 2)     # dt / d(rmelt)
        jac = np.stack((dtx, dtrm, dt8, dt0), axis=-1)

    cov_ts = jac @ cov_s @ jac.T

    # combined errors:
    cov_t = cov_tr + cov_ts
    st = np.sqrt(np.diag(cov_t))

    results = {
        'age': t,
        'age_1s': st,
        'age_95ci': 1.96 * st,
        'cov_t': cov_t,
        'cor_t': misc.covmat_to_cormat(cov_t)
    }

    if wav:
        if not cov:
            cov_t = np.diag(np.diag(cov_t))
        if wav_method in ('ca'):
            wav_results = wtd_average.classical_wav(t, cov=cov_t,
                                method=wav_method)
        elif wav_method in ('ra', 'sp', 'r2'):
            wav_results = wtd_average.robust_wav(t, cov=cov_t, method=wav_method)
        else:
            raise ValueError('wav method not recognised')

        results['wav_age'] = wav_results['ave']
        results['wav_age_95ci'] = wav_results['ave_95ci']
        results['wav'] = wav_results

        # Make wav plot:
        fig = plotting.wav_plot(t, results['age_95ci'], wav_results['ave'],
                    wav_results['ave_95ci'], sorted=sorted)
        ax = fig.get_axes()[0]
        ax.set_ylabel('$^{206}$Pb/$^{238}$U age (Ma)')
        results['wav_plot'] = fig

    return results


def pb207_age():
    return


#==========================
# Concordia-intercept age
#==========================

def concint_age():
    return

def isochron_age():
    return


#=================================
# Monte Carlo age uncertainties
#=================================

def mc_pb206_cor(t, x, sx, ThU_type=1, DThU=None, DThU_se=None, ThU_min=None,
            ThU_melt=None, ThU_melt_se=None, trials=10_000,  dc_errors=False,
            negative_ages=False, rand=False, hist=(False, False),
            wav_method='ca', wav=False):
    """
    !!! Experimental function !!!
    Compute age uncertainties for corrected Pb/U age using Monte Carlo
    approach.
    """
    assert not any([np.isnan(t) for t in t])       # nan ages should be removed

    n = x.size
    flags = np.zeros((trials, n), dtype='uint8')
    ts = np.empty((trials, n))                  # pre-allocate ages array

    # simulate fThU values
    if ThU_type == 1:
        fThU = cfg.rng.normal(DThU, DThU_se, trials)
        fThU = np.broadcast_to(fThU.reshape(trials, 1), (trials, n))
    elif ThU_type == 2:
        ThU_melt = cfg.rng.normal(ThU_melt, ThU_melt_se, (trials, 1))
        fThU = np.broadcast_to(ThU_min, (trials, n)) / ThU_melt

    if dc_errors:
        lam238 = cfg.rng.normal(cfg.lam238, cfg.s238, trials)
        lam230 = cfg.rng.normal(cfg.lam230, cfg.s230, trials)
    else:
        lam238 = cfg.lam238
        lam230 = cfg.lam230

    for i in range(n):
        xs = cfg.rng.normal(x[i], sx[i], trials)
        ts[:, i] = -1. / lam238 * np.log(1. - xs + lam238 / lam230 * (fThU[:, i] - 1.))

    good = np.all(flags == 0, axis=1)
    age_95ci = [np.quantile(t[good], (0.025, 0.975)) for t in ts.T]

    # compile results
    results = {
        'age_type': 'corrected 206Pb',
        'age_1s': [np.std(t[good]) for t in ts.T],
        'age_pm_95ci': [np.mean((t - m, p - t)) for m, p in age_95ci],
        'age_95ci': [np.quantile(t[good], (0.025, 0.975)) for t in ts.T],
        'mean_age': [np.mean(t[good]) for t in ts.T],
        'median_age': [np.median(t[good]) for t in ts.T],
        'cov_t': np.cov(np.transpose(ts[good])),
        'cor_t': np.corrcoef(np.transpose(ts[good])),
        'trials': trials,
        'fails': sum(flags != 0),
        'not_converged': sum(flags == mc.NON_CONVERGENCE),
        'negative_ages': sum(flags == mc.NEGATIVE_AGE)
    }

    if wav:
        cov_t = results['cov_t']
        if wav_method in ('ca'):
            wav_results = wtd_average.classical_wav(t, cov=cov_t,
                                method=wav_method)
        elif wav_method in ('ra', 'sp', 'r2'):
            wav_results = wtd_average.robust_wav(t, cov=cov_t, method=wav_method)
        else:
            raise ValueError('wav method not recognised')

        results['wav_age'] = wav_results['ave']
        results['wav_age_95ci'] = wav_results['ave_95ci']
        results['wav'] = wav_results

        # Make wav plot:
        fig = plotting.wav_plot(t, results['age_pm_95ci'], wav_results['ave'],
                    wav_results['ave_95ci'], sorted=sorted)
        ax = fig.get_axes()[0]
        ax.set_ylabel('$^{206}$Pb/$^{238}$U age (Ma)')
        results['wav_plot'] = fig

    return results
