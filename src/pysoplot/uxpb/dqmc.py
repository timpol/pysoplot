"""
Monte Carlo disequilibrium U-Pb age errors

"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

from . import minimise
from . import ludwig
from .. import monte_carlo as mc
from .. import cfg
from .. import concordia
from .. import plotting
from .. import useries


#===========================================================================
# Concordia-intercept ages
#===========================================================================

def concint_age(t, fit, A, sA, init, trials=50_000, dc_errors=False,
        diagram='tw', u_errors=False, negative_ar=True, negative_ages=True,
        hist=(False, False), intercept_plot=True,
        intercept_plot_kw=None, conc_kw=None):
    """
    Propagate disequilibrium U-Pb concordia-intercept age uncertainties using
    Monte Carlo simulation.
    
    Parameters
    ----------
    t0 : float
        age solution
    fit : dict
        linear regression fit parameters
    A : np.ndarray
        one-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U]
    sA : np.ndarray
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A
    init : array-like
        two-element list of boolean values, the first is True if [234U/238U]
        is an initial value and False if a present-day value, the second is True
        if [230Th/238U] is an initial value and False if a present-day value

    """
    assert diagram == 'tw'  # wc not yet implemented
    flags = np.zeros(trials, dtype='uint8')

    if intercept_plot_kw is None:
        intercept_plot_kw = {}
    if conc_kw is None:
        conc_kw = {}

    theta, flags = mc.draw_theta(fit, trials, flags)
    U = cfg.rng.normal(cfg.U, cfg.sU, trials) if u_errors else cfg.U
    DC8, flags = draw_decay_const(trials, flags, dc_errors=dc_errors)
    DC5, flags = draw_decay_const(trials, flags, dc_errors=dc_errors, 
                                  series='235U')
    BC8 = ludwig.bateman(DC8, series='238U')
    BC5 = ludwig.bateman(DC5, series='235U')

    pA = np.empty((4, trials))
    pA[:3, :], flags = draw_ar(A[:-1], sA[:-1], trials, flags, positive_only=not negative_ar)
    pA[3, :], flags = draw_ar(A[-1], sA[-1], trials, flags, series='235U',
                              positive_only=not negative_ar)

    args = (*theta, pA[:3], pA[-1], DC8, DC5, BC8, BC5, U)

    # run vectorised Newton routine:
    fmin, dfmin = minimise.concint(t0=1.0, diagram='tw', init=init)
    ts, c, zd = optimize.newton(fmin, np.ones(trials) * t, dfmin, args=args,
                                full_output=True)
    flags = mc.check_ages(ts, c, flags, negative_ages=negative_ages)

    # calculate initial activity ratios
    A48i, A08i = useries.Ai_solutions(ts, pA[:3, :], init, DC8)

    if not negative_ar:
        if A48i is not None:
            flags = mc.update_flags(flags, A48i < 0., mc.NEGATIVE_AR_SOL)
        if A08i is not None:
            flags = mc.update_flags(flags, A08i < 0., mc.NEGATIVE_AR_SOL)

    ok = flags == 0
    if sum(ok) == 0:
        raise RuntimeError('all Monte Carlo simulation trials failed')

    age_95ci = np.quantile(ts[ok], (0.025, 0.975))

    # compute results
    results = {
        'age_type': 'concordia-intercept',
        'age_1s': np.std(ts[ok]),
        'age_95ci': age_95ci,
        'age_95pm': np.nanmean([t - age_95ci[0], age_95ci[1] - t]),
        'mean_age': np.nanmean(ts[ok]),
        'median_age': np.nanmedian(ts[ok]),
        'trials': trials,
        'fails': np.sum(flags != 0),
        'not_converged': np.sum(flags == mc.NON_CONVERGENCE),
        'negative_ages': np.sum(flags == mc.NEGATIVE_AGE),
        'negative_ar': np.sum(flags == mc.NEGATIVE_AR_SIM),
        'negative_ar_soln': np.sum(flags == mc.NEGATIVE_AR_SOL)
    }

    if A48i is not None:
        results['mean_[234U/238U]_i'] = np.nanmean(A48i[ok])
        results['median_[234U/238U]_i'] = np.nanmedian(A48i[ok])
        results['[234U/238U]i_1sd'] = np.std(A48i[ok])
        results['[234U/238U]i_95ci'] = np.quantile(A48i[ok], (0.025, 0.975))
        results['cor_age_[234U/238U]i'] = np.corrcoef(np.row_stack((A48i[ok],
                                                A48i[ok])))[0, 1]

    if A08i is not None:
        results['mean_[230Th/238U]_i'] = np.nanmean(A08i[ok])
        results['median_[230Th/238U]_i'] = np.nanmedian(A08i[ok])
        results['[230Th/238U]i_1sd'] = np.nanstd(A08i[ok])
        results['[230Th/238U]i_95ci'] = np.quantile(A08i[ok], (0.025, 0.975))
        results['cor_age_[230Th/238U]i'] = np.corrcoef(np.row_stack((A08i[ok],
                                                 A08i[ok])))[0, 1]

    # compile plots:
    if intercept_plot:
        fig = diseq_intercept_plot(ts, fit, pA[:3, :], pA[3], DC8, DC5, BC8, BC5,
                   U, flags, dp=None, init=init, dc_errors=dc_errors,
                   u_errors=u_errors, **intercept_plot_kw)
        ax = fig.get_axes()[0]
        # TODO: allow concordia func to accept pre-simulated A values to account
        # rejected values...
        concordia.plot_diseq_concordia(ax, A, init, diagram, sA=sA,
                                negative_ar=negative_ar, **conc_kw)
        results['fig'] = fig

    if any(hist):
        if hist[0]:
            a, b = theta
            xs, ys = concordia.diseq_xy(ts, pA, init, diagram)
            fig = mc.age_hist(ts[ok], diagram, a[ok], b[ok], xs[ok], ys[ok])
            results['age_hist'] = fig
        if hist[1]:
            fig = mc.ar_hist(np.transpose(np.transpose(pA[:3])[ok]),
                             pA[-1][ok], init)
            results['ar_hist'] = fig
        if hist[1] and (A48i is not None or A08i is not None):
            if A48i is not None:
                A48i = A48i[ok]
            if A08i is not None:
                A08i = A08i[ok]
            fig = mc.ar_sol_hist(ts[ok], A48i, A08i)
            results['arvi_hist'] = fig

    return results


#===========================================================================
# U-Pb isochron ages
#===========================================================================

def isochron_age(t, fit, A, sA, init=(True, True), trials=50_000,
            dc_errors=False, negative_ar=True, negative_ages=True,
            hist=(False, False), age_type='iso-Pb6U8'):
    """
    Propogate disequilibrium isochron age uncertainties using Monte Carlo
    approach.
    """
    assert age_type in ('iso-Pb6U8', 'iso-Pb7U5')
    flags = np.zeros(trials, dtype='uint8')

    theta, flags = mc.draw_theta(fit, trials, flags)

    if age_type == 'iso-Pb6U8':
        pA, flags = draw_ar(A, sA, trials, flags, positive_only=not negative_ar)
        DC, flags = draw_decay_const(trials, flags, dc_errors=dc_errors)
        BC = ludwig.bateman(DC, series='238U')
    else:
        pA, flags = draw_ar(A, sA, trials, flags, series='235U',
                            positive_only=not negative_ar)
        DC, flags = draw_decay_const(trials, flags, dc_errors=dc_errors, series='235U')
        BC = ludwig.bateman(DC, series='235U')
    # compile args for numercial age routine:
    args = (theta[1], pA, DC, BC)

    # run vectorised Newton routine:
    fmin, dfmin = minimise.isochron(t0=t, age_type=age_type, init=init)
    ts, c, zd = optimize.newton(fmin, np.ones(trials) * t, dfmin, args=args,
                                full_output=True)
    flags = mc.check_ages(ts, c, flags, negative_ages=negative_ages)

    # back-calculate activity ratios
    if age_type == 'iso-Pb6U8':
        A48i, A08i = useries.Ai_solutions(ts, pA, init, DC)
        if not negative_ar:
            if A48i is not None:
                flags = mc.update_flags(flags, A48i < 0., mc.NEGATIVE_AR_SOL)
            if A08i is not None:
                flags = mc.update_flags(flags, A08i < 0., mc.NEGATIVE_AR_SOL)

    ok = flags == 0

    if sum(ok) == 0:
        raise RuntimeError('all Monte Carlo simulation trials failed')

    age_95ci = np.quantile(ts[ok], (0.025, 0.975))

    # compute results
    results = {
        'age_type': f'{age_type} isochron age',
        'age_1s': np.nanstd(ts[ok]),
        'age_95ci': age_95ci,
        'age_95pm': np.nanmean((t - age_95ci[0], age_95ci[1] - t)),
        'mean_age': np.nanmean(ts[ok]),
        'median_age': np.nanmedian(ts[ok]),
        'trials': trials,
        'fails': np.sum(flags != 0),
        'not_converged': np.sum(flags == mc.NON_CONVERGENCE),
        'negative_ages': np.sum(flags == mc.NEGATIVE_AGE),
        'negative_ar': np.sum(flags == mc.NEGATIVE_AR_SIM),
        'negative_ar_soln': np.sum(flags == mc.NEGATIVE_AR_SOL)
    }

    if age_type == 'iso-Pb6U8':
        if A48i is not None:
            results['mean_[234U/238U]i'] = np.nanmean(A48i[ok])
            results['median_[234U/238U]i'] = np.nanmedian(A48i[ok])
            results['[234U/238U]i_1sd'] = np.nanstd(A48i[ok])
            results['[234U/238U]i_95ci'] = np.quantile(A48i[ok], (0.025, 0.975))
            results['cor_[234U/238U]i_t'] = \
                np.corrcoef(np.row_stack((ts[ok], A48i[ok])))[0, 1]

        if A08i is not None:
            results['mean_init_[230Th/238U]'] = np.nanmean(A08i[ok])
            results['median_init_[230Th/238U]'] = np.nanmedian(A08i[ok])
            results['[230Th/238U]i_1sd'] = np.nanstd(A08i[ok])
            results['[230Th/238U]i_95ci'] = np.quantile(A08i[ok], (0.025, 0.975))
            results['cor_[230Th/238U]i_t'] = \
                np.corrcoef(np.row_stack((ts[ok], A08i[ok])))[0, 1]

    # compile plots
    if any(hist):
        if hist[0]:
            a, b = theta
            fig = mc.age_hist(ts[ok], age_type, a[ok], b[ok])
            results['age_hist'] = fig
        if hist[1]:
            A238 = None if age_type != 'iso-Pb6U8' else pA.T[ok].T
            A235 = None if age_type != 'iso-Pb7U5' else pA.T[ok].T
            fig = mc.ar_hist(A238, A235, init=init)
            results['ar_hist'] = fig
        if age_type == 'iso-Pb6U8':
            if hist[1] and (not init[0] or not init[1]):
                if not init[0]:
                    A48i = A48i[ok]
                if not init[1]:
                    A08i = A08i[ok]
                fig = mc.ar_sol_hist(ts[ok], A48i, A08i)
                results['arvi_hist'] = fig

    return results


#===========================================================================
# Forced-concordance [234U/238U] errors
#===========================================================================

def fc_A48i(t57, A48i, fit_57, fit_86, A, sA, init, trials=50_000,
            negative_ar=True, negative_ages=True, hist=(0,0,0)):
    """
    Forced-concordance initial [234U/238U] value errors.
    """
    assert init[1]
    flags = np.zeros(trials)
    theta57 = cfg.rng.multivariate_normal(fit_57['theta'], fit_57['covtheta'], trials)
    theta86 = cfg.rng.multivariate_normal(fit_86['theta'], fit_86['covtheta'], trials)
    
    pA = np.empty((4, trials))
    pA[-1, :], flags = draw_ar(A[-1], sA[-1], trials, flags, series='235U',
                               positive_only=not negative_ar)
    pA[:3, :], flags = draw_ar(A[:-1], sA[:-1], trials, flags, positive_only=not negative_ar)

    DC8, flags = draw_decay_const(trials, flags, dc_errors=False)
    DC5, flags = draw_decay_const(trials, flags, dc_errors=False, series='235U')
    BC8 = ludwig.bateman(DC8, series='238U')
    BC5 = ludwig.bateman(DC8, series='235U')

    args57 = (theta57[:, 1], pA[-1], DC5, BC5)
    
    # run vectorised Newton routine to get iso-57 ages
    fmin, dfmin = minimise.isochron(t0=1.0, age_type='iso-Pb7U5')
    ts, c, zd = optimize.newton(fmin, np.ones(trials) * t57, dfmin, args=args57,
                                full_output=True)
    flags = mc.check_ages(ts, c, flags, negative_ages=negative_ages)

    # run vectorised Newton routine to get initial [234U/238U] values:
    fmin, dfmin = minimise.concordant_A48()
    A48i_s, c, zd = optimize.newton(fmin, np.full(trials, A48i), dfmin,
            args=(ts, theta86[:, 1], pA[:3], DC8, BC8), full_output=True)
    flags = mc.check_ages(A48i_s, c, flags, negative_ages=negative_ar) # pretend ar48i solutions are ages

    ok = (flags == 0)
    if sum(ok) == 0:
        raise RuntimeError('all Monte Carlo simulation trials failed')
    age_95ci = np.quantile(ts[ok], (0.025, 0.975))
    A48i_95ci = np.quantile(A48i_s[ok], (0.025, 0.975))

    # compute results
    results = {
        'age_type': 'forced-concordance [234U/238U]_i',
        '207Pb_age_1sd': np.nanstd(ts[ok]),
        '207Pb_age_95ci': age_95ci,
        '207Pb_age_95pm': [np.nanmean((age_95ci[1] - t57, t57 - age_95ci[0]))],
        'mean_207Pb_age': np.nanmean(ts[ok]),
        'median_207Pb_age': np.nanmedian(ts[ok]),
        '[234U/238U]i_1sd': np.nanstd([A48i_s[ok]]),
        '[234U/238U]i_95ci': A48i_95ci,
        '[234U/238U]i_95pm': [np.nanmean((A48i_95ci[1] - A48i, A48i - A48i_95ci[0]))],
        'mean_[234U/238U]i': np.nanmean([A48i_s[ok]]),
        'median_[234U/238U]i': np.nanmedian([A48i_s[ok]]),
        'trials': trials,
        'fails': np.sum(flags != 0),
        'not_converged': np.sum(flags == mc.NON_CONVERGENCE),
        'negative_ages': np.sum(flags == mc.NEGATIVE_AGE),
        'negative_ar': np.sum(flags == mc.NEGATIVE_AR_SIM)
    }

    if any(hist):
        if hist[0]:
            a, b = np.transpose(theta57)
            fig = mc.age_hist(ts[ok], 'iso-Pb7U5', a[ok], b[ok])
            results['age_hist'] = fig
        if hist[1]:
            A238 = [A48i_s[ok], *pA[1:3].T[ok].T]
            A235 = pA[-1].T[ok].T
            fig = mc.ar_hist(A238, A235)
            results['ar_hist'] = fig

    return results


#===========================================================================
# Pb/U ages
#===========================================================================

def pbu_age(t, x, sx, trials=10_000, DThU=None, DThU_1s=None, DPaU=None,
            DPaU_1s=None, ThU_min=None, ThU_melt=None, ThU_melt_1s=None,
            DThU_const=True, method='Ludwig', age_type='Pb6U8',
            negative_ages=True, negative_ar=True, rand=False,
            hist=(False, False)):
    """
    Propagate disequilibrium Pb/U age uncertainties using Monte Carlo approach.

    If rand is True, will also simulate
    random only age uncertainties. Expects initial activity ratio values and
    does not yet account for activity ratio errors.

    Parameters
    ----------
    x :
        Pb/U ratios
    sx :
        Pb/U ratio errors (1s, abs.)

    Notes
    -----
    Does not account for decay constant errors.
    negative_ar option not yet implemented

    """
    assert age_type in ('Pb6U8', 'Pb7U5')
    assert method in ('Ludwig', 'Guillong')
    assert not any([np.isnan(t) for t in t])       # nan ages should be removed
    if negative_ar:
        warnings.warn('negative_ar option not yet implemented in Monte Carlo simulation')

    A48_eq = 1.0
    A68_eq = 1.0

    n = x.size
    flags = np.zeros((trials, n), dtype='uint8')
    ts = np.empty((trials, n))                  # pre-allocate ages array
    if rand:
        flagr = np.zeros((trials, n), dtype='uint8')
        tr = np.empty((trials, n))             # random only uncertainties

    # simulate fXU values
    if age_type == 'Pb6U8':
        if DThU_const:
            fThU = np.random.normal(DThU, DThU_1s, trials)
            fXU = np.broadcast_to(fThU.reshape(trials, 1), (trials, n))
        else:
            ThU_melt_s = np.random.normal(ThU_melt, ThU_melt_1s, (trials, 1))
            fXU = np.broadcast_to(ThU_min, (trials, n)) / ThU_melt_s

    elif age_type == 'Pb7U5':
        fPaU = np.random.normal(DPaU, DPaU_1s, trials)
        fXU = np.broadcast_to(fPaU.reshape(trials, 1), (trials, n))

    if method == 'Ludwig':
        fmin, dfmin = minimise.pbu(age_type=age_type, t0=np.nanmean(t))
        if age_type == 'Pb6U8':
            DC = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
            BC = ludwig.bateman(DC)
        else:
            DC = (cfg.lam235, cfg.lam231, cfg.lam227)
            BC = ludwig.bateman(DC, series='235U')

    elif method == 'Guillong':
        fmin, dfmin = minimise.guillong(t0=np.nanmean(t))

    # Do rand + systematic age simulation for each age data point:
    for i in range(n):
        xs = cfg.rng.normal(x[i], sx[i], trials)

        # simulate activity ratios
        if method == 'Ludwig':
            if age_type == 'Pb6U8':
                args = (xs, (A48_eq, fXU[:, i], A68_eq), DC, BC)
            else:
                args = (xs, (fXU[:, i]), DC, BC)
        elif method == 'Guillong':
            if age_type == 'Pb6U8':
                args = (xs, fXU[:, i], cfg.lam238, cfg.lam230)
            else:
                args = (xs, fXU[:, i], cfg.lam235, cfg.lam231)

        t0 = np.full(trials, t[i])
        r, c, zd = optimize.newton(fmin, t0, dfmin, args=args, full_output=True)
        flags[:, i] = mc.check_ages(r, np.logical_or(c, ~zd), flags[:, i],
                negative_ages=negative_ages)
        ts[:, i] = r

        # Compute random only errors (for plotting purposes)
        if rand:
            if age_type == 'Pb6U8':
                if DThU_const:
                    rfXU = DThU
                else:
                    rfXU = ThU_min[i] / ThU_melt
            elif age_type == 'Pb7U5':
                rfXU = np.atleast_1d(DPaU)

            if method == 'Ludwig':
                if age_type == 'Pb6U8':
                    DC = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
                    BC = ludwig.bateman(DC)
                    args = (xs, [A48_eq, rfXU, A68_eq], DC, BC)
                else:
                    DC = (cfg.lam235, cfg.lam231)
                    BC = ludwig.bateman(DC, series='235U')
                    args = (xs, rfXU, DC, BC)

            elif method == 'Guillong':
                if age_type == 'Pb6U8':
                    args = (xs, rfXU, cfg.lam238, cfg.lam230)
                else:
                    args = (xs, rfXU, cfg.lam235, cfg.lam231)

            r, c, zd = optimize.newton(fmin, t0, dfmin, args=args, full_output=True)
            flagr[:, i] = mc.check_ages(r, np.logical_or(c, ~zd), flags[:, i],
                                     negative_ages=negative_ages)
            tr[:, i] = r

    ok = np.all(flags == 0, axis=1)
    age_95ci = [np.quantile(t[ok], (0.025, 0.975)) for t in ts.T]
    # compile results
    results = {
        'age_type': age_type,
        'age_1s': [np.nanstd(t[ok]) for t in ts.T],
        'age_95pm': [np.nanmean((t - m, p - t)) for m, p in age_95ci],
        'age_95ci': age_95ci,
        'mean_age': [np.nanmean(t[ok]) for t in ts.T],
        'median_age': [np.nanmedian(t[ok]) for t in ts.T],
        'cov_t': np.cov(np.transpose(ts[ok])),
        'trials': trials,
        'fails': np.sum(flags != 0, axis=1),
        'not_converged': np.sum(flags == mc.NON_CONVERGENCE, axis=1),
        'negative_ages': np.sum(flags == mc.NEGATIVE_AGE, axis=1),
        'negative_distr_coeff': np.sum(flags == mc.NEGATIVE_AR_SIM, axis=1)
    }
    if rand:
        okr = np.all(flagr == 0, axis=1)
        rand_age_95ci = [np.quantile(t[okr], (0.025, 0.975)) for t in tr.T]
        results['age_rand_1s'] = [np.nanstd(t[okr]) for t in tr.T]
        results['age_rand_95ci'] = rand_age_95ci
        results['age_rand_95pm'] = [np.nanmean((t - m, p - t)) for m, p in rand_age_95ci]
        results['mean_rand_age'] = [np.nanmean(t[okr]) for t in tr.T]
        results['median_rand_age'] = [np.nanmedian(t[okr]) for t in tr.T]
        results['rand_fails'] = np.sum(flagr != 0, axis=1)
        results['rand_not_converged'] = np.sum(flagr == mc.NON_CONVERGENCE, axis=1)
        results['rand_negative_ages'] = np.sum(flagr == mc.NEGATIVE_AGE, axis=1)
        results['rand_distr_coeff'] = np.sum(flagr == mc.NEGATIVE_AR_SIM, axis=1)

    if any(hist):
        raise ValueError('not yet coded')

    return results


def mod207_age(t, x, sx, y, sy, r_xy, Pb76, Pb76se, ThU_min=None,
               ThU_melt=None, ThU_melt_1s=None, DThU=None, DThU_1s = None,
               DPaU = None, DPaU_1s = None, DThU_const = True, trials=10_000,
               method='Ludwig', negative_ages=True, negative_ar=True,
               rand=False, hist=(False, False)):
    """
    Compute Pb/U age uncertainties using Monte Carlo approach. Does not
    account for decay constant errors. If rand is True, will also simulate
    random only age uncertainties. Expects initial activity ratio values and
    does not yet account for activity ratio errors.

    U and decay constant errors not included.
    """
    assert method in ('Ludwig', 'Sakata')
    assert not any([np.isnan(t) for t in t])       # nan ages should be removed
    if negative_ar:
        warnings.warn('negative_ar option not yet implemented in Monte Carlo simulation')

    A48_eq = 1.0
    A68_eq = 1.0
    init = (True, True)

    n = x.size
    flags = np.zeros((trials, n), dtype='uint8')
    ts = np.empty((trials, n))                  # pre-allocate ages array
    if rand:
        flagr = np.zeros((trials, n), dtype='uint8')
        tr = np.empty((trials, n))             # random only uncertainties

    # simulate fXU values
    if DThU_const:
        fThU = np.random.normal(DThU, DThU_1s, trials)
        fThU = np.broadcast_to(fThU.reshape(trials, 1), (trials, n))
    else:
        ThU_melt_s = np.random.normal(ThU_melt, ThU_melt_1s, (trials, 1))
        fThU = np.broadcast_to(ThU_min, (trials, n)) / ThU_melt_s

    fPaU = np.random.normal(DPaU, DPaU_1s, trials)
    fPaU = np.broadcast_to(fPaU.reshape(trials, 1), (trials, n))

    flags = mc.update_flags(flags, fThU < 0., mc.NEGATIVE_AR_SIM)
    flags = mc.update_flags(flags, fPaU < 0., mc.NEGATIVE_AR_SIM)

    # simulate U and Pb76 values
    Pb76 = cfg.rng.normal(Pb76, Pb76se, trials)
    U = cfg.U

    if method == 'Ludwig':
        fmin, dfmin = minimise.mod207(t0=np.nanmean(t))
        DC8 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        DC5 = (cfg.lam235, cfg.lam231)
        BC8 = ludwig.bateman(DC8)
        BC5 = ludwig.bateman(DC5, series='235U')

    elif method == 'Sakata':
        fmin, dfmin = minimise.sakata(t0=np.nanmean(t))

    # Initialise arrays for storing histograms:
    if hist[0]: age_hist = []
    if hist[1]: arv_hist = []

    # Do random + systematic age simulation for each age data point:
    for i in range(n):
        mu = np.array([x[i], y[i]])
        cov = np.array([[sx[i] ** 2, r_xy[i] * sx[i] * sy[i]],
                        [r_xy[i] * sx[i] * sy[i], sy[i] ** 2]])
        dp = cfg.rng.multivariate_normal(mu, cov, trials)
        xs, ys = dp[:, 0], dp[:, 1]

        # simulate activity ratios
        if method == 'Ludwig':
            args = (xs, ys, [A48_eq, fThU[:, i], A68_eq, fPaU[:, i]],
                    DC8, DC5, BC8, BC5, U, Pb76)
        elif method == 'Sakata':
            args = (xs, ys, fThU[:, i], fPaU[:, i], cfg.lam238, cfg.lam230,
                    cfg.lam235, cfg.lam231, U, Pb76)

        t0 = np.full(trials, t[i])
        r, c, zd = optimize.newton(fmin, t0, dfmin, args=args, full_output=True)
        flags[:, i] = mc.check_ages(r, np.logical_or(c, ~zd), flags[:, i],
                                    negative_ages=negative_ages)
        ts[:, i] = r

        # TODO: check for reverse discordance?

        # make histograms
        if hist[0]:
            fig = mc.age_hist(ts[:, i], flags[:, i], 'mod207Pb', x=xs, y=ys)
            age_hist.append(fig)
        if hist[1]:
            warnings.warn('ar hist option not yet implemented')

        # Random errors only (e.g. for plotting)
        if rand:
            if DThU_const:
                rfThU = DThU
                rfThU = np.broadcast_to(rfThU, (n,))
            else:
                rfThU = ThU_min / ThU_melt

            rfPaU = np.atleast_1d(DPaU)

            # no need to check for negative distr. coefficients here since
            # they have no error attached

            if method == 'Ludwig':
                args = (xs, ys, [A48_eq, rfThU[i], A68_eq, rfPaU],
                        DC8, DC5, BC8, BC5, U, Pb76)
            elif method == 'Sakata':
                args = (xs, ys, rfThU[i], rfPaU, cfg.lam238, cfg.lam230,
                        cfg.lam235, cfg.lam231, U, Pb76)

            r, c, zd = optimize.newton(fmin, t0, dfmin, args=args, full_output=True)
            flagr[:, i] = mc.check_ages(r, np.logical_or(c, ~zd), flags[:, i],
                                        negative_ages=negative_ages)
            tr[:, i] = r



    # compile results
    ok = np.all(flags == 0, axis=1)
    results = {
        'age_type': 'Modified 207Pb',
        'method': method,
        'age_1s': [np.nanstd(t[ok]) for t in ts.T],
        'age_95ci': [np.quantile(t[ok], (0.025, 0.975)) for t in ts.T],
        'mean_age': [np.nanmean(t[ok]) for t in ts.T],
        'median_age': [np.nanmedian(t[ok]) for t in ts.T],
        'cov_t': np.cov(np.transpose(ts[ok])),
        'trials': trials,
        'fails': np.sum(flags != 0, axis=1),
        'not_converged': np.sum(flags == mc.NON_CONVERGENCE, axis=1),
        'negative_ages': np.sum(flags == mc.NEGATIVE_AGE, axis=1),
        'negative_distr_coeff': np.sum(flags == mc.NEGATIVE_AR_SIM, axis=1),
    }
    if rand:
        okr = np.all(flagr == 0, axis=1)
        results['age_rand_1s'] = [np.nanstd(t[okr]) for t in tr.T]
        results['age_rand_95ci'] = [np.quantile(t[okr], (0.025, 0.975)) for t in tr.T]
        results['mean_rand_age'] = [np.nanmean(t[okr]) for t in tr.T]
        results['median_rand_age'] = [np.nanmedian(t[okr]) for t in tr.T]
        results['rand_fails'] = np.sum(flagr != 0, axis=1)
        results['rand_not_converged'] = np.sum(flagr == mc.NON_CONVERGENCE, axis=1)
        results['rand_negative_ages'] = np.sum(flagr == mc.NEGATIVE_AGE, axis=1)
        results['rand_distr_coeff'] = np.sum(flagr == mc.NEGATIVE_AR_SIM, axis=1)

    if hist[0]:
        results['age_hist'] = age_hist
    if hist[1]:
        results['ar_hist'] = None

    return results


#===============================================
# Simulate constants and activity ratios
#===============================================

def draw_ar(A, sA, trials, flags=None, positive_only=False,
            series='238U'):
    """  
    Draw random activity ratio values.
    
    """
    assert series in ('238U', '235U')
    
    if series == '238U':
        A48 = cfg.rng.normal(A[0], sA[0], trials)
        A08 = cfg.rng.normal(A[1], sA[1], trials)
        A68 = cfg.rng.normal(A[2], sA[2], trials)
        if flags is not None and positive_only:
            for x in (A48, A08, A68):
                flags = mc.update_flags(flags, x < 0., mc.NEGATIVE_AR_SIM)
        pA = np.array((A48, A08, A68))
    
    else:
        pA = cfg.rng.normal(A, sA, trials)
        if positive_only:
            flags = mc.update_flags(flags, pA < 0., mc.NEGATIVE_AR_SIM)
    
    return pA, flags


def draw_decay_const(trials, flags, dc_errors=False, series='238U', cor=False):
    """  
    Draw random decay constant values.

    """
    assert series in ('238U', '235U')
    
    if series == '238U':
        # note: effects of error correlations are probably always negligible ?
        if dc_errors:
            if cor:
                cov_84 = cfg.cor_238_234 * (cfg.s238 * cfg.s234)
                cov_80 = cfg.cor_238_230 * (cfg.s238 * cfg.s230)
                cov_40 = cfg.cor_234_230 * (cfg.s234 * cfg.s230)
    
                V = np.array([[cfg.s238 ** 2, cov_84, cov_80, 0.],
                              [cov_84, cfg.s234 ** 2, cov_40, 0.],
                              [cov_80, cov_40, cfg.s230 ** 2, 0.],
                              [0., 0., 0., cfg.s226 ** 2]])
    
                DC = cfg.rng.multivariate_normal(
                    (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226), V, trials
                )

            else:
                DC = cfg.rng.normal((cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226),
                                    (cfg.s238, cfg.s234, cfg.s230, cfg.s226),
                                    (trials, 4))
            DC = np.transpose(DC)
        else:
            DC = np.array((cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226))
    
    else:
        if dc_errors:
            lam235 = cfg.rng.normal(cfg.lam235, cfg.s235, trials)
            lam231 = cfg.rng.normal(cfg.lam231, cfg.s231, trials)
            DC = np.array((lam235, lam231))
        else:
            DC = np.array((cfg.lam235, cfg.lam231))
    
    return DC, flags


#===============================================
# Plots
#===============================================

def diseq_intercept_plot(ts, fit, A8, A15, DC8, DC5, BC8, BC5, U, flags,
            init=(True, True), dp=None, dc_errors=False, u_errors=False,
            diagram='tw', xlim=(None, None), ylim=(None, None),
            intercept_points=True, intercept_ellipse=False):
    """
    Plot simulated disequilibrium concordia intercept points.
    """
    fig, ax = plt.subplots(**cfg.fig_kw, subplot_kw=cfg.subplot_kw)
    ax = fig.axes[0]
    ok = flags == 0

    # Filter out failed iteration values:
    if dc_errors:
        DC8 = np.transpose(np.transpose(DC8)[ok])
        DC5 = np.transpose(np.transpose(DC5)[ok])
        BC8 = np.transpose(np.transpose(BC8)[ok])
        BC5 = np.transpose(np.transpose(BC5)[ok])
    if u_errors:
        U = U[ok]

    A8 = np.transpose(np.transpose(A8)[ok])
    A15 = A15[ok]

    x = 1. / ludwig.f(ts[ok], A8, DC8, BC8, init=init)
    y = ludwig.g(ts[ok], A15, DC5, BC5) * x / U
    mc.intercept_plot_axis_limits(ax, x, y, diagram=diagram)

    if intercept_points:
        ax.plot(x, y, **cfg.conc_intercept_markers_kw)

    if intercept_ellipse:
        cov = np.cov(x, y)
        e = plotting.confidence_ellipse2(ax, np.nanmean(x), np.nanmean(y), cov,
                     **cfg.conc_intercept_ellipse_kw)
        ax.add_patch(e)

    # Plot data ellipses if dp given:
    if dp is not None:
        plotting.plot_dp(ax, *dp, reset_axis_limits=False)

    # Label axes and apply plot settings etc.
    plotting.apply_plot_settings(fig, plot_type='intercept', diagram=diagram,
                xlim=xlim, ylim=ylim, tight_layout=True)
    # Plot regression line and envelope
    plotting.rline(ax, fit['theta'])
    plotting.renv(ax, fit)

    return fig

