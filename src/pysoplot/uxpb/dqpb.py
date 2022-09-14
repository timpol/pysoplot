"""
Functions and routines for computing disequilibrium U-Pb ages

References
----------
.. [Engel2019]
    Engel, J., Woodhead, J., Hellstrom, J., Maas, R., Drysdale, R., Ford, D.,
    2019. Corrections for initial isotopic disequilibrium in the speleothem
    U-Pb dating method. Quaternary Geochronology 54, 101009.
    https://doi.org/10.1016/j.quageo.2019.101009
.. [Guillong2014]
    Guillong, M., von Quadt, A., Sakata, S., Peytcheva, I., Bachmann, O., 2014.
    LA-ICP-MS Pb-U dating of young zircons from the Kos–Nisyros volcanic centre,
    SE aegean arc. Journal of Analytical Atomic Spectrometry 29, 963–970.
    https://doi.org/10.1039/C4JA00009A
.. [Ludwig1977]
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.
.. [Sakata2017]
    Sakata, S., Hirakawa, S., Iwano, H., Danhara, T., Guillong, M., Hirata, T.,
    2017. A new approach for constraining the magnitude of initial
    disequilibrium in Quaternary zircons by coupled uranium and thorium decay
    series dating. Quaternary Geochronology 38, 1–12.
    https://doi.org/10.1016/j.quageo.2016.11.002

"""

import numpy as np
from scipy import optimize

from . import ludwig
from . import minimise
from . import dqmc

from .. import useries
from .. import wtd_average
from .. import cfg
from .. import plotting
from .. import exceptions


exp = np.exp
log = np.log
nan = np.nan


#===========================================
# Age calculation routines
#===========================================

def concint_age(fit, A, sA, init, t0, diagram='tw', dc_errors=False, trials=50_000,
        u_errors=False, negative_ar=True, negative_ages=True,
        intercept_plot=True, hist=(False, False), conc_kw=None,
        intercept_plot_kw=None, A48i_lim=(0., 20.), A08i_lim=(0., 10.),
        age_lim=(0., 20.)):
    """
    Compute disequilibrium U-Pb concordia-intercept age and age uncertainties using
    Monte Carlo simulation. Optionally, produce a plot of the concordia-intercept.
    
    Parameters
    ----------
    fit : dict
        linear regression fit parameters
    A : array-like
        one-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U]
    sA : array-like
        one-dimensional array of activity ratio value uncertainties given 
        as 1 sigma absolute and arranged in the same order as A
    init : array-like
        two-element list of boolean values, the first is True if [234U/238U]
        is an initial value and False if a present-day value, the second is True
        if [230Th/238U] is an initial value and False if a present-day value
    t0 : float
        initial guess for numerical age solving (the equilibrium age is
        typically good enough)
    
    """
    assert type(init[0]) == bool and type(init[1] == bool)

    if intercept_plot_kw is None:
        intercept_plot_kw = {}
    if conc_kw is None:
        conc_kw = {}

    # Compute age:
    a, b = fit['theta']
    
    # If a present-day activity ratio given, check for all possible intercept age
    # solutions within given age limits.
    if any(init):
        r = concint_multiple(a, b, A, init, t0, age_lim=age_lim,  t_step=1e-5,
                             A48i_lim=A48i_lim, A08i_lim=A08i_lim)
        ages, A48i, A08i = r
        t = ages[0]
    else:
        t = concint(a, b, A, init, t0)
        if t < 0:
            raise RuntimeError(f'negative disequilibrium age solution: {t} Ma')

    # Get initial acitivity rario solutions:
    A48i, A08i = useries.Ai_solutions(t, A[:-1], init, (cfg.lam238,
                            cfg.lam234, cfg.lam230))

    # Compute age uncertainties:
    mc = dqmc.concint_age(t, fit, A, sA, init, trials=trials, dc_errors=dc_errors,
                  u_errors=u_errors, negative_ar=negative_ar,
                  negative_ages=negative_ages,hist=hist,
                  intercept_plot=intercept_plot, conc_kw=conc_kw,
                  intercept_plot_kw=intercept_plot_kw)

    results = {
        'age_type': 'concordia-intercept',
        'diagram': diagram,
        'age': t,
        'age_1s': mc['age_1s'],
        'age_95ci': mc['age_95ci'],
        'age_95pm': np.mean((t - mc['age_95ci'][0], mc['age_95ci'][1] - t)),
        '[234U/238U]i': A48i,
        '[230Th/238U]i': A08i,
        'mc': mc,
    }

    if not init[0]:
        results['[234U/238U]i_95ci'] = mc['[234U/238U]i_95ci']
        results['[234U/238U]i_95pm'] = np.mean((t - mc['[234U/238U]i_95ci'][0],
                                      mc['[234U/238U]i_95ci'][1] - t))
    if not init[1]:
        results['[230Th/238U]i_95ci'] = mc['[230Th/238U]i_95ci']
        results['[230Th/238U]i_95pm'] = np.mean((t - mc['[230Th/238U]i_95ci'][0],
                                        mc['[230Th/238U]i_95ci'][1] - t))

    return results


def isochron_age(fit, A, sA, t0, init=(True, True), age_type='iso-Pb6U8',
                 norm_isotope='204Pb', hist=(False, False),
                 trials=50_000, dc_errors=False, negative_ar=True,
                 negative_ages=True):
    """
    Compute disequilibrium 238U-206Pb or 235U-207Pb isochron age and age
    uncertainties using Monte Carlo methods.

    Parameters
    ----------
    fit : dict
        linear regression fit parameters
    A : array-like
        one-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U]
    sA : array-like
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A
    init : array-like
        two-element list of boolean values, the first is True if [234U/238U]
        is an initial value and False if a present-day value, the second is True
        if [230Th/238U] is an initial value and False if a present-day value
    t0 : float
        initial guess for numerical age solving (the equilibrium age is
        typically good enough)

    """
    assert age_type in ('iso-Pb6U8', 'iso-Pb7U5')

    if age_type == 'iso-Pb6U8':
        assert len(A) == 3
        assert len(sA) == 3
    else:
        assert type(A) in (float, int)
        assert type(sA) in (float, int)
    
    # Compute age:
    a, b = fit['theta']
    t = isochron(b, A, t0, age_type, init=init)
    if t < 0:
        raise RuntimeError(f'negative disequilibrium age solution: {t} Ma')

    # Back-calculate activity ratios:
    if age_type == 'iso-Pb6U8':
        DC = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        A48i, A08i = useries.Ai_solutions(t, A, init, DC)
    else:
        A48i, A08i = (None, None)

    # Compute age uncertainties:
    mc = dqmc.isochron_age(t, fit, A, sA, init=init, trials=trials,
                        negative_ar=negative_ar, negative_ages=negative_ages,
                        hist=hist, age_type=age_type, dc_errors=dc_errors)

    results = {
        'age_type': f'{age_type} isochron',
        'norm_isotope': norm_isotope,
        'age': t,
        'age_1s': mc['age_1s'],
        'age_95ci': mc['age_95ci'],
        'age_95pm': mc['age_95pm'],
        '[234U/238U]i': A48i,
        '[230Th/238U]i': A08i,
        'mc': mc,
    }

    return results


def pbu_age(x, sx, t0, ThU_min=None, ThU_melt=None, ThU_melt_1s=None,
        DThU=None, DThU_1s=None, DPaU=None, DPaU_1s=None, DThU_const=True,
        age_type='Pb6U8', wav=False, trials=30_000, wav_method='sp',
        method='Ludwig', wav_plot_prefix='ka', sorted=False, rand=True,
        cov=True, hist=(False, False), negative_ages=True, negative_ar=True,
        ylim=(None, None), dp_labels=None):
    """
    Compute disequilibrium 206Pb/238U or 207Pb/235U ages, and optionally compute
    a weighted average age.

    Parameters
    ----------
    x : np.ndarray
        one-dimensional array of Pb/U ratios
    sx : np.ndarray
        one-dimensional array of Pb/U ratio errors (1s, abs.)
    t0 : np.ndarray
        initial guesses for disequilibrium age solutions
    wav : bool
        compute weighted average age
    wav_type : str
        wtd. average type
    method : str
        method of age calculation
    rand : bool
        plot random age uncerts on wtd. average plot
    cov : bool
        account for age covariance structure in wtd. average

    """
    assert age_type in ('Pb6U8', 'Pb7U5')
    assert method in ('Ludwig', 'Guillong')
    n = x.shape[0]
    assert sx.shape[0] == n

    if age_type == 'Pb6U8':
        if DThU_const:
            DThU = float(DThU)
            DThU_1s = float(DThU_1s)
        else:
            assert len(ThU_min) == n
            ThU_melt = float(ThU_melt)
            ThU_melt_1s = float(ThU_melt_1s)
    else:
        DPaU = float(DPaU)
        DPaU_1s = float(DPaU_1s)

    if np.atleast_1d(t0).size:
        t0 = np.broadcast_to(t0, (n,))

    # equilibrium activity ratio values
    A48_eq = cfg.A48_eq
    A68_eq = cfg.A08_eq

    if age_type == 'Pb6U8':
        if DThU_const:
            fXU = np.broadcast_to(DThU, (n,))
        else:
            XU_melt = ThU_melt
            XU_min = ThU_min
            fXU = XU_min / XU_melt
    else:
        # only type 1 values for Pa/U
        DPaU = DPaU
        fXU = np.broadcast_to(DPaU, (n,))

    # Comupte ages:
    t = np.empty(n)
    for i in range(n):
        if method == 'Guillong':
            t[i] = guillong(x[i], fXU[i], t0[i], age_type)
        else:
            if age_type == 'Pb6U8':
                t[i] = pbu(x[i], [A48_eq, fXU[i], A68_eq], t0[i], age_type)
            else:
                t[i] = pbu(x[i], fXU[i], t0[i], age_type)

    if np.all(np.isnan(t)):
        raise ValueError('all age solutions failed to converge')

    # Compute age uncertainties:
    mc = dqmc.pbu_age(
        t, x, sx, age_type=age_type, ThU_min=ThU_min,
        ThU_melt=ThU_melt, ThU_melt_1s=ThU_melt_1s, DThU=DThU,
        DThU_1s=DThU_1s, DPaU=DPaU, DPaU_1s=DPaU_1s,
        DThU_const=DThU_const, method=method, negative_ages=negative_ages,
        negative_ar=negative_ar, rand=rand, trials=trials, hist=hist,
    )

    results = {
        'age_type': age_type,
        'age': t,
        'age_1s': mc['age_1s'],
        'age_95ci': mc['age_95ci'],
        'age_95pm': mc['age_95pm'],
        'cov_t': mc['cov_t'],
        'mc': mc
    }

    rand_pm = None

    # compute wtd. average
    if wav:
        cov_t = mc['cov_t']
        if not cov:
            cov_t = np.diag(np.diag(cov_t))
        elif rand:
            results['age_rand_95ci'] = mc['age_rand_95ci']
            results['age_rand_95pm'] = mc['age_rand_95pm']
            rand_pm = np.asarray(results['age_rand_95pm'])
        if wav_method in ('ca'):
            wav_results = wtd_average.classical_wav(t, cov=cov_t, method=wav_method)
        elif wav_method in ('ra', 'rs'):
            wav_results = wtd_average.robust_wav(t, cov=cov_t, method=wav_method)
        else:
            raise ValueError('wav method not recognised')

        results['wav_age'] = wav_results['ave']
        results['wav_age_95pm'] = wav_results['ave_95pm']
        results['wav'] = wav_results

        # Make wav plot:
        age_mult = 1. if wav_plot_prefix == 'Ma' else 1000.
        fig = plotting.wav_plot(t, np.asarray(results['age_95pm']), wav_results['ave'],
                    wav_results['ave_95pm'], rand_pm=rand_pm,
                    sorted=sorted, ylim=ylim, x_multiplier=age_mult,
                    dp_labels=dp_labels)
        ax = fig.get_axes()[0]
        if age_type == 'Pb6U8':
            ax.set_ylabel(f'$^{{206}}$Pb/$^{{238}}$U age ({wav_plot_prefix})')
        else:
            ax.set_ylabel(f'$^{{207}}$Pb/$^{{235}}$U age ({wav_plot_prefix})')
        results['fig_wav'] = fig

    return results


def mod207_age(dp, Pb76, Pb76se, t0, ThU_min=None, ThU_melt=None, ThU_melt_1s=None,
            DThU=None, DThU_1s=None, DPaU=None, DPaU_1s=None, DThU_const=True,
            wav=False, wav_method='sp', method='Ludwig',
            sorted=False, rand=True, cov=True, negative_ages=True,
            negative_ar=True, ylim=(None, None), hist=(False, False),
            wav_plot_prefix='Ma', trials=50_000, dp_labels=None):
    """
    Compute modified 207Pb ages and optionally compute a weighted average age.

    Parameters
    ----------
    dp : np.ndarray
        5 x n array of data points
    t0 : array-like
        initial disequilibrium age guesses
    wav : bool
        compute weighted average age
    wav_method : str
        wtd. average method - one of 'ca', 'ra', 'sp', 'r2'
    method : str
        method of age calculation, one of 'Ludwig' or 'Sakata'
    rand : bool
        plot random age uncerts on wtd. average plot
    cov : bool
        account for age covariance structure in wtd. average

    """
    assert method in ('Ludwig', 'Sakata')
    x, sx, y, sy, r_xy = dp
    n = x.shape[0]
    assert sx.shape[0] == y.shape[0] == sy.shape[0] == r_xy.shape[0] == n
    if DThU_const:
        DThU = float(DThU)
        DThU_1s = float(DThU_1s)
    else:
        assert len(ThU_min) == n
        ThU_melt = float(ThU_melt)
        ThU_melt_1s = float(ThU_melt_1s)
    DPaU = float(DPaU)
    DPaU_1s = float(DPaU_1s)

    if np.atleast_1d(t0).size:
        t0 = np.broadcast_to(t0, (n,))

    # secular eq. values should be fine for most cases
    A48_eq = cfg.A48_eq
    A68_eq = cfg.A68_eq

    if DThU_const:
        fThU = np.broadcast_to(DThU, (n,))
    else:
        fThU = ThU_min / ThU_melt

    # only type 1 useful for Pa/U?
    fPaU = np.broadcast_to(DPaU, (n,))

    # Comupte ages:
    t = np.zeros(n)
    for i in range(n):
        if method == 'Ludwig':
            A = [A48_eq, fThU[i], A68_eq, fPaU[i]]
            init = (True, True)     # no present values for now
        # try:
        if method == 'Sakata':
            t[i] = sakata(x[i], y[i], fThU[i], fPaU[i], Pb76, t0[i])
        else:
            t[i] = mod207(x[i], y[i], A, Pb76, t0[i])
        # except:
        #     t[i] = nan

    if np.all(np.isnan(t)):
        raise ValueError('all age solutions failed to converge')

    # Compute age uncertainties:
    # dc_errors not yet implemented
    mc = dqmc.mod207_age(t, x, sx, y, sy, r_xy, Pb76, Pb76se, ThU_min=ThU_min,
            ThU_melt=ThU_melt, ThU_melt_1s=ThU_melt_1s, DThU=DThU,
            DThU_1s=DThU_1s, DPaU=DPaU, DPaU_1s=DPaU_1s, DThU_const=DThU_const,
            trials=trials, method=method, rand=rand, negative_ages=negative_ages,
            negative_ar=negative_ar, hist=hist)

    results = {
        'age_type': 'mod-207Pb',
        'age': t,
        'age_1s': mc['age_1s'],
        'age_95ci': mc['age_95ci'],
        'age_95pm': np.mean(abs(np.asarray(mc['age_95ci']) - t.reshape(n, -1)), axis=1),
        'cov_t': mc['cov_t'],
        'mc': mc
    }

    rand_pm = None

    # compute wtd. average
    if wav:
        if cov:
            cov_t = mc['cov_t']
            results['cov_t'] = cov_t
            wav_args = dict(cov=cov_t, method=wav_method)
        else:
            wav_args = dict(sx=np.diag(np.diag(mc['cov_t'])))
        if rand:
            results['age_rand_95ci'] = mc['age_rand_95ci']
            results['age_rand_95pm'] = np.mean(abs(np.asarray(mc['age_rand_95ci'])
                                                   - t.reshape(n, -1)), axis=1)
            rand_pm = results['age_rand_95pm']
        if wav_method in ('ca'):
            wav_results = wtd_average.classical_wav(t, **wav_args)
        elif wav_method in ('ra', 'rs', 'r2'):
            wav_results = wtd_average.robust_wav(t, **wav_args)
        else:
            raise ValueError('wav method not recognised')

        results['wav_age'] = wav_results['ave']
        results['wav_age_95pm'] = wav_results['ave_95pm']
        results['wav'] = wav_results

        # Make wav plot:
        age_mult = 1. if wav_plot_prefix == 'Ma' else 1000.
        fig = plotting.wav_plot(t, results['age_95pm'], wav_results['ave'],
                    wav_results['ave_95pm'], rand_pm=rand_pm,
                    sorted=sorted, ylim=ylim, x_multiplier=age_mult,
                    dp_labels=dp_labels)
        ax = fig.get_axes()[0]
        ax.set_ylabel(f'Modified $^{{207}}$Pb age ({wav_plot_prefix})')
        results['fig_wav'] = fig

    return results


def fc_A48i(fit57, fit86, A, sA, t0=1.0, norm_isotope='204Pb',
            negative_ar=True, negative_ages=True, hist=(False, False),
            trials=50_000):
    """
    Compute "forced concordance" [234U/238U] value following the approach of
    Engel et al., (2019).

    Parameters
    ----------
    fit57 :
        207Pb isochron linear regression fit
    fit86 :
        206Pb isochron linear regression fit
    A : array-like
        one-dimensional array of activity ratio values arranged as follows
        - np.nan, [230Th/238U], [226Ra/238U], [231Pa/235U]
    sA : array-like
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A

    """

    # compute iso-57 diseq age
    t57 = isochron(fit57['theta'][1], A[-1], t0, 'iso-Pb7U5')
    # compute concordant init [234U/238U] value
    DC8 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    BC8 = ludwig.bateman(DC8)

    A48i = concordant_A48i(t57, fit86['theta'][1], A[1], A[2], DC8, BC8,
                           A48i_guess=1.)

    # Compute init [234U/238U] uncertainties:
    mc = dqmc.fc_A48i(t57, A48i, fit57, fit86, A, sA, [True, True], trials=trials,
                    negative_ar=negative_ar, negative_ages=negative_ages,
                    hist=hist)

    results = {
        'norm_isotope': norm_isotope,
        '207Pb_age': t57,
        '207Pb_age_95pm': [0., 0.],
        '[234U/238U]i': A48i,
        'mc': mc,
    }
    return results


#=======================================
# Numerical age calculation functions
#=======================================

def concint(a, b, A, init, t0):
    """
    Numercially compute disequilibrium U-Pb concordia-intercept age.
    """
    DC8 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    DC5 = (cfg.lam235, cfg.lam231)
    BC8 = ludwig.bateman(DC8)
    BC5 = ludwig.bateman(DC5, series='235U')
    fmin, dfmin = minimise.concint(t0=1.0, diagram='tw', init=init)
    args = (a, b, A[:-1], A[-1], DC8, DC5, BC8, BC5, cfg.U)
    r = optimize.newton(fmin, t0, dfmin, full_output=True, disp=False, args=args)
    if not r[1].converged:
        raise exceptions.ConvergenceError('disequilibrium concordia age did '
                  'not converge after maximum number of iterations')
    t = r[0]
    return t


def isochron(b, A, t0, age_type, init=(True, True)):
    """
    Numerically compute disequilbrium U-Pb isohcron age.
    """
    assert age_type in ('iso-Pb6U8', 'iso-Pb7U5')
    if age_type == 'iso-Pb6U8':
        DC = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        BC = ludwig.bateman(DC)
    elif age_type == 'iso-Pb7U5':
        DC = (cfg.lam235, cfg.lam231)
        BC = ludwig.bateman(DC, series='235U')
    args = (b, A, DC, BC)
    fmin, dfmin = minimise.isochron(t0=t0, age_type=age_type, init=init)
    r = optimize.newton(fmin, t0, dfmin, full_output=True, disp=False,
                        args=args)
    if not r[1].converged:
        raise exceptions.ConvergenceError(f'disequilibrium isochron age did not '
              f'converge did not converge after maximum number of iterations, '
              f'fmin = {fmin(r[0], *args)}')
    t = r[0]
    return t


def pbu(x, A, t0, age_type, init=(True, True)):
    """
    Numerically compute single analysis Pb/U age using Ludwig equations.
    """
    assert age_type in ('Pb6U8', 'Pb7U5')
    if age_type == 'Pb6U8':
        DC = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        BC = ludwig.bateman(DC)
    elif age_type == 'Pb7U5':
        DC = (cfg.lam235, cfg.lam231)
        BC = ludwig.bateman(DC, series='235U')
    args = (x, A, DC, BC)
    fmin, dfmin = minimise.pbu(t0=t0, age_type=age_type, init=init)
    t, r = optimize.newton(fmin, t0, dfmin, full_output=True, disp=False,
            args=args)
    if not r.converged:
        raise exceptions.ConvergenceError('Pb/U age routine did not converge '
                'after maximum number of iterations')
    return t


def mod207(x, y, A, Pb76, t0, init=(True, True)):
    """
    Numerically compute modified 207Pb ages using the equations of Ludwig (1977).
    """
    DC8 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    DC5 = (cfg.lam235, cfg.lam231)
    BC8 = ludwig.bateman(DC8)
    BC5 = ludwig.bateman(DC5, series='235U')
    args = (x, y, A, DC8, DC5, BC8, BC5, cfg.U, Pb76)
    fmin, dfmin = minimise.mod207(t0=t0, init=init)
    t, r = optimize.newton(fmin, t0, dfmin, full_output=True, args=args)
    if not r.converged:
        raise exceptions.ConvergenceError('Modified 207Pb age routine did not '
                'converge after maximum number of iterations')
    return t


def guillong(x, fXU, t0, age_type):
    """
    Numerically compute  single analysis Pb/U age using the equations and
    approach of Guillong (2014).

    Parameters
    ----------
    x : float or np.ndarray
        measured Pb/U ratio(S)
    fXU : float of np.ndaraay
        Th-U or Pa-U "fractionation factor"
    """
    if age_type == 'Pb6U8':
        DC = (cfg.lam238, cfg.lam230)
    else:
        DC = (cfg.lam235, cfg.lam231)
    fmin, dfmin = minimise.guillong(t0=t0)
    t, nr = optimize.newton(fmin, t0, dfmin, args=(x, fXU, *DC), full_output=True)
    if not nr.converged:
        raise exceptions.ConvergenceError('Guillong modified Pb/U age routine '
                  'did not converge after maximum number of iterations')
    return t


def sakata(x, y, fThU, fPaU, Pb76, t0, disp=False):
    """
    Numerically compute modified 207Pb age using the equations and approach of
    Sakata (2017).
    """
    fmin, _ = minimise.sakata(t0=t0)
    args = (x, y, fThU, fPaU, cfg.lam238, cfg.lam230, cfg.lam235, cfg.lam231,
            cfg.U, Pb76)
    t, nr = optimize.bisect(fmin, 1e-06, 100, args=args, full_output=True,
                            disp=disp)
    if not nr.converged:
        raise exceptions.ConvergenceError('Sakata modified 207Pb age routine '
                  'did not converge after maximum number of iterations')
    return t


def concordant_A48i(t75, b86, A08, A68, DC8, BC8, A48i_guess=1.):
    """
    Numerically compute initial U234/U238 activity ratio that forces concordance
    between 238U and 235U isochron ages.

    Minimises function: f = F(t75, A234A238) - slope_86, where t75 is
    the 207Pb/x-235U/x isochron age.
    """
    args = (t75, b86, [nan, A08, A68], DC8, BC8)
    fmin, dfmin = minimise.concordant_A48()
    r = optimize.newton(fmin, A48i_guess, dfmin, args=args,
            full_output=True, disp=False)
    if not r[1].converged:
        raise exceptions.ConvergenceError('forced concordant initial [234U/238U] '
               'routine did not converge after maximum number of iterations')
    return r[0]


def concint_multiple(a, b, A, init, t0, age_lim=(0., 20.), t_step=1e-5,
        A48i_lim=(0., 20.), A08i_lim=(0., 20.)):
    """
    Search for all initial disequilibrium concordia-intercept age solutions
    within specified age and initial activity ratio limits.

    If both [234U/238U] and [230Th/238U] are inputted as present-day
    values, it is possible for the concordia curve to curve around leading to two
    concordia intercepts in close age proximity. Typically, where this is the 
    case, it seems the  upper intercept will have a physically implausible 
    initial activity ratio solution, so lower intercept is chosen.

    """
    if not age_lim[0] <= t0 <= age_lim[1]:
        raise ValueError('age guess must be b/w upper and lower age limits')

    DC8 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    DC5 = (cfg.lam235, cfg.lam231)
    BC8 = ludwig.bateman(DC8)
    BC5 = ludwig.bateman(DC5, series='235U')

    # compile args for age solution
    args = (a, b, A[:-1], A[-1], DC8, DC5, BC8, BC5, cfg.U)

    # Find all solutions within age limits -- in case there are more than 1.
    fmin, dfmin = minimise.concint(t0, diagram='tw', init=init)
    roots, _ = find_roots(fmin, dfmin, args, range=age_lim, step=t_step)

    # If no ages found yet, try using inputted age guess directly in numerical
    # age routine:
    if len(roots) == 0:
        try:
            t = concint(a, b, A, init, t0)
        except exceptions.ConvergenceError:
            raise RuntimeError('no disequilibrium age solutions found')
        roots = np.atleast_1d(t)

    # Reject ages outside limits
    accept = np.where(np.logical_and(age_lim[0] < roots, roots < age_lim[1]),
                      np.full(roots.shape, True), np.full(roots.shape, False))
    accept = np.atleast_1d(accept)
    roots = roots[accept]

    # User youngest intercept age
    if roots.size > 0:
        ind = np.argsort(roots)
        roots = roots[ind]

    # Calculate activity ratios if present-day ratios given
    A48i = np.zeros(roots.shape) if not init[0] else None
    A08i = np.zeros(roots.shape) if not init[1] else None

    for i, t in enumerate(roots):
        if not init[0]:
            # present 234U/238U activity ratio:
            A48i[i] = useries.ar48i(t, A[0], cfg.lam238, cfg.lam234)
            if not init[1]:
                A08i[i] = useries.ar08i(t, A[0], A[1], cfg.lam238, cfg.lam234,
                                        cfg.lam230, init=init[0])
        elif not init[0]:
            # initial 234U/238U activity ratio:
            A08i[i] = useries.ar08i(t, A[0], A[1], cfg.lam238, cfg.lam234,
                                    cfg.lam230, init=init[0])

    # check that activiy ratios are within limits
    if not init[0]:
        accept = np.where(np.logical_and(A48i_lim[0] < A48i,
                        A48i < A48i_lim[1]), accept,
                        np.full(roots.shape, False))
    if not init[1]:
        accept = np.where(np.logical_and(A08i_lim[0] < A08i,
                        A08i < A08i_lim[1]), accept,
                        np.full(roots.shape, False))

    # now mask out rejected values
    ages = roots[accept]
    if A48i is not None:
        A48i = A48i[accept]
    if A08i is not None:
        A08i = A08i[accept]

    if len(ages) < 1:
        raise ValueError('no disequilibrium age solutions found within '
                         'age and activity ratio limits')

    return ages, A48i, A08i


def find_roots(f, df, args, range=(0., 20.), step= 1e-5, rtol=1e-9):
    """
    Use brute force method to find all roots within given age range. Not
    guaranteed to work if two roots are close together, so use small step
    size.
    """
    # Find where f changes sign
    x = np.arange(range[0], range[1] + step, step)
    y = f(x, *args)
    sign_change = np.diff(np.sign(y)) != 0.
    x_ind = np.where(sign_change)[0]

    # Find guess at roots
    x0 = x[x_ind]
    roots = []

    # Polish roots using Newton-Raphson
    for i, v in enumerate(x0):
        root, rr = optimize.newton(f, v, df, args=args, rtol=rtol,
                                   full_output=True, disp=False)
        if rr.converged and not np.isnan(root):
            roots.append(root)

    # Remove any duplicate roots
    if len(roots) != 0:
        out = [roots[0]]
        for r in roots[1:]:
            if not np.any(np.isclose(np.full(len(out), r),
                                     out, atol=0, rtol=1e-04)):
                out.append(r)
    else:
        out = []

    return np.array(out), y


#=======================================
# Analytical age uncertainty functions
#=======================================

# def conint_age_uncert():
#     """ Analytical concordia-intercept age uncertainties.
#     """
#     return


# def mod207Pb_age_uncert():
#     """ Analytical mod. 207Pb age uncertainties.
#     """
#     return





