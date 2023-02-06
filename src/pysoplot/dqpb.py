"""
Functions and routines for computing disequilibrium U-Pb ages.

"""

import warnings
import numpy as np
from scipy import optimize

from pysoplot import useries
from pysoplot import ludwig
from pysoplot import minimise
from pysoplot import wtd_average
from pysoplot import cfg
from pysoplot import plotting
from pysoplot import mc
from pysoplot import exceptions
from pysoplot import stats


exp = np.exp
log = np.log
nan = np.nan


#===========================================
# Age calculation routines
#===========================================

def concint_age(fit, A, sA, init, t0, diagram='tw', dc_errors=False,
        trials=50_000, u_errors=False, negative_ratios=True, negative_ages=True,
        intercept_plot=True, hist=(False, False), conc_kw=None,
        intercept_plot_kw=None, A48i_lim=(0., 20.), A08i_lim=(0., 10.),
        age_lim=(0., 20.), uncert='mc'):
    """
    Compute disequilibrium U-Pb concordia intercept age and age uncertainties
    using Monte Carlo simulation. Optionally, produce a plot of the
    concordia intercept.

    Notes
    ------
    Only accepts data in 2-D Tera-Wasserburg form at present.
    
    Parameters
    ----------
    fit : dict
        Linear regression fit parameters.
    A : array-like
        One-dimensional array of activity ratio values arranged as follows
        - [234U/238U], [230Th/238U], [226Ra/238U], [231Pa/235U].
    sA : array-like
        One-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A.
    init : array-like
        Two-element list of boolean values, the first is True if [234U/238U]
        is an initial value and False if a present-day value, the second is True
        if [230Th/238U] is an initial value and False if a present-day value.
    t0 : float
        Initial guess for numerical age solving (the equilibrium age is
        typically good enough).
    uncert : {'mc', 'none'}
        Method of computing age uncertainties.
    diagram : {'tw'}
        Concordia diagram type. Currently only accepts 'tw', i.e.
        Tera-Waserburg.
    
    """
    assert diagram == 'tw'
    assert type(init[0]) == bool and type(init[1] == bool)

    if intercept_plot_kw is None:
        intercept_plot_kw = {}
    if conc_kw is None:
        conc_kw = {}

    # Compute age:
    a, b = fit['theta']
    
    # If a present-day activity ratio given, check for all possible intercept age
    # solutions within given age limits.
    if not all(init):
        r = concint_multiple(a, b, A, init, t0, age_lim=age_lim,  t_step=1e-5,
                             A48i_lim=A48i_lim, A08i_lim=A08i_lim)
        ages, a234_238_i, a230_238_i = r
        t = ages[0]
    else:
        t = concint(a, b, A, init, t0)
        if t < 0:
            raise RuntimeError(f'negative disequilibrium age solution: {t} Ma')

    # Get initial acitivity rario solutions:
    a234_238_i, a230_238_i = useries.init_ratio_solutions(t, A[:-1], init,
                   (cfg.lam238, cfg.lam234, cfg.lam230))

    results = {
        'age_type': 'concordia-intercept',
        'diagram': diagram,
        'age': t,
        '[234U/238U]i': a234_238_i,
        '[230Th/238U]i': a230_238_i
    }

    if uncert == 'mc':
        # Check measured activity ratios
        if not init[0]:
            p = stats.two_sample_p(A[0], sA[0], cfg.a234_238_eq, cfg.a234_238_eq_1s)
            if p > 0.05:
                raise ValueError(f'Cannot run monte carlo simulation: [234U/238U] '
                        f'not sufficiently resolved from equilibrium, p = {p}')
        if not init[1]:
            p = stats.two_sample_p(A[0], sA[0], cfg.a234_238_eq, cfg.a234_238_eq_1s)
            if p > 0.05:
                raise ValueError(f'Cannot run monte carlo simulation: [230Th/238U] '
                        f'not sufficiently resolved from equilibrium, p = {p}')

        # Compute age uncertainties:
        mc_result = mc.concint_diseq_age(t, fit, A, sA, init, trials=trials,
                            dc_errors=dc_errors, u_errors=u_errors,
                            negative_ratios=negative_ratios, negative_ages=negative_ages,
                            hist=hist, intercept_plot=intercept_plot, conc_kw=conc_kw,
                            intercept_plot_kw=intercept_plot_kw)

        results.update({
            'age_1s': mc_result['age_1s'],
            'age_95ci': mc_result['age_95ci'],
            'age_95pm': np.mean((t - mc_result['age_95ci'][0],
                                 mc_result['age_95ci'][1] - t)),
            'mc': mc_result,})

        if not init[0]:
            results['[234U/238U]i_95ci'] = mc_result['[234U/238U]i_95ci']
            results['[234U/238U]i_95pm'] = np.mean((t - mc_result['[234U/238U]i_95ci'][0],
                                          mc_result['[234U/238U]i_95ci'][1] - t))
        if not init[1]:
            results['[230Th/238U]i_95ci'] = mc_result['[230Th/238U]i_95ci']
            results['[230Th/238U]i_95pm'] = np.mean((t - mc_result['[230Th/238U]i_95ci'][0],
                                            mc_result['[230Th/238U]i_95ci'][1] - t))

    return results


def isochron_age(fit, A, sA, t0, init=(True, True), age_type='iso-206Pb',
        norm_isotope='204Pb', hist=(False, False), trials=50_000,
        dc_errors=False, negative_ratios=True, negative_ages=True):
    """
    Compute disequilibrium 238U-206Pb or 235U-207Pb isochron age and Monte
    Carlo age uncertainties.

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
    age_type : {'iso-206Pb', 'iso-207Pb'}
        Isochron age type.

    """
    assert age_type in ('iso-206Pb', 'iso-207Pb')

    if age_type == 'iso-206Pb':
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
    if age_type == 'iso-206Pb':
        a234_238_i, a230_238_i = useries.init_ratio_solutions(t, A, init, [cfg.lam238,
                                    cfg.lam234, cfg.lam230])
        if not init[0]:
            p = stats.two_sample_p(A[0], sA[0], cfg.a234_238_eq, cfg.a234_238_eq_1s)
            if p > 0.05:
                raise ValueError(f'Cannot run monte carlo simulation: [234U/238U] '
                        f'not sufficiently resolved from equilibrium, p = {p}')
        if not init[1]:
            p = stats.two_sample_p(A[0], sA[0], cfg.a234_238_eq, cfg.a234_238_eq_1s)
            if p > 0.05:
                raise ValueError(f'Cannot run monte carlo simulation: [230Th/238U] '
                        f'not sufficiently resolved from equilibrium, p = {p}')
    else:
        a234_238_i, a230_238_i = (None, None)

    # Compute age uncertainties:
    mc_result = mc.isochron_diseq_age(t, fit, A, sA, init=init, trials=trials,
                           negative_ratios=negative_ratios, negative_ages=negative_ages,
                           hist=hist, age_type=age_type, dc_errors=dc_errors)

    results = {
        'age_type': f'{age_type} isochron',
        'norm_isotope': norm_isotope,
        'age': t,
        'age_1s': mc_result ['age_1s'],
        'age_95ci': mc_result ['age_95ci'],
        'age_95pm': mc_result ['age_95pm'],
        '[234U/238U]i': a234_238_i,
        '[230Th/238U]i': a230_238_i,
        'mc': mc_result,
    }

    return results


def pbu_age(x, Vx, t0, DThU=None, DThU_1s=None, DPaU=None, DPaU_1s=None, alpha=None,
    alpha_1s=None, age_type='206Pb*', uncert='analytical',
    rand=False, wav=False, wav_opts=None, mc_opts=None):
    """
    Compute disequilibrium radiogenic 206Pb*/238U, 207Pb*/235U ages, or 
    207Pb-corrected age assuming a constant ratio of minerl-melt partition
    coefficients, and optionally compute a weighted average age.

    Parameters
    ----------
    x : ndarray (1-D, 2-D)
        Either a 1 x n array of measured 206Pb*/238U for each aliquot (for
        206Pb* age), or 2 x n array of measured 238U/206Pb and 207Pb/206Pb (for
        207Pb-corrected age).
    Vx : ndarray (2-D)
        Covariance matrix of uncertainties on measured isotope ratios. This
        should be an n x n array for 206Pb* and 207Pb* ages, or a 2n x 2n array
        for 207Pb-corrected ages.
    DThU : float, optional
        Ratio of mineral-melt distribution coefficients, DTh/DU.
    DThU_1s : float, optional
        Uncertainty on ratio of mineral-melt distribution coefficients, DTh/DU.
    DPaU : float, optional
        Ratio of mineral-melt distribution coefficients, DPu/DU.
    DPaU_1s : float, optional
        Uncertainty on ratio of mineral-melt distribution coefficients, DPu/DU.
    alpha : float, optional
        Common 207Pb/206Pb ratio.
    alpha_1s : float, optional
        Uncertainty on common 207Pb/206Pb ratio.
    t0 : ndarray or float, optional
        Age guess(es).
    uncert : {'mc', 'analytical'}, optional
        Method of propagating age uncertainties.
    rand : bool
        If true, compute random only uncertainties as well as total
        uncertainties (e.g., for plotting error bars).
    mc_opts : dict
        Monte Carlo simulation options.
    wav_opts : dict
        Weighted average calculation and plotting options.

    Returns
    --------
    results : dict
        Age and uncertainty results.

    """
    assert age_type in ('206Pb*', '207Pb*', 'cor207Pb'), 'age_type not recognised'

    if mc_opts is None:
        mc_opts = dict(trials=50_000, negative_ages=False, negative_ratios=False)

    if wav_opts is None:
        wav_opts = dict(wav_method='ra', cov=True, plot=True, plot_prefix='Ma',
                        ylim=(None, None), dp_labels=None, sorted=False)
    else:
        if wav and 'wav_method' in wav_opts.keys():
            if wav_opts['wav_method'] not in ('ca', 'ra', 'rs'):
                raise ValueError('wav_method not recognised')

    if age_type in ('206Pb*', '207Pb*'):
        if x.ndim != 1 or Vx.shape != (x.size, x.size):
            raise ValueError('inputs have incompatible dimensions')
        if age_type == '206Pb*':
            if (DThU is None) or (DThU_1s is None):
                raise ValueError('DThU and DThU_1s must be provided for 206Pb* age')
        else:
            if (DPaU is None) or (DPaU_1s is None):
                raise ValueError('DPaU and DPaU_1s must be provided for 207Pb* age')
        n = len(x)

    if age_type == 'cor207Pb':
        if x.ndim != 2 or x.shape[0] != 2 or Vx.shape != (2 * x[0].size, 2 * x[0].size):
            raise ValueError('inputs have incompatible dimensions')
        if (DPaU is None) or (DPaU_1s is None) or (alpha is None):
            raise ValueError('must provide y, DPaU, DPaU_1s and alpha for cor-207Pb age')
        n = len(x[0])

    v_shape = (n, n)

    if t0 is not None:
        if isinstance(t0, float):
            t0 = np.full(n, t0)
        else:
            assert t0.ndim == 1 and t0.size == n, "shape of t0 incompatible with x"

    # allocate arrays to store results
    t = np.zeros(n)

    for i in range(n):

        try:
            if age_type == '206Pb*':
                A = [cfg.a234_238_eq, DThU, cfg.a226_238_eq]
            elif age_type == '207Pb*':
                A = DPaU
            else:
                A = [cfg.a234_238_eq, DThU, cfg.a226_238_eq, DPaU]

            if age_type in ('206Pb*', '207Pb*'):
                t[i] = pbu(x[i], A, t0[i], age_type)
            else:
                t[i] = cor207(x[0, i], x[1, i], A, alpha, t0[i])

        except exceptions.ConvergenceError:
            t[i] = np.nan

    bad = np.argwhere(np.isnan(t))
    if bad.size != 0:
        msg = f'age calculation failed for data points {[x for x in bad.flatten()]}'
        warnings.warn(msg)

    results = {
        'age_type': age_type,
        'age': t,
    }

    # Propagate total (random analytical + systematic) age uncertainties using
    # either analytical or Monte Carlo approach.
    kwargs = dict(age_type=age_type)

    if uncert == 'mc':
        if age_type == '206Pb*':
            kwargs.update(dict(DThU=DThU, DThU_1s=DThU_1s))
        elif age_type == '207Pb*':
            kwargs.update(dict(DPaU=DPaU, DPaU_1s=DPaU_1s))
        else:
            kwargs.update(dict(DThU=DThU, DThU_1s=DThU_1s, DPaU=DPaU,
                    DPaU_1s=DPaU_1s, alpha=alpha, alpha_1s=alpha_1s))

        if bad.size != 0:
            msg = 'failed ages - could not run Monte Carlo simulation'
            warnings.warn(msg)
            return results

        mc_results = mc.pbu_diseq_age(t, x, Vx, **kwargs, **mc_opts)
        results['mc'] = mc_results
        results['age_1s'] = results['mc']['age_1s']
        results['age_95ci'] = mc_results['age_95ci']
        results['cov_t'] = mc_results['cov_t']
        results['age_95pm'] = mc_results['age_95pm']

    else:
        kwargs = dict(age_type=age_type)

        if age_type in ('206Pb*', 'cor207Pb'):
            V_a230_238 = np.full((n, n), DThU_1s ** 2)
            kwargs.update(dict(a230_238=np.full(n, DThU), V_a230_238=V_a230_238))
        if age_type in ('207Pb*', 'cor207Pb'):
            V_a231_235 = np.full((n, n), DPaU_1s ** 2)
            kwargs.update(dict(a231_235=np.full(n, DPaU), V_a231_235=V_a231_235))
        if age_type == 'cor207Pb':
            kwargs.update(dict(alpha=alpha, alpha_1s=alpha_1s))

        Vt = pbu_uncert(t, x, Vx, **kwargs)
        st = np.sqrt(np.diag(Vt))

        results['cov_t'] = Vt
        results['age_1s'] = np.sqrt(np.diag(Vt))
        results['age_95pm'] = 1.96 * results['age_1s']
        results['age_95ci'] = [[t - 1.96 * st, t + 1.96 * st] for t, st in zip(t, st)]

    # Propagate random only age uncertainties
    if rand:

        if uncert == 'mc':
            kwargs = dict(age_type=age_type)
            if age_type == '206Pb*':
                kwargs.update(dict(DThU=DThU, DThU_1s=0.))
            elif age_type == '207Pb*':
                kwargs.update(dict(DPaU=DPaU, DPaU_1s=0.))
            else:
                kwargs.update(dict(DThU=DThU, DThU_1s=0., DPaU=DPaU, DPaU_1s=0.,
                                   alpha=alpha, alpha_1s=0.))

            bad = np.argwhere(np.isnan(t))
            if bad.size != 0:
                msg = f'age calculation for data points {[x for x in bad.flatten()]} ' \
                      f'failed - could not run Monte Carlo simulation'
                warnings.warn(msg)
                return results

            mc_results = mc.pbu_diseq_age(t, x, Vx, **kwargs, **mc_opts)
            results['uncert'] = 'Monte Carlo'
            results['mc_rand'] = mc_results
            results['rand_cov_t'] = mc_results['cov_t']
            results['age_rand_95pm'] = mc_results['age_95pm']

        else:
            kwargs = dict(age_type=age_type)

            if age_type in ('206Pb*', 'cor207Pb'):
                kwargs.update(dict(a230_238=np.full(n, DThU),
                                   V_a230_238=np.full((n, n), 0.)))
            if age_type in ('207Pb*', 'cor207Pb'):
                kwargs.update(dict(a231_235=np.full(n, DPaU),
                                   V_a231_235=np.full((n, n), 0.)))
            if age_type == 'cor207Pb':
                kwargs.update(dict(alpha=alpha, alpha_1s=0.))

            Vt = pbu_uncert(t, x, Vx, **kwargs)

            results['uncert'] = 'analytical'
            results['rand_cov_t'] = Vt
            results['age_rand_1s'] = np.sqrt(np.diag(Vt))
            results['age_rand_95pm'] = 1.96 * results['age_rand_1s']

    # Compute wtd. average
    if wav:

        if bad.size != 0:
            msg = f'could not compute wtd. average - age calculation failed for some data points'
            warnings.warn(msg)
            return results

        cov_t = results['cov_t']
        if not wav_opts['cov']:
            cov_t = np.diag(np.diag(cov_t))

        kwargs = dict(V=cov_t, method=wav_opts['wav_method'])
        if wav_opts['wav_method'] == 'ca':
            wav_results = wtd_average.classical_wav(t, **kwargs)
        else:
            wav_results = wtd_average.robust_wav(t, **kwargs)

        results['wav_age'] = wav_results['ave']
        results['wav_age_95pm'] = wav_results['ave_95pm']
        results['wav'] = wav_results

        # Make wav plot:
        if rand:
            rand_pm = np.asarray(results['age_rand_95pm'])
        age_mult = 1. if wav_opts['plot_prefix'] == 'Ma' else 1000.
        kwargs = dict(x_multiplier=age_mult, sorted=wav_opts['sorted'],
                      ylim=wav_opts['ylim'], dp_labels=wav_opts['dp_labels'])
        if rand:
            kwargs.update(dict(rand_pm=rand_pm))
        if wav_opts['plot']:
            fig = plotting.wav_plot(t, np.asarray(results['age_95pm']), wav_results['ave'],
                        wav_results['ave_95pm'], **kwargs)
            ax = fig.get_axes()[0]
            if age_type == '206Pb*':
                ax.set_ylabel(f"$^{{206}}$Pb/$^{{238}}$U age ({wav_opts['plot_prefix']})")
            elif age_type == '207Pb*':
                ax.set_ylabel(f"$^{{207}}$Pb/$^{{235}}$U age ({wav_opts['plot_prefix']})")
            else:
                ax.set_ylabel(f"$^{{207}}$Pb-corrected age ({wav_opts['plot_prefix']})")
            results['fig_wav'] = fig

    return results


def pbu_iterative_age(x, Vx, ThU_melt, ThU_melt_1s, t0, Pb208_206=None,
        V_Pb208_206=None, Th232_U238=None, V_Th232_U238=None, DPaU=None,
        DPaU_1s=None, alpha=None, alpha_1s=None, age_type='206Pb*',
        uncert='analytical', rand=False, wav=False, wav_opts=None,
        mc_opts=None):
    """
    Compute disequilibrium 206Pb*/U238 or 207Pb-corrected ages iteratively
    along with age uncertainties. Either initial Th/U_min can be inferred from
    measured 232Th/238U, or from radiogenic 208Pb/206Pb and age.

    Parameters
    ----------
    x : ndarray (1-D, 2-D)
        Either a 1 x n array of measured 206Pb*/238U for each aliquot (for
        206Pb* age), or 2 x n array of measured 238U/206Pb and 207Pb/206Pb (for
        207Pb-corrected age).
    Vx : ndarray (2-D)
        Covariance matrix of uncertainties on measured isotope ratios. This
        should be an n x n array for 206Pb* and 207Pb* ages, or a 2n x 2n array
        for 207Pb-corrected ages.
    ThU_melt : float
        Th/U ratio of the melt.
    ThU_melt_1s : float
        Uncertainty on Th/U ratio of the melt (1 sigma).
    Th232_U238 : ndarray (1-D), optional
        Measured 232Th/238U for each aliquot.
    V_Th232_U238 : ndarray (2-D), optional
        Covariance matrix of uncertainties on measured 232Th/238U.
    Pb208_206 : ndarray (1-D)
        Measured radiogenic 208Pb/206Pb for each aliquot.
    V_Pb208_206 : ndarray (2-D):
        Covariance matrix of uncertainties on radiogenic 208Pb/206Pb values.
    DPaU : float, optional
        Ratio of mineral-melt distribution coefficients, DPu/DU.
    DPaU_1s : float, optional
        Uncertainty on ratio of mineral-melt distribution coefficients, DPu/DU.
    alpha : float, optional
        Common 207Pb/206Pb ratio.
    alpha_1s : float, optional
        Uncertainty on common 207Pb/206Pb ratio.
    t0 : ndarray or float, optional
        Age guess(es).
    uncert : {'mc', 'analytical'}, optional
        Method of propagating age uncertainties.
    rand : bool
        If true, compute random only uncertainties as well as total
        uncertainties (e.g., for plotting error bars).
    mc_opts : dict
        Monte Carlo simulation options.
    wav_opts : dict
        Weighted average calculation and plotting options.

    Returns
    --------
    results : dict
        Age and uncertainty results.


    """
    assert age_type in ('206Pb*', 'cor207Pb'), 'age_type not recognised'

    if mc_opts is None:
        mc_opts = dict(trials=50_000, negative_ages=False, negative_ratios=False)

    if wav_opts is None:
        wav_opts = dict(wav_method='ra', cov=True, plot=True, plot_prefix='Ma',
                        ylim=(None, None), dp_labels=None, sorted=False)
    else:
        if wav and 'wav_method' in wav_opts.keys():
            if wav_opts['wav_method'] not in ('ca', 'ra', 'rs'):
                raise ValueError('wav_method not recognised')

    if age_type == '206Pb*':
        if x.ndim != 1 or Vx.shape != (x.size, x.size):
            raise ValueError('inputs have incompatible dimensions')
        n = len(x)

    if age_type == 'cor207Pb':
        if x.ndim != 2 or x.shape[0] != 2 or Vx.shape != (2 * x[1].size, 2 * x[1].size):
            raise ValueError('inputs have incompatible dimensions')
        if (DPaU is None) or (alpha is None):
            raise ValueError('must provide y, D_PaU, and alpha for cor-207Pb age')
        n = len(x[0])

    v_shape = (n, n)

    if Th232_U238 is not None:
        meas_Th232_U238 = True
        if Pb208_206 is not None:
            raise ValueError('must provide either Pb208_206 or Th232_U238')
        if V_Th232_U238 is None:
            raise ValueError('V_Th232_U238 must be provided if Th232_238U is')
        if Th232_U238.shape != (n,) or V_Th232_U238.shape != v_shape:
            raise ValueError('Th232_U238 and/or V_Th232_U238 have incompatible shapes')
    else:
        meas_Th232_U238 = False
        if Pb208_206.shape != (n,) or V_Pb208_206.shape != v_shape:
            raise ValueError('Pb208_206 and/or V_Pb208_206 have incompatible shapes')
        if V_Pb208_206 is None:
            raise ValueError('V_Pb208_2068 must be provided if Pb208_206 is')

    if isinstance(t0, float):
        t0 = np.full(n, t0)
    else:
        assert t0.ndim == 1 and t0.size == n, "shape of t0 incompatible with x"

    # allocate arrays to store results
    t = np.zeros(n)
    ThU_min = np.zeros(n)

    for i in range(n):

        kwargs = dict(age_type=age_type)

        try:
            if meas_Th232_U238:
                kwargs.update(dict(Th232_U238=Th232_U238[i]))
            else:
                kwargs.update(dict(Pb208_206=Pb208_206[i]))
            if age_type == 'cor207Pb':
                kwargs.update(dict(DPaU=DPaU, alpha=alpha))

            if age_type == '206Pb*':
                t[i], ThU_min[i] = pbu_iterative(x[i], ThU_melt, t0[i], **kwargs)
            else:
                t[i], ThU_min[i] = pbu_iterative(x[:, i], ThU_melt, t0[i], **kwargs)

            # f1, f2, _, f4 = ludwig.f_comp(t[i] [cfg.a234_238_eq, np.nan,
            #                 cfg.a226_238_eq], Lam238, coef238)
            # ThU_min[i] = ThU_melt_sim * (x[i] - (f1 + f2 + f4)) / (Lam238[0]/Lam238[2]
            #                 * (coef238[7] * np.exp((Lam238[0]-Lam238[2]) * t[i])
            #                 + coef238[8] * np.exp((Lam238[0]-Lam238[3]) * t[i])
            #                 + np.exp(Lam238[0] * t[i])))

        except exceptions.ConvergenceError:
            t[i] = np.nan
            ThU_min[i] = np.nan

    results = {
        'age_type': age_type,
        'age': t,
        'ThU_min': ThU_min,
    }

    # Propagate total (random analytical + systematic) age uncertainties using
    # either analytical or Monte Carlo approach.
    kwargs = dict(age_type=age_type)
    if meas_Th232_U238:
        kwargs.update(dict(Th232_U238=Th232_U238, V_Th232_U238=V_Th232_U238))
    else:
        kwargs.update(dict(Pb208_206=Pb208_206, V_Pb208_206=V_Pb208_206))
    if age_type == 'cor207Pb':
        kwargs.update(dict(alpha=alpha, alpha_1s=alpha_1s, DPaU=DPaU,
                      DPaU_1s=DPaU_1s))

    if uncert == 'mc':
        bad = np.argwhere(np.isnan(t))
        if bad.size != 0:
            msg = f'age calculation for data points {[x for x in bad.flatten()]} ' \
                  f'failed - could not run Monte Carlo simulation'
            warnings.warn(msg)
            return results

        mc_results = mc.pbu_iterative_age(t, ThU_min, x, Vx, ThU_melt, ThU_melt_1s,
                            **kwargs, **mc_opts)
        results['uncert'] = 'Monte Carlo'
        results['mc'] = mc_results
        results['age_1s'] = results['mc']['age_1s']
        results['age_95pm'] = mc_results['age_95pm']
        results['age_95ci'] = mc_results['age_95ci']
        results['ThU_min_1s'] = results['mc']['ThU_min_1s']
        results['ThU_min_95ci'] = results['mc']['ThU_min_95ci']
        results['cov_t'] = results['mc']['cov_t']

    else:
        Vt = pbu_iterative_uncert(t, ThU_min, x, Vx, ThU_melt, ThU_melt_1s,
                                        **kwargs)
        st = np.sqrt(np.diag(Vt))
        results['uncert'] = 'analytical'
        results['cov_t'] = Vt
        results['age_1s'] = np.sqrt(np.diag(Vt))
        results['age_95pm'] = 1.96 * results['age_1s']
        results['age_95ci'] = [[t - 1.96 * st, t + 1.96 * st] for t, st in zip(t, st)]
        results['ThU_min_1s'] = None
        results['ThU_min_95ci'] = None

    # Propagate random only age uncertainties
    if rand:
        kwargs = dict(age_type=age_type)
        if meas_Th232_U238:
            kwargs.update(dict(Th232_U238=Th232_U238, V_Th232_U238=V_Th232_U238))
        else:
            kwargs.update(dict(Pb208_206=Pb208_206, V_Pb208_206=V_Pb208_206))
        if age_type == 'cor207Pb':
            kwargs.update(dict(alpha=alpha, alpha_1s=0., DPaU=DPaU, DPaU_1s=0.))

        if uncert == 'mc':
            bad = np.argwhere(np.isnan(t))
            if bad.size != 0:
                msg = f'age calculation for data points {[x for x in bad.flatten()]} ' \
                      f'failed - could not run Monte Carlo simulation'
                raise ValueError(msg)
            mc_rand = mc.pbu_iterative_age(t, ThU_min, x, Vx, ThU_melt, 0., **kwargs,
                                           **mc_opts)
            results['mc_rand'] = mc_rand
            results['rand_cov_t'] = results['mc_rand']['cov_t']
            results['age_rand_95pm'] = results['mc_rand']['age_95pm']

        else:
            Vt = pbu_iterative_uncert(t, ThU_min, x, Vx, ThU_melt, 0., **kwargs)
            results['rand_cov_t'] = Vt
            results['age_rand_1s'] = np.sqrt(np.diag(Vt))
            results['age_rand_95pm'] = 1.96 * results['age_rand_1s']

    # Compute wtd. average
    if wav:
        cov_t = results['cov_t']
        if not wav_opts['cov']:
            cov_t = np.diag(np.diag(cov_t))
        if rand:
            rand_pm = np.asarray(results['age_rand_95pm'])
        kwargs = dict(V=cov_t, method=wav_opts['wav_method'])
        if wav_opts['wav_method'] == 'ca':
            wav_results = wtd_average.classical_wav(t, **kwargs)
        else:
            wav_results = wtd_average.robust_wav(t, **kwargs)

        results['wav_age'] = wav_results['ave']
        results['wav_age_95pm'] = wav_results['ave_95pm']
        results['wav'] = wav_results

        # Make wav plot:
        age_mult = 1. if wav_opts['plot_prefix'] == 'Ma' else 1000.
        kwargs = dict(x_multiplier=age_mult, sorted=wav_opts['sorted'],
                      ylim=wav_opts['ylim'], dp_labels=wav_opts['dp_labels'])
        if rand:
            kwargs.update(dict(rand_pm=rand_pm))
        if wav_opts['plot']:
            fig = plotting.wav_plot(t, np.asarray(results['age_95pm']), wav_results['ave'],
                        wav_results['ave_95pm'], **kwargs)
            ax = fig.get_axes()[0]
            if age_type == '206Pb*':
                ax.set_ylabel(f"$^{{206}}$Pb/$^{{238}}$U age ({wav_opts['plot_prefix']})")
            else:
                ax.set_ylabel(f"$^{{207}}$Pb/$^{{235}}$U age ({wav_opts['plot_prefix']})")
            results['fig_wav'] = fig

    return results


def forced_concordance(fit57, fit86, A, sA, t0=1.0, norm_isotope='204Pb',
            negative_ratios=True, negative_ages=True, hist=(False, False),
            trials=50_000):
    """
    Compute "forced concordance" [234U/238U] value following the approach of
    Engel et al. (2019).

    Parameters
    ----------
    fit57 : dict
        207Pb isochron linear regression fit
    fit86 : dict
        206Pb isochron linear regression fit
    A : array-like
        one-dimensional array of activity ratio values arranged as follows
        - np.nan, [230Th/238U], [226Ra/238U], [231Pa/235U]
    sA : array-like
        one-dimensional array of activity ratio value uncertainties given
        as 1 sigma absolute and arranged in the same order as A

    References
    -----------
    Engel, J., Woodhead, J., Hellstrom, J., Maas, R., Drysdale, R., Ford, D.,
    2019. Corrections for initial isotopic disequilibrium in the speleothem
    U-Pb dating method. Quaternary Geochronology 54, 101009.
    https://doi.org/10.1016/j.quageo.2019.101009

    """

    # compute iso-57 diseq age
    t57 = isochron(fit57['theta'][1], A[-1], t0, 'iso-207Pb')
    # compute concordant init [234U/238U] value
    Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    coef238 = ludwig.bateman(Lam238)

    a234_238_i = concordant_A48i(t57, fit86['theta'][1], A[1], A[2], Lam238, coef238,
                                 a0=1.)

    # Compute init [234U/238U] uncertainties:
    mc_result = mc.forced_concordance(t57, a234_238_i, fit57, fit86, A, sA, [True, True], trials=trials,
                                 negative_ratios=negative_ratios, negative_ages=negative_ages,
                                 hist=hist)

    results = {
        'norm_isotope': norm_isotope,
        '207Pb_age': t57,
        '207Pb_age_95pm': [0., 0.],
        '[234U/238U]i': a234_238_i,
        'mc': mc_result,
    }
    return results


#=======================================
# Numerical age calculation functions
#=======================================

def concint(a, b, A, init, t0):
    """
    Numercially compute disequilibrium U-Pb concordia-intercept age.
    """
    Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    Lam235 = (cfg.lam235, cfg.lam231)
    coef238 = ludwig.bateman(Lam238)
    coef235 = ludwig.bateman(Lam235, series='235U')
    fmin, dfmin = minimise.concint(diagram='tw', init=init)
    args = (a, b, A[:-1], A[-1], Lam238, Lam235, coef238, coef235, cfg.U)
    r = optimize.newton(fmin, t0, dfmin, args=args, full_output=True, disp=False)
    if not r[1].converged:
        raise exceptions.ConvergenceError('disequilibrium concordia age did '
                  'not converge after maximum number of iterations')
    t = r[0]
    return t


def isochron(b, A, t0, age_type, init=(True, True)):
    """
    Numerically compute disequilbrium U-Pb isohcron age.
    """
    assert age_type in ('iso-206Pb', 'iso-207Pb')
    if age_type == 'iso-206Pb':
        Lam = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        coef = ludwig.bateman(Lam)
    elif age_type == 'iso-207Pb':
        Lam = (cfg.lam235, cfg.lam231)
        coef = ludwig.bateman(Lam, series='235U')
    args = (b, A, Lam, coef)
    fmin, dfmin = minimise.isochron(age_type=age_type, init=init)
    r = optimize.newton(fmin, t0, dfmin, args=args, full_output=True,
                        disp=False)
    if not r[1].converged:
        raise exceptions.ConvergenceError('disequilibrium isochron age did not '
              'converge did not converge after maximum number of iterations')
    t = r[0]
    return t


def pbu(x, A, t0, age_type, alpha=None, init=(True, True)):
    """
    Numerically compute single analysis Pb/U age using Ludwig equations.
    """
    assert age_type in ('206Pb*', '207Pb*')
    if age_type == '206Pb*':
        Lam = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
        coef = ludwig.bateman(Lam)
    elif age_type == '207Pb*':
        Lam = (cfg.lam235, cfg.lam231)
        coef = ludwig.bateman(Lam, series='235U')
    args = (x, A, Lam, coef)
    fmin, dfmin = minimise.pbu(age_type=age_type, init=init)
    t, r = optimize.newton(fmin, t0, dfmin, args=args, full_output=True,
                           disp=False)
    if not r.converged:
        raise exceptions.ConvergenceError('disequilibrium Pb/U age did not converge '
                'after maximum number of iterations')
    return t


def pbu_iterative(x, ThU_melt, t0, Th232_U238=None, Pb208_206=None,
        DPaU=None, alpha=None, age_type='206Pb*', maxiter=50,
        tol=1e-08):
    """
    Compute disequilibrium 206Pb*/U238 or 207Pb-corrected ages iteratively.
    Either (232Th/238U)i is inferred from measured 232Th/238U, or from
    radiogenic 208Pb/206Pb and age.

    Parameters
    -----------
    x : ndarray (1-D, 2-D)
        Either measured 206Pb*/238U for each aliquot (for 206Pb* age),
        or measured 238U/206Pb and 207Pb/206Pb (for 207Pb-corrected age).
    ThU_melt : ndarray (1-D)
        Th/U ratio of the melt.
    Th232_U238 : ndarray (1-D)
        Measured 232Th/238U for each aliquot.
    Pb208_206 : ndarray (1-D)
        Measured radiogenic 208Pb/206Pb for each aliquot.
    D_PaU : float, optional
        Ratio of mineral-melt distribution coefficients, DPu/DU.
    alpha : float
        Common 207Pb/206Pb ratio.
    t0 : float, optional
        Age guess.

    Raises
    -------
    exceptions.ConvergenceError
        If no convergent age solution is found.

    """
    assert age_type in ('206Pb*', 'cor207Pb'), 'age_type not recognised'

    Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    coef238 = ludwig.bateman(Lam238)

    if age_type == 'cor207Pb':
        Lam235 = (cfg.lam235, cfg.lam231)
        coef235 = ludwig.bateman(Lam235, series='235U')
        x, y = x
        Pb206_U238 = 1. / x
    else:
        Pb206_U238 = x

    meas_Th232_U238 = True if Th232_U238 is not None else False
    fmin, dfmin = minimise.pbu(age_type=age_type)   # note: use pbu not pbu_iterative here

    iter = 1

    while iter < maxiter:
        iter += 1

        if not meas_Th232_U238:
            Th232_U238 = Pb206_U238 * Pb208_206 * (1. / (np.exp(cfg.lam232 * t0) - 1.))

        # Compute initial Th/U ratio from measured 232Th/238U ratio (perhaps
        # this is excessive?):
        ThU_min = Th232_U238 * (exp(cfg.lam232 * t0)
            / (exp(cfg.lam238 * t0) + np.exp(cfg.lam235 * t0) / cfg.U))
        fThU = ThU_min / ThU_melt

        if age_type == '206Pb*':
            args = (x, [cfg.a234_238_eq, fThU, cfg.a226_238_eq], Lam238,
                    coef238)
        else:
            args = (x, y, [cfg.a234_238_eq, fThU, cfg.a226_238_eq, DPaU], alpha,
                    cfg.U, Lam238, Lam235, coef238, coef235)

        t, r = optimize.newton(fmin, t0, dfmin, args=args, full_output=True,
                       disp=False)

        if not r.converged:
            raise exceptions.ConvergenceError(f'No age solution found for '
              f'iteration {iter} of Pb/U iterative age routine')

        if abs(t - t0) / t0 < tol:
            # Get Th/U_min solution
            if not meas_Th232_U238:
                Th232_U238 = Pb206_U238 * Pb208_206 * (1. / (np.exp(cfg.lam232 * t0) - 1.))
            ThU_min = Th232_U238 * (exp(cfg.lam232 * t)
                    / (exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U))
            return t, ThU_min

        t0 = t

    raise exceptions.ConvergenceError('Pb/U iterative age'
            'did not converge after maximum number of iterations')


def cor207(x, y, A, alpha, t0, init=(True, True)):
    """
    Numerically compute disequilibrium 207Pb-corrected ages.

    References
    ----------
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium
    on U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.

    """
    Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    Lam235 = (cfg.lam235, cfg.lam231)
    coef238 = ludwig.bateman(Lam238)
    coef235 = ludwig.bateman(Lam235, series='235U')
    args = (x, y, A, alpha, cfg.U, Lam238, Lam235, coef238, coef235)
    fmin, dfmin = minimise.pbu(age_type='cor207Pb', init=init)
    t, r = optimize.newton(fmin, t0, dfmin, args=args, full_output=True,
                           disp=True)
    if not r.converged:
        raise exceptions.ConvergenceError('Modified 207Pb age routine did not '
                'converge after maximum number of iterations')
    return t


def concordant_A48i(t75, b86, a230_238, a226_238, Lam238, coef238,
                    a0=1.):
    """
    Numerically compute initial U234/U238 activity ratio that forces concordance
    between 238U and 235U isochron ages.

    Minimises function: f = F(t75, A234A238) - slope_86, where t75 is
    the 207Pb/x-235U/x isochron age.
    """
    args = (t75, b86, [nan, a230_238, a226_238], Lam238, coef238)
    fmin, dfmin = minimise.concordant_A48()
    r = optimize.newton(fmin, a0, dfmin, args=args, full_output=True,
                        disp=False)
    if not r[1].converged:
        raise exceptions.ConvergenceError('forced concordant initial [234U/238U] '
               'value did not converge after maximum number of iterations')
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

    Lam238 = (cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226)
    Lam235 = (cfg.lam235, cfg.lam231)
    coef238 = ludwig.bateman(Lam238)
    coef235 = ludwig.bateman(Lam235, series='235U')

    # compile args for age solution
    args = (a, b, A[:-1], A[-1], Lam238, Lam235, coef238, coef235, cfg.U)

    # Find all solutions within age limits -- in case there are more than 1.
    fmin, dfmin = minimise.concint(diagram='tw', init=init)
    roots, _ = find_roots(fmin, dfmin, args, range=age_lim, step=t_step)

    # If no ages found yet, try using inputted age guess directly in numerical
    # age routine:
    if len(roots) == 0:
        try:
            t = concint(a, b, A, init, t0)
        except exceptions.ConvergenceError:
            raise exceptions.ConvergenceError('no disequilibrium age solutions found')
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
    a234_238_i = np.zeros(roots.shape) if not init[0] else None
    a230_238_i = np.zeros(roots.shape) if not init[1] else None

    for i, t in enumerate(roots):
        if not init[0]:
            # present 234U/238U activity ratio:
            a234_238_i[i] = useries.ar48i(t, A[0], cfg.lam238, cfg.lam234)
            if not init[1]:
                a230_238_i[i] = useries.ar08i(t, A[0], A[1], cfg.lam238, cfg.lam234,
                                        cfg.lam230, init=init[0])
        elif not init[0]:
            # initial 234U/238U activity ratio:
            a230_238_i[i] = useries.ar08i(t, A[0], A[1], cfg.lam238, cfg.lam234,
                                    cfg.lam230, init=init[0])

    # check that activiy ratios are within limits
    if not init[0]:
        accept = np.where(np.logical_and(A48i_lim[0] < a234_238_i,
                        a234_238_i < A48i_lim[1]), accept,
                        np.full(roots.shape, False))
    if not init[1]:
        accept = np.where(np.logical_and(A08i_lim[0] < a230_238_i,
                        a230_238_i < A08i_lim[1]), accept,
                        np.full(roots.shape, False))

    # now mask out rejected values
    ages = roots[accept]
    if a234_238_i is not None:
        a234_238_i = a234_238_i[accept]
    if a230_238_i is not None:
        a230_238_i = a230_238_i[accept]

    if len(ages) < 1:
        raise ValueError('no disequilibrium concordia-intercept age solutions found within '
                         'age and activity ratio limits')

    return ages, a234_238_i, a230_238_i


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


def pbu_uncert(t, x, Vx, a230_238=None, a231_235=None,
        V_a230_238=None, V_a231_235=None, alpha=None, alpha_1s=None,
        age_type='206Pb*'):
    
    """ 
    Compute uncertainties for a suite of disequilibrium 206Pb*/238U,
    207Pb*/235U or 207Pb-corrected ages using analytical error propagation.
    
    Parameters
    -----------
    x : ndarray (1-D, 2-D)
        Measured isotope ratios. This should be a 1-D array for 206Pb* and
        207Pb* ages, or a 2 x n array for 207Pb-corrected ages.
    Vx : ndarray (2-D)
        Covariance matrix of uncertainties on measured isotope ratios. This
        should be an n x n array for 206Pb* and 207Pb* ages, or a 2n x 2n array
        for 207Pb-corrected ages.
    a230_238 : ndarray (1-D), optional
        Initial [230Th/238U] activity ratios.
    V_a230_238 : ndarray (2-D), optional
        Covariance matrix of initial [230Th/238U] uncertainties.
    a231_235 : ndarray (1-D), optional
        Initial [231Pa/235U] activity ratios.
    V_a231_235 : ndarray (2-D), optional
        Covariance matrix of initial [231Pa/235U] uncertainties.
    alpha : float, optional
        Common 207Pb/206Pb ratio.
    alpha_1s : float, optional
        Uncertainty (1 sigma) on common 207Pb/206Pb ratio.
    age_type : {'206Pb*', '207Pb*', 'cor207Pb'}
        Age type to calculate uncertainties for.

    Returns
    -------
    Vt : ndarray (2-D)
        Covariance matrix of ages.
    
    """

    assert age_type in ('206Pb*', '207Pb*', 'cor207Pb')

    if age_type in ('206Pb*', '207Pb*'):
        if x.ndim != 1 or Vx.shape != (x.size, x.size):
            raise ValueError('inputs have incompatible dimensions')

    if age_type == 'cor207Pb':
        if x.ndim != 2 or x.shape[0] != 2 or Vx.shape != (2 * x[0].size, 2 * x[0].size):
            raise ValueError('inputs have incompatible dimensions')
        x, y = x
        Vxy = Vx

    n = len(x)
    zeros = np.zeros((n, n))

    if age_type in ('206Pb*', 'cor207Pb'):
        if a230_238.ndim != 1 or V_a230_238.shape != (n, n):
            raise ValueError('a230_238 and V_a230_238 have incompatible '
                             'dimensions')
        Lam238 = [cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226]
        coef238 = ludwig.bateman(Lam238)

        f1, f2, f3, f4 = ludwig.f_comp(t, [cfg.a234_238_eq, a230_238, cfg.a226_238_eq],
                                      Lam238, coef238)
        df1, df2, df3, df4 = ludwig.dfdt_comp(t, [cfg.a234_238_eq, a230_238, cfg.a226_238_eq],
                                              Lam238, coef238)
        f = f1 + f2 + f3 + f4
        dfdt = df1 + df2 + df3 + df4

    if age_type in ('207Pb*', 'cor207Pb'):
        if a231_235.ndim != 1 or V_a231_235.shape != (n, n):
            raise ValueError('a231_235 and V_a231_235 have incompatible '
                             'dimensions')
        Lam235 = [cfg.lam235, cfg.lam231]
        coef235 = ludwig.bateman(Lam235, series='235U')

        g1, g2 = ludwig.g_comp(t, a231_235, Lam235, coef235)
        dg1, dg2 = ludwig.dgdt_comp(t, a231_235, Lam235, coef235)
        g = g1 + g2
        dgdt = dg1 + dg2
      
    if age_type == '206Pb*':
        # dtdx
        dtdx = 1. / ludwig.dfdt(t, [cfg.a234_238_eq, a230_238, cfg.a226_238_eq],
                                Lam238, coef238)

        # dtdA08
        num = x - f1 - f2 - f4
        dnum = - df1 - df2 - df4
        den = Lam238[0]/Lam238[2] * (coef238[7] * exp((Lam238[0]-Lam238[2]) * t)
                + coef238[8] * exp((Lam238[0]-Lam238[3]) * t) + exp(Lam238[0]*t))
        dden = Lam238[0]/Lam238[2] * (
                coef238[7] * (Lam238[0]-Lam238[2]) * exp((Lam238[0]-Lam238[2]) * t)
                + coef238[8] * (Lam238[0]-Lam238[3]) * exp((Lam238[0]-Lam238[3]) * t)
                + Lam238[0] * exp(Lam238[0]*t))
        dtdA08 = den ** 2 / (dnum * den - dden * num)

        V = np.block([[Vx, zeros], [zeros, V_a230_238]])
        J = np.row_stack((np.diag(dtdx), np.diag(dtdA08)))

    elif age_type == '207Pb*':
        # dtdx
        dtdx = 1. / ludwig.dgdt(t, a231_235, Lam235, coef235)

        # dtd(a231_235)
        num = x - g1
        dnum = - dg1
        den = Lam235[0] / Lam235[1] *  (exp(Lam235[0] * t)
                - exp((Lam235[0]-Lam235[1]) * t))
        dden = Lam235[0] / Lam235[1] *  (Lam235[0] * exp(Lam235[0] * t)
                 - (Lam235[0]-Lam235[1]) * exp((Lam235[0]-Lam235[1]) * t))
        dtdA15 = den ** 2 / (dnum * den - dden * num)

        V = np.block([[Vx, zeros], [zeros, V_a231_235]])
        J = np.row_stack((np.diag(dtdx), np.diag(dtdA15)))

    elif age_type == 'cor207Pb':
        # dt/dy
        dydt = x / cfg.U * dgdt - alpha * x * dfdt
        dtdy = 1. / dydt
        
        # dt/dx
        dxdt = - (y - alpha) * (dgdt / cfg.U - alpha * dfdt) / (
                    g / cfg.U - alpha * f) ** 2
        dtdx = 1. / dxdt
        
        # dt/d(alpha)
        den = f - 1. / x
        dadt = (dgdt / cfg.U * den - dfdt * (g / cfg.U - y / x)) / den ** 2
        dtda = 1. / dadt
        
        # dt/d(a230_238)
        num = g / (cfg.U * alpha) - (y - alpha) / (alpha * x) - f1 - f2 - f4
        dnum = dgdt / (cfg.U * alpha) - df1 - df2 - df4
        den = Lam238[0] / Lam238[2] * (coef238[7] * exp((Lam238[0] - Lam238[2]) * t)
               + coef238[8] * exp((Lam238[0] - Lam238[3]) * t) + exp(Lam238[0] * t))
        dden = Lam238[0] / Lam238[2] * (
                coef238[7] * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                + Lam238[0] * exp(Lam238[0] * t))
        dA08dt = (dnum * den - num * dden) / den ** 2
        dtdA08 = 1. / dA08dt
        
        # dt/d(a231_235)
        num = alpha * cfg.U * f + cfg.U * (y - alpha) / x - g1
        den = Lam235[0] / Lam235[1] * (exp(Lam235[0] * t)
               - exp((Lam235[0] - Lam235[1]) * t))
        dnum = alpha * cfg.U * dfdt - dg1
        dden = Lam235[0] / Lam235[1] * (Lam235[0] * exp(Lam235[0] * t)
               - (Lam235[0] - Lam235[1]) * exp((Lam235[0] - Lam235[1]) * t))
        dA15dt = (dnum * den - num * dden) / den ** 2
        dtdA15 = 1. / dA15dt

        # covariance matrix
        Va = np.block([[V_a230_238, zeros], [zeros, V_a231_235]])
        V = np.zeros((4 * n + 1, 4 * n + 1))
        V[:-1, :-1] = np.block([[Vxy, np.zeros((2 * n, 2 * n))],
                                [np.zeros((2 * n, 2 * n)), Va]])
        V[-1, -1] = alpha_1s

        J = np.row_stack((np.diag(dtdx), np.diag(dtdy), np.diag(dtdA08),
                             np.diag(dtdA15), dtda.reshape(-1, n)))

    # age covariance matrix
    Vt = J.T @ V @ J

    return Vt


def pbu_iterative_uncert(t, ThU_min, x, Vx, ThU_melt, ThU_melt_1s,
        Th232_U238=None, V_Th232_U238=None, Pb208_206=None, V_Pb208_206=None,
        DPaU=None, DPaU_1s=None, alpha=None, alpha_1s=None, age_type='206Pb*'):
    """
    Compute uncertainties for a suite of co-genetic disequilibrium 206Pb*/238U,
    207Pb*/235U or 207Pb-corrected ages using analytical error propagation.
    ThU_min is either computed from measured 232Th/238U, or computed numerically
    from radiogenic 208Pb/206Pb and age.

    Parameters
    -----------
    t : ndarray (1-D)
        Ages.
    ThU_min : ndarray (1-D)
        (Th/U)_min. solutions.
    x : ndarray (1-D, 2-D)
        Measured isotope ratios. This should be a 1-D array for 206Pb* and
        207Pb* ages, or a 2 x n array for 207Pb-corrected ages.
    Vx : ndarray (2-D)
        Covariance matrix of uncertainties on measured isotope ratios. This
        should be an n x n array for 206Pb* and 207Pb* ages, or a 2n x 2n array
        for 207Pb-corrected ages.
    ThU_melt : float
        Th/U ratio of the melt.
    ThU_melt_1s : float
        Uncertainty (1 sigma) on Th/U ratio of the melt.
    Pb208_206 : ndarray (1-D), optional
        Radiogenic 208Pb/206Pb ratios.
    V_Pb208_206 : ndarray (2-D), optional
        Covariance matrix for radiogenic 208Pb/206Pb ratios.
    Th232_U238 : ndarray, optional
        Measured 232Th/238U ratio.
    V_Th232_U238 : ndarray, optional
        Covariance matrix for measured 232Th/238U ratios.
    alpha : float, optional
        Common 207Pb/206Pb ratio.
    alpha_1s : float, optional
        Uncertainty (1 sigma) on common 207Pb/206Pb ratio.
    age_type : {'206Pb*', '207Pb*', 'cor207Pb'}
        Age type to calculate uncertainties for.

    Returns
    --------
    Vt : ndarray (2-D)
       Covariance matrix of ages.

    """
    #TODO: implement calculations for Th/U values
    assert age_type in ('206Pb*', 'cor207Pb'), 'age_type not permitted'

    if age_type  == '206Pb*':
        if x.ndim != 1 or Vx.shape != (x.size, x.size):
            raise ValueError('inputs have incompatible dimensions')

    if age_type == 'cor207Pb':
        if x.ndim != 2 or x.shape[0] != 2 or Vx.shape != (2. * x[1].size, 2. * x[1].size):
            raise ValueError('inputs have incompatible dimensions')

    meas_Th232_U238 = False

    if Pb208_206 is not None:
        if Th232_U238 is not None:
            raise ValueError('cannot provide both Pb208_206 and Th232_U238')
        if V_Pb208_206 is None:
            raise ValueError('must provide V_Pb208_206 if Pb208_206 is given')
    elif Th232_U238 is not None:
        if V_Th232_U238 is None:
            raise ValueError('must provide V_Th232_U238 if Th232_U238 is given')
        meas_Th232_U238 = True
    else:
        raise ValueError('must provide either Pb208_206 or Th232_U238')

    n = len(x)
    zeros = np.zeros((n, n))
    czeros = np.zeros((n, 1))

    Lam238 = [cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226]
    coef238 = ludwig.bateman(Lam238)

    fThU = ThU_min / ThU_melt
    f1, f2, f3, f4 = ludwig.f_comp(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                            Lam238, coef238)
    df1, df2, df3, df4 = ludwig.dfdt_comp(t, [cfg.a234_238_eq, fThU, cfg.a226_238_eq],
                                   Lam238, coef238)

    if age_type == '206Pb*':

        if meas_Th232_U238:
            # dt/dx
            fThU = Th232_U238 / ThU_melt * exp(cfg.lam232 * t) / (
                    exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U)
            den = (exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U)
            dfThU = Th232_U238 / ThU_melt * ((cfg.lam232 * exp(cfg.lam232 * t)
                    * den - exp(cfg.lam232 * t) * (cfg.lam238 * exp(cfg.lam238 * t)
                    + cfg.lam235 * np.exp(cfg.lam235 * t) / cfg.U)) / den ** 2)
            df3 = dfThU * Lam238[0] / Lam238[2] * (coef238[7] * exp((Lam238[0] - Lam238[2]) * t)
                    + coef238[8] * exp((Lam238[0] - Lam238[3]) * t) + exp(Lam238[0] * t)) \
                    + fThU * Lam238[0] / Lam238[2] * (
                          coef238[7] * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                          + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                          + Lam238[0] * exp(Lam238[0] * t))
            dtdx = 1. / (df1 + df2 + df3 + df4)

            # dt/d(Th232_U238)
            num = x - (f1 + f2 + f4)
            dnum = - (df1 + df2 + df4)
            denu = fThU / Th232_U238
            denv = Lam238[0] / Lam238[2] * (coef238[7] * exp((Lam238[0] - Lam238[2]) * t)
                    + coef238[8] * exp((Lam238[0] - Lam238[3]) * t) + exp(Lam238[0] * t))
            ddenu = dfThU / Th232_U238
            ddenv = Lam238[0] / Lam238[2] * (coef238[7]
                        * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                        + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                        + Lam238[0] * exp(Lam238[0] * t))
            den = denu * denv
            dden = ddenu * denv + denu * ddenv
            dtdTh232U238 = den ** 2 / (dnum * den - num * dden)

            # dt/d(ThU_melt)
            numu = fThU * ThU_melt
            dnumu = dfThU * ThU_melt
            numv = Lam238[0] / Lam238[2] * (coef238[7] * exp((Lam238[0] - Lam238[2]) * t)
                        + coef238[8] * exp((Lam238[0] - Lam238[3]) * t) + exp(Lam238[0] * t))
            dnumv = Lam238[0] / Lam238[2] * (coef238[7]
                        * (Lam238[0] - Lam238[2]) * exp((Lam238[0] - Lam238[2]) * t)
                        + coef238[8] * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                        + Lam238[0] * exp(Lam238[0] * t))
            num = numu * numv
            dnum = dnumu * numv + numu * dnumv
            den = x - (f1 + f2 + f4)
            dden = -(df1 + df2 + df4)
            dtdThU_melt = den ** 2 / (dnum * den - num * dden)

            V = np.block([[Vx, zeros, czeros],
                          [zeros, V_Th232_U238, czeros],
                          [czeros.T, czeros.T, ThU_melt_1s ** 2]])
            J = np.block([[np.diag(dtdx)],
                          [np.diag(dtdTh232U238)],
                          [dtdThU_melt.reshape(1, n)]])

        else:

            c = 1. / (np.exp(cfg.lam232 * t) - 1.) * np.exp(cfg.lam232 * t) / (
                np.exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U)
            num = np.exp(cfg.lam232 * t) / (np.exp(cfg.lam232 * t) - 1.)
            dnum = -cfg.lam232 * np.exp(cfg.lam232 * t) \
                   / (np.exp(cfg.lam232 * t) - 1.) ** 2
            den = np.exp(cfg.lam238 * t) + np.exp(cfg.lam235 * t) / cfg.U
            dden = cfg.lam238 * np.exp(cfg.lam238 * t) \
                    + cfg.lam235 * np.exp(cfg.lam235 * t) / cfg.U
            dcdt =  (dnum * den - num * dden) / den ** 2

            # dt/dx
            num = f1 + f2 + f4
            dnum = df1 + df2 + df4

            denu = - Pb208_206 / ThU_melt * c
            denv = Lam238[0]/Lam238[2] * (coef238[7] * exp((Lam238[0]-Lam238[2]) * t)
                        + coef238[8] * exp((Lam238[0]-Lam238[3]) * t) + exp(Lam238[0]*t))
            den = 1. + denu * denv
            ddenv = Lam238[0] / Lam238[2] * (coef238[7] * (Lam238[0] - Lam238[2])
                     * exp((Lam238[0] - Lam238[2]) * t) + coef238[8]
                     * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                        + Lam238[0] * exp(Lam238[0] * t))
            ddenu = - Pb208_206 / ThU_melt * dcdt
            dden = ddenu * denv + denu * ddenv
            dtdx = den ** 2 / (dnum * den - num * dden)

            # dt/d(ThU_melt)
            numu = x * Pb208_206 * c
            dnumu = x * Pb208_206 * dcdt
            numv = Lam238[0] / Lam238[2] * (coef238[7] * exp((Lam238[0] - Lam238[2]) * t)
                        + coef238[8] * exp((Lam238[0] - Lam238[3]) * t) + exp(Lam238[0] * t))
            dnumv = Lam238[0] / Lam238[2] * (coef238[7] * (Lam238[0] - Lam238[2])
                         * exp((Lam238[0] - Lam238[2]) * t) + coef238[8]
                         * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                         + Lam238[0] * exp(Lam238[0] * t))
            num = numu * numv
            dnum = dnumu * numv + numu * dnumv
            den = x - (f1 + f2 + f4)
            dden = -(df1 + df2 + df4)
            dtdThU_melt = den ** 2 / (dnum * den - num * dden)

            # dt/d(Pb208_206)
            num = f1 + f2 + f4
            dnum = df1 + df2 + df3

            denu = c / ThU_melt
            ddenu = dcdt / ThU_melt
            denv = Lam238[0]/Lam238[2] * (coef238[7] * exp((Lam238[0]-Lam238[2]) * t)
                    + coef238[8] * exp((Lam238[0]-Lam238[3]) * t) + exp(Lam238[0]*t))
            ddenv = Lam238[0] / Lam238[2] * (coef238[7] * (Lam238[0] - Lam238[2])
                        * exp((Lam238[0] - Lam238[2]) * t) + coef238[8]
                        * (Lam238[0] - Lam238[3]) * exp((Lam238[0] - Lam238[3]) * t)
                        + Lam238[0] * exp(Lam238[0] * t))

            den = denu * denv
            dden = ddenu * denv + denu * ddenv
            # Pb208_206 = 1. / den - num / (x * den)
            dPb208_206dt = -dden / den ** 2 - (dnum * den - num * dden) / (x * den ** 2)
            dtdPb208_206 = 1. / dPb208_206dt

            V = np.block([[Vx, zeros, czeros],
                          [zeros, V_Pb208_206, czeros],
                          [czeros.T, czeros.T, ThU_melt_1s ** 2]])
            J = np.block([[np.diag(dtdx)],
                          [np.diag(dtdPb208_206)],
                          [dtdThU_melt.reshape(1, n)]])

    elif age_type == 'cor207Pb':
        raise ValueError('analytical uncertainties not yet implemented for iterative '
                         'cor-207Pb ages')

    # age covariance matrix
    Vt = J.T @ V @ J

    return Vt

