"""
"Classical" isochron ages.

"""

import numpy as np

from . import monte_carlo as mc
from . import cfg


diagram_names = {'iso-Pb6U8': '238U-206Pb isochron age',
                 'iso-Pb7U5': '235U-207Pb isochron age'}


def age(b, sb=None, age_type='iso-Pb6U8', dc_errors=False):
    """
    Calculate classical isochron age.
    """
    assert age_type in ('iso-Pb6U8', 'iso-Pb7U5')
    if age_type == 'iso-Pb6U8':
        dc = cfg.lam238
        sdc = cfg.s238
    elif age_type == 'iso-Pb7U5':
        dc = cfg.lam235
        sdc = cfg.s235
    t = 1. / dc * np.log(b + 1.)
    if sb is None:
        return t
    sdc = 0 if dc_errors else sdc
    db = 1. / (dc * (b + 1.))
    ddc = 1. / ((dc ** 2) * np.log(b + 1.)) if dc_errors else 0.
    st = np.sqrt((ddc * sdc) ** 2 + (db * sb) ** 2)
    return t, st


def mc_uncert(fit, age_type='iso-68', dc_errors=False, norm_isotope='204Pb',
              trials=50_000, hist=False):
    """
    Compute classical isochron age uncertainties using Monte Carlo approach.
    """
    flags = np.zeros(trials)
    t = age(fit['theta'][1], age_type='iso-Pb6U8')

    if age_type == 'iso-Pb6U8':
        dc = cfg.lam238
        sdc = cfg.s238
    elif age_type == 'iso-Pb7U5':
        dc = cfg.lam235
        sdc = cfg.s235

    # draw randomised slope / intercept values
    theta, flags = mc.draw_theta(fit, trials, flags)

    # draw randomised decay constant values
    if dc_errors:
        dc = cfg.rng.normal(dc, sdc, trials)

    # simulate ages
    ts = 1. / dc * np.log(theta[1] + 1.)
    flags = mc.update_flags(flags, ts < 0, mc.NEGATIVE_AGE)

    ok = flags == 0
    age_95ci = np.quantile(ts[ok], (0.025, 0.975))
    results = {
        'age_type': diagram_names[age_type],
        'normalising_isotope': norm_isotope,
        'age_1s': np.std(ts[ok]),
        'age_95ci': age_95ci,
        'age_95pm': np.nanmean([t - age_95ci[0], age_95ci[1] - t]),
        'mean_age': np.nanmean(ts[ok]),
        'median_age': np.nanmedian(ts[ok]),
        'y-int_1s': np.nanmean(theta[0][ok]),
        'y-int_95ci': np.nanquantile(theta[0][ok], (0.025, 0.975)),
        'mean_y-int': np.nanmean(theta[0][ok]),
        'median_y-int': np.nanmedian(theta[0][ok]),
        'trials': trials,
        'fails': np.sum(flags != 0),
        'not_converged': np.sum(flags == mc.NON_CONVERGENCE),
        'negative_ages': np.sum(flags == mc.NEGATIVE_AGE)
    }

    if hist:
        fig = mc.age_hist(ts, age_type, a=theta[1], b=theta[0])
        results['age_hist'] = fig

    return results
