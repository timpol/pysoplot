"""
"Classical" isochron ages.

"""

import numpy as np

from . import mc as mc
from . import cfg


diagram_names = {'iso-206Pb': '238U-206Pb isochron age',
                 'iso-207Pb': '235U-207Pb isochron age'}


def age(b, sb=None, age_type='iso-206Pb', dc_errors=False):
    """
    Calculate classical isochron age.

    Parameters
    -----------
    b : float
        Linear regression slope
    sb : float, optional
        Uncertaintiy in regression slope (:math:`1\sigma`)
    age_type : {'iso-Pb6U8', 'iso-Pb7U5'}
        Isochron age type.

    """
    assert age_type in ('iso-206Pb', 'iso-207Pb')
    if age_type == 'iso-206Pb':
        lam = cfg.lam238
        slam = cfg.s238
    elif age_type == 'iso-207Pb':
        lam = cfg.lam235
        slam = cfg.s235
    t = 1. / lam * np.log(b + 1.)
    if sb is None:
        return t
    slam = 0 if dc_errors else slam
    db = 1. / (lam * (b + 1.))
    dlam = 1. / ((lam ** 2) * np.log(b + 1.)) if dc_errors else 0.
    st = np.sqrt((dlam * slam) ** 2 + (db * sb) ** 2)
    return t, st


def mc_uncert(fit, age_type='iso-206Pb', dc_errors=False, norm_isotope='204Pb',
              trials=50_000, hist=False):
    """
    Compute classical isochron age uncertainties using Monte Carlo approach.

    Compute Monte Carlo age uncertainties for equilibrium concordia intercept
    age.

    Parameters
    -----------
    fit : dict
        Linear regression fit parameters.
    age_type : {'iso-Pb6U8', 'iso-Pb7U5'}
        Isochron age type.
    trials : int
        Number of Monte Carlo trials.

    """
    failures = np.zeros(trials)
    t = age(fit['theta'][1], age_type=age_type)

    if age_type == 'iso-206Pb':
        lam = cfg.lam238
        slam = cfg.s238
    elif age_type == 'iso-207Pb':
        lam = cfg.lam235
        slam = cfg.s235

    # draw randomised slope / intercept values
    theta, failures = mc.draw_theta(fit, trials, failures)

    # draw randomised decay constant values
    if dc_errors:
        lam = cfg.rng.normal(lam, slam, trials)

    # simulate ages
    ts = 1. / lam * np.log(theta[1] + 1.)
    failures = mc.check_ages(ts, ~np.isnan(ts), failures, negative_ages=False)

    ok = (failures == 0)
    if np.sum(ok) == 0:
        raise ValueError('no successful Monte Carlo simulations')

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
        'fails': np.sum(failures != 0),
        'not_converged': np.sum(failures == mc.NON_CONVERGENCE),
        'negative_ages': np.sum(failures == mc.NEGATIVE_AGE)
    }

    if hist:
        fig = mc.age_hist(ts, age_type, a=theta[1], b=theta[0])
        results['age_hist'] = fig

    return results
