"""
Pb/U equations assuming disequilibrium in a single nuclide only.


"""

import numpy as np


def f(t, fThU, l8, l0):
    """
    206Pb/238U atomic ratio as a function of time and actvitiy
    ratios using equation from Guillong et al. (2014).

    Notes
    -----
    This equation assumes that 226Ra is always in radioactive
    equilibrium with 230Th. Where [226Ra/238U]i does not equal [230Th/238U]i,
    this equation may be inaccurate.

    References
    ----------
    Guillong, M., von Quadt, A., Sakata, S., Peytcheva, I., Bachmann, O., 2014.
    LA-ICP-MS Pb-U dating of young zircons from the Kos–Nisyros volcanic centre,
    SE aegean arc. Journal of Analytical Atomic Spectrometry 29, 963–970.
    https://doi.org/10.1039/C4JA00009A

    """
    return np.exp(l8 * t) - 1. + l8 / l0 * (fThU - 1) \
           * (1 - np.exp(-l0 * t)) * np.exp(l8 * t)


def g(t, fPaU, l5, l1):
    """
    207Pb/235U atomic ratio as a function of time and actvitiy
    ratios using equation from Sakata et al. (2018).

    References
    -----------
    Sakata, S., Hirakawa, S., Iwano, H., Danhara, T., Guillong, M., Hirata, T.,
    2017. A new approach for constraining the magnitude of initial
    disequilibrium in Quaternary zircons by coupled uranium and thorium decay
    series dating. Quaternary Geochronology 38, 1–12.
    https://doi.org/10.1016/j.quageo.2016.11.002

    """
    return (np.exp(l5 * t) - 1.) + l5 / l1 * (fPaU - 1) \
           * (1 - np.exp(-l1 * t)) * np.exp(l5 * t)
