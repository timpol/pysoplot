"""
Disequlibrium Pb/U functions based on Wendt and Carl (1985).

"""

import numpy as np

from pysoplot import cfg
from pysoplot import useries


def f(t, A, DC, init=(True, True)):
    """
    206Pb/236U atomic ratio as a function of time and actvitiy
    ratios using equation from [Wendt1985]_.

    References
    ----------
    .. [Wendt1985] Wendt, I., Carl, C., 1985. U/Pb dating of discordant 0.1
       Ma old secondary U minerals. Earth and Planetary Science Letters 73, 278–284.

    """
    assert cfg.secular_eq

    # Get initial activity ratios if present value given.
    A48i = A[0] if init[0] else useries.ar48i(t, A[0], *DC[:2])
    A08i = A[1] if init[1] else useries.ar08i(t, A48i, A[1], *DC[:3], init=True)

    A0 = A48i - 1
    B0 = A08i - 1
    C0 = A[2] - 1

    K1 = -A0 * DC[0] / DC[1] * ((DC[2] * DC[3]) * (DC[2] - DC[1]) * (DC[3] - DC[1]))
    K2 = DC[0] * DC[3] / (DC[3] - DC[2]) * (A0 / (DC[2] - DC[1]) - B0 / DC[2])
    K3 = DC[0] / (DC[3] - DC[2]) * (B0 - A0 * DC[2] / (DC[3] - DC[1])) - C0 * (DC[0] / DC[3])
    K4 = A0 * (DC[0] / DC[1]) + B0 * (DC[0] / DC[2]) + C0 * (DC[0] / DC[3])

    f2 = (np.exp(DC[0] * t) - 1) + np.exp(DC[0] * t) * (K1 * np.exp(-DC[1] * t)
            + K2 * np.exp(-DC[2] * t) + K3 * np.exp(-DC[3] * t) + K4)

    return f2


def g(t, A15i, DC):
    """
    207Pb/235U atomic ratio as a function of time and actvitiy
    ratios using equation from [Wendt1985]_.

    References
    ----------
    .. [Wendt1985] Wendt, I., Carl, C., 1985. U/Pb dating of discordant 0.1
       Ma old secondary U minerals. Earth and Planetary Science Letters 73, 278–284.
    """
    D0 = A15i - 1.
    f1 = np.exp(DC[0] * t) - 1 + D0 * (DC[0] / DC[1]) * np.exp(DC[0] * t) \
         * (1 - np.exp(-DC[1] * t))
    return f1
