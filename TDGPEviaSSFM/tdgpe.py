"""
"""

import numpy as np
import scipy.fft

from .tools import getkVector
from .tools import probDensity
from .tools import fft
from .tools import ifft


def ssfm(x, psi_x_0, V_func, kappa, m):
    from .configs import dt
    from .configs import N_t
    from .configs import unitSystem

    if unitSystem == "natural":
        from .constants import FundamentalNat
        hbar = FundamentalNat.hbar.value
    elif unitSystem == "SI":
        from .constants import FundamentalSI
        hbar = FundamentalSI.hbar.value


    def _N_op_func(psi_x, x, t):
        return -(V_func(x, t) + kappa * probDensity(psi_x))/hbar

    dx = x[1] - x[0]
    # Get the k vector arranged as normally returned by scipy
    k = (2 * np.pi) * scipy.fft.fftfreq(x.size, dx)
    psi_x_frames = np.empty((N_t, x.size), dtype=np.complex128)
    psi_x_frames[0] = psi_x_0
    psi_k_frames = np.empty((N_t, x.size), dtype=np.complex128)
    psi_k_frames[0] = fft(psi_x_0)
    V_frames = np.empty((N_t, x.size), dtype=np.float64)
    V_frames[0] = V_func(x, 0)

    for i in range(N_t - 1):
        # Calculate the first half time step with the N-op (Nhs)
        psi_x_Nhs = (
            np.exp(0.5j * dt * _N_op_func(psi_x_frames[i], x, i * dt)) * psi_x_frames[i]
        )
        # Calculate the full time step with the L-op (Lfs)
        psi_x_Lfs = scipy.fft.ifft(
            np.exp(-0.5j * hbar * dt * k ** 2 / m) * scipy.fft.fft(psi_x_Nhs)
        )
        # Calculate the second half time step with the N-op
        psi_x_frames[i + 1] = (
            np.exp(0.5j * dt * _N_op_func(psi_x_Lfs, x, i * dt)) * psi_x_Lfs
        )

        # Calculate the k space wave function
        psi_k_frames[i + 1] = fft(psi_x_frames[i + 1])

        # Fill the V_frames array
        V_frames[i + 1] = V_func(x, (i + 1) * dt)

    # Get the k vector arranged with the zero point in the middle
    k = getkVector(x.size, dx)

    return k, psi_x_frames, psi_k_frames, V_frames
