"""
"""

import numpy as np

from .tools import fft
from .tools import ifft
from .tools import getNyquistFreqRange


def ssfm(x, dt, N_t, psi_x_0, N_op_func, m):
    dx = x[1] - x[0]
    k = getNyquistFreqRange(x.size, dx)
    psi_x = psi_x_0

    for _ in range(N_t):
        # Calculate the first half time step (hts)
        psi_x_hts = np.exp(1j*(dt/2)*N_op_func(psi_x, x, t))*psi_x
        # Calculate the full space step (fss)
        psi_x_fss = ifft(np.exp(1j*dt*(((1j*2*np.pi*k)**2)/2*m))*fft(psi_x_hts))
        # Calculate the second half time step
        psi_x = np.exp(1j*(dt/2)*N_op_func(psi_x, x, t))*ifft(psi_x_fss)
    
    psi_k = fft(psi_x)
    
    return psi_x, psi_k, k

