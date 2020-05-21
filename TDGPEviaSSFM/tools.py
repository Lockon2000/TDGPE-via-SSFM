"""
"""

import numpy as np
import scipy.fft
import scipy.integrate


def fft(*arg, **kwarg):
    return scipy.fft.fftshift(scipy.fft.fft(*arg, **kwarg))


def ifft(*arg, **kwarg):
    return scipy.fft.fftshift(scipy.fft.ifft(*arg, **kwarg))


def getkVector(arrayLength, sampleSpacing):
    return (2 * np.pi) * scipy.fft.fftshift(
        scipy.fft.fftfreq(arrayLength, sampleSpacing)
    )


def probDensity(psi):
    return np.absolute(psi) ** 2


def computeTotalProbability(x, psi_x):
    return scipy.integrate.simps(probDensity(psi_x), x)


def computeKineticEnergy(x, psi_x, m):
    from .configs import unitSystem

    if unitSystem == "natural":
        from .constants import FundamentalNat
        hbar = FundamentalNat.hbar.value
    elif unitSystem == "SI":
        from .constants import FundamentalSI
        hbar = FundamentalSI.hbar.value

    dx = x[1] - x[0]
    kineticTerm = (hbar**2 / (2 * m)) * probDensity(np.gradient(psi_x, dx))

    return scipy.integrate.simps(kineticTerm, x)


def computeExternalPotentialEnergy(x, psi_x, V):
    externalPotentialTerm = V * probDensity(psi_x)

    return scipy.integrate.simps(externalPotentialTerm, x)


def computeInternalPotentialEnergy(x, psi_x, kappa):
    internalPotentialTerm = 0.5 * kappa * probDensity(psi_x) ** 2

    return scipy.integrate.simps(internalPotentialTerm, x)


def computeTotalEnergy(x, psi_x, V, kappa, m):
    totalEnergy = (
        computeKineticEnergy(x, psi_x, m) +
        computeExternalPotentialEnergy(x, psi_x, V) +
        computeInternalPotentialEnergy(x, psi_x, kappa)
    )

    return totalEnergy
