import numpy as np
import scipy.fft
import scipy.integrate


def fft(*arg, **kwarg):
    return scipy.fft.fftshift(scipy.fft.fft(*arg, **kwarg))

def ifft(*arg, **kwarg):
    return scipy.fft.fftshift(scipy.fft.ifft(*arg, **kwarg))

def getNyquistFreqRange(arrayLength, sampleSpacing):
    return (2*np.pi)*scipy.fft.fftshift(scipy.fft.fftfreq(arrayLength, sampleSpacing))


def probDensity(psi):
    return np.absolute(psi)**2

def computeTotalProbability(x, psi_x):
    return scipy.integrate.simps(probDensity(psi_x), x)

def computeTotalEnergy(x, psi_x, V, m, kappa):
    dx = x[1]-x[0]
    term1 = (1/(2*m))*probDensity(np.gradient(psi_x, dx))
    term2 = V*probDensity(psi_x)
    term3 = 0.5*kappa*probDensity(psi_x)**2
    return scipy.integrate.simps(term1 + term2 + term3, x)
