"""
TDGPEviaSSFM

A python package to solve the time-dependent Gross–Pitaevskii equation numerically using the Split-Step Fourier method.
"""

__version__ = "0.9.0"


import numpy as np

from . import configs
from . import tools
from . import tdgpe
from . import interpolation
from . import plotting
from . import animating
from . import constants


def configure(**options):
    for option, value in options.items():
        if option in [
            "dt",
            "N_t",
            "skippingFactor",
            "partFactor",
            "secsToMsecsConversionFactor",
            "savePath",
            "unitSystem",
            "smoothingParameters"
        ]:
            configs.__dict__[option] = value


def solveSE(x, psi_x_0, V_func, kappa, m, furtherInfo, staticPlots=True, animation=True, dry=False):
    if dry:
        dx = x[1] - x[0]

        plotting.plotState(
            x,
            tools.getkVector(x.size, dx),
            psi_x_0,
            tools.fft(psi_x_0),
            V_func(x, 0),
            kappa,
            m,
            f"Initial Plot of $\\Psi(x,t)$ and $\\tilde{{\\Psi}}(k,t)$",
            furtherInfo,
        )

        return

    from .configs import N_t

    k, psi_x_frames, psi_k_frames, V_frames = tdgpe.ssfm(
        x, psi_x_0, V_func, kappa, m)

    N = N_t - 1
    if staticPlots:
        if type(staticPlots) == list:
            for factor in staticPlots:
                index = int(factor * N)
                plotting.plotState(
                    x,
                    k,
                    psi_x_frames[index],
                    psi_k_frames[index],
                    V_frames[index],
                    kappa,
                    m,
                    f"$\\Psi(x,t)$ and $\\tilde{{\\Psi}}(k,t)$ at {factor} of time",
                    furtherInfo,
                )
        elif type(staticPlots) == bool:
            plotting.plotState(
                x,
                k,
                psi_x_frames[0],
                psi_k_frames[0],
                V_frames[0],
                kappa,
                m,
                None,
                furtherInfo,
            )
        else:
            raise ValueError(
                "Wrong value for staticPlots! Only bool or list are allowed"
            )

    if animation:
        animating.animateEvolution(
            x, k, psi_x_frames, psi_k_frames, V_frames, kappa, m, furtherInfo
        )

    return k, psi_x_frames, psi_k_frames, V_frames


def calculateKineticEnergyCourse(x, dt, psi_x_frames, m):
    t = np.arange(psi_x_frames.shape[0]) * dt

    K = np.array(list(map(lambda psi_x: tools.computeKineticEnergy(
        x, psi_x, m), psi_x_frames)))

    return t, K


def calculateExternalPotentialEnergyCourse(x, dt, psi_x_frames, V_frames):
    t = np.arange(psi_x_frames.shape[0]) * dt
    data = np.concatenate((psi_x_frames, V_frames), axis=1).reshape(
        (psi_x_frames.shape[0], 2, psi_x_frames.shape[1]))

    V_ext = np.array(list(map(lambda psi_x__and__V: tools.computeExternalPotentialEnergy(
        x, psi_x__and__V[0], psi_x__and__V[1].real), data)))

    return t, V_ext


def calculateInternalPotentialEnergyCourse(x, dt, psi_x_frames, kappa):
    t = np.arange(psi_x_frames.shape[0]) * dt

    V_int = np.array(list(map(lambda psi_x: tools.computeInternalPotentialEnergy(
        x, psi_x, kappa), psi_x_frames)))

    return t, V_int


def calculateTotalEnergyCourse(x, dt, psi_x_frames, V_frames, kappa, m):
    t = np.arange(psi_x_frames.shape[0]) * dt
    data = np.concatenate((psi_x_frames, V_frames), axis=1).reshape(
        (psi_x_frames.shape[0], 2, psi_x_frames.shape[1]))

    E = np.array(list(map(lambda psi_x__and__V: tools.computeTotalEnergy(
        x, psi_x__and__V[0], psi_x__and__V[1].real, kappa, m), data)))

    return t, E
