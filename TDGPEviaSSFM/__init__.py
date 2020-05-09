"""
TDGPEviaSSFM

A python package to solve the time-dependent Gross–Pitaevskii equation numerically using the Split-Step Fourier method.
"""

__version__ = "0.3.0"


from . import configs
from . import tools
from . import tdgpe
from . import interpolation
from . import plotting
from . import animation


def configure(**options):
    for option, value in options.items():
        if option in [
            "dt",
            "N_t",
            "skippingFactor",
            "partFactor",
            "secsToMsecsConversionFactor",
            "savePath",
        ]:
            configs.__dict__[option] = value


def solveSE(x, psi_x_0, V_func, kappa, m, furtherInfo, staticPlots=True):
    k, psi_x_frames, psi_k_frames, V_frames = tdgpe.ssfm(x, psi_x_0, V_func, kappa, m)

    N = x.size
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

    animation.animateEvolution(
        x, k, psi_x_frames, psi_k_frames, V_frames, kappa, m, furtherInfo
    )
