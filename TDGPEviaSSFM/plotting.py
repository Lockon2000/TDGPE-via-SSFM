"""
"""

import numpy as np
import matplotlib.pyplot as plt

from .tools import probDensity
from .tools import computeTotalProbability
from .tools import computeTotalEnergy


def plotState(x, k, psi_x, psi_k, V, kappa, m, title=None, furtherInfo={}):
    from .configs import dt
    from .configs import N_t

    # Parameter Calculations #

    dx = x[1] - x[0]
    xBoundary = x[-1]
    xRange = x[-1] - x[0]
    xSize = x.size

    dk = k[1] - k[0]
    kBoundary = k[-1]
    kRange = k[-1] - k[0]
    kSize = k.size

    tRange = dt * N_t

    probDens_psi_k = probDensity(psi_k)
    totalProb_psi_x = computeTotalProbability(x, psi_x)
    totalEnergy = computeTotalEnergy(x, psi_x, V, kappa, m)

    # Plot Creation and Configuration #

    fig, ax = plt.subplots(2)

    # Injection of Plot Information #

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("$\\Psi(x,t)$ and $\\tilde{\\Psi}(k,t)$", fontsize=16)

    fig.text(
        0.01,
        0.925,
        f"Total Probabilty = {totalProb_psi_x:.3f}\n"
        f"Total Energy = {totalEnergy:.3f}",
        fontsize="large",
    )
    fig.text(
        0.9,
        0.925,
        f"$m = {m}$\n"
        f"$\\kappa = {kappa:.2e}$",
        fontsize="large",
    )
    fig.text(
        0.01,
        0.015,
        f"$dx = {dx:.3f}, x_{{boundary}} = {xBoundary:.3f}, x_{{range}} = {xRange:.3f}, x_{{size}} = {xSize}$\n"
        f"$dk = {dk:.3f}, k_{{boundary}} = {kBoundary:.3f}, k_{{range}} = {kRange:.3f}, k_{{size}} = {kSize}$\n"
        f"$dt = {dt:.3f}, t_{{range}} = {tRange:.3f}, N_t = {N_t}$",
        fontsize="large",
    )
    fig.text(
        0.4, 0.015, " | ".join(f"{key}: {value}" for key, value in furtherInfo.items())
    )

    # Data Plotting #

    ax[0].set(title="Position Space", xlabel="$x$", ylabel="$\\Psi(x,t)$ and $V(x,t)$")
    x_min, x_max = 1.1 * x.min(), 1.1 * x.max()
    psi_x_min, psi_x_max = -(2 * np.absolute(psi_x).max()), 2 * np.absolute(psi_x).max()
    ax[0].axis([x_min, x_max, psi_x_min, psi_x_max])
    ax[0].plot(x, psi_x.imag, label="$im(\\Psi(x,t))$")[0]
    ax[0].plot(x, psi_x.real, label="$re(\\Psi(x,t))$")[0]
    ax[0].plot(x, np.absolute(psi_x), label="$|\\Psi(x,t)|$")[0]
    ax[0].plot(x, V, label="$V(x,t)$")[0]
    ax[0].legend(loc="best")

    ax[1].set(title="k Space", xlabel="$k$", ylabel="$|\\tilde{\\Psi}(k,t)|^2$")
    k_min, k_max = 1.1 * k.min(), 1.1 * k.max()
    prob_dens_k_min, prob_dens_k_max = 0, 1.25 * probDens_psi_k.max()
    ax[1].axis([k_min, k_max, prob_dens_k_min, prob_dens_k_max])
    ax[1].plot(k, probDens_psi_k, label="$|\\tilde{\\Psi}(k,t)|^2$")[0]
    ax[1].legend(loc="best")

    plt.show()
