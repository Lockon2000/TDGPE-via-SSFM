"""
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import IPython.core.display as IPython_display
import pycav.display as pycav_display

from .tools import probDensity
from .tools import computeTotalProbability
from .tools import computeTotalEnergy
from .units import *


def animateEvolution(x, k, psi_x_frames, psi_k_frames, V_frames, kappa, m, furtherInfo):
    from .configs import dt
    from .configs import N_t
    from .configs import skippingFactor
    from .configs import partFactor
    from .configs import secsToMsecsConversionFactor
    from .configs import savePath

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

    probDens_psi_k_frames = probDensity(psi_k_frames)

    # Initial Data Calculations #

    totalProb_psi_x = computeTotalProbability(x, psi_x_frames[0])
    totalEnergy = computeTotalEnergy(x, psi_x_frames[0], V_frames[0], kappa, m)

    # Plot Creation and Configuration #

    fig, ax = plt.subplots(2)

    # Injection of Plot Information #

    fig.suptitle("$\\Psi(x,t)$ and $\\tilde{\\Psi}(k,t)$", fontsize=16)
    infoText_tl =     fig.text(
        0.01,
        0.925,
        f"Elapsed Time = 0 ${unit_t['symbol']}$\n"
        f"Total Probabilty = {totalProb_psi_x:>7.2%}\n"
        f"Total Energy = {totalEnergy*unit_E['conversionFactor']:.4G} ${unit_E['symbol']}$",
        fontsize="large",
    )
    fig.text(
        0.9,
        0.925,
        f"$m$ = {m*unit_m['conversionFactor']:.4G} ${unit_m['symbol']}$\n"
        f"$\\kappa$ = {kappa:.4G}",
        fontsize="large",
    )
    fig.text(
        0.01,
        0.015,
        f"$unit_{{x}}$ = {unit_l['conversionFactor']:.5G} ${unit_l['symbol']}$, $dx$ = {dx:.5G}, "
        f"$x_{{size}}$ = {xSize}, $x_{{boundary}}$ = {xBoundary:.5G}, "
        f"$x_{{range}}$ = {xRange:.5G} = {xRange*unit_l['conversionFactor']:.5G} ${unit_l['symbol']}$\n"
        f"$unit_{{k}}$ = {unit_k['conversionFactor']:.5G} ${unit_k['symbol']}$, $dk$ = {dk:.5G}, "
        f"$k_{{size}}$ = {kSize}, $k_{{boundary}}$ = {kBoundary:.5G}, "
        f"$k_{{range}}$ = {kRange:.5G} = {kRange*unit_k['conversionFactor']:.5G} ${unit_k['symbol']}$\n"
        f"$unit_{{t}}$ = {unit_t['conversionFactor']:.5G} ${unit_t['symbol']}$, $dt$ = {dt:.5G}, $N_t$ = {N_t}, "
        f"$t_{{range}}$ = {tRange:.5G} = {tRange*unit_t['conversionFactor']:.5G} ${unit_t['symbol']}$",
        fontsize="large",
    )
    fig.text(
        0.6, 0.015, " | ".join(f"{key}: {value}" for key, value in furtherInfo.items())
    )

    # Data Plotting #

    ax[0].set(title="Position Space", xlabel="$x$", ylabel="$\\Psi(x,t)$ and $V(x,t)$")
    x_min, x_max = 1.1 * x.min(), 1.1 * x.max()
    psi_x_min, psi_x_max = (
        -(2 * np.absolute(psi_x_frames).max()),
        2 * np.absolute(psi_x_frames).max(),
    )
    ax[0].axis([x_min, x_max, psi_x_min, psi_x_max])
    line0_0 = ax[0].plot(x, psi_x_frames[0].imag, label="$im(\\Psi_x(x,t))$")[0]
    line0_1 = ax[0].plot(x, psi_x_frames[0].real, label="$re(\\Psi_x(x,t))$")[0]
    line0_2 = ax[0].plot(x, np.absolute(psi_x_frames[0]), label="$|\\Psi_x(x,t)|$")[0]
    line0_3 = ax[0].plot(x, V_frames[0], label="$V(x,t)$")[0]
    ax[0].legend(loc="best")

    ax[1].set(title="k Space", xlabel="$k$", ylabel="$|\\tilde{\\Psi}(k,t)|^2$")
    k_min, k_max = 1.1 * k.min(), 1.1 * k.max()
    probDens_k_min, probDens_k_max = 0, 1.25 * probDens_psi_k_frames.max()
    ax[1].axis([k_min, k_max, probDens_k_min, probDens_k_max])
    line1_0 = ax[1].plot(
        k, probDens_psi_k_frames[0], label="$|\\tilde{\\Psi}(k,t)|^2$"
    )[0]
    ax[1].legend(loc="best")

    def nextframe(n_t):
        totalProb_psi_x = computeTotalProbability(x, psi_x_frames[skippingFactor * n_t])
        totalEnergy = computeTotalEnergy(
            x,
            psi_x_frames[skippingFactor * n_t],
            V_frames[skippingFactor * n_t],
            kappa,
            m,
        )
        infoText_tl.set_text(
            f"Elapsed Time = {dt*n_t*skippingFactor*unit_t['conversionFactor']:.4G} ${unit_t['symbol']}$\n"
            f"Total Probabilty = {totalProb_psi_x:>7.2%}\n"
            f"Total Energy = {totalEnergy*unit_E['conversionFactor']:.4G} ${unit_E['symbol']}$",
        )

        line0_0.set_data(x, psi_x_frames[skippingFactor * n_t].imag)
        line0_1.set_data(x, psi_x_frames[skippingFactor * n_t].real)
        line0_2.set_data(x, np.absolute(psi_x_frames[skippingFactor * n_t]))
        line0_3.set_data(x, V_frames[skippingFactor * n_t])

        line1_0.set_data(k, np.absolute(psi_k_frames[skippingFactor * n_t]) ** 2)

    animation = anim.FuncAnimation(
        fig,
        nextframe,
        interval=dt * secsToMsecsConversionFactor * skippingFactor,
        frames=int((N_t * partFactor) // skippingFactor),
        repeat=False,
    )
    displayable_animation = pycav_display.create_animation(
        animation, fname=eval(f'f"{savePath}"'),
    )

    IPython_display.display(
        pycav_display.display_animation(displayable_animation)
    ) if displayable_animation is not None else None
