"""
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import IPython.core.display as IPython_display
import pycav.display as pycav_display

from .tools import probDensity
from .tools import computeTotalProbability
from .tools import computeTotalEnergy


def animateEvolution(x, k, psi_x_frames, psi_k_frames, V_frames, kappa, m, furtherInfo):
    from .configs import dt
    from .configs import N_t
    from .configs import skippingFactor
    from .configs import partFactor
    from .configs import secsToMsecsConversionFactor
    from .configs import savePath
    from .configs import unitSystem
    from .configs import smoothingParameters

    if unitSystem == "SI":
        from .siunits import unit_l, unit_t, unit_m, unit_v, unit_p, unit_k, unit_E
    elif unitSystem == "natural":
        from .natunits import unit_l, unit_t, unit_m, unit_v, unit_p, unit_k, unit_E

    # Parameter Calculations #

    dx = x[1] - x[0]
    sx = x / dx
    unit_sl = {
        "conversionFactor": unit_l["conversionFactor"] * dx,
        "symbol": unit_l["symbol"],
    }
    dsx = 1  # Trivial, as we scale down or up to make this always the case
    sxBoundary = sx[-1]
    sxRange = sx[-1] - sx[0]
    sxSize = sx.size

    dk = k[1] - k[0]
    kBoundary = k[-1]
    kRange = k[-1] - k[0]
    kSize = k.size

    tRange = dt * N_t

    # Initial Data Calculations #

    smoothPsi_x_frames = scipy.signal.savgol_filter(
        np.absolute(psi_x_frames), smoothingParameters[0]*2+1, smoothingParameters[1])
    probDens_psi_k_frames = probDensity(psi_k_frames)
    totalProb_psi_x = computeTotalProbability(x, psi_x_frames[0])
    totalEnergy = computeTotalEnergy(x, psi_x_frames[0], V_frames[0], kappa, m)

    # Plot Creation and Configuration #

    fig, ax = plt.subplots(3)
    ax_0_1 = ax[0].twinx()
    ax_1_1 = ax[1].twinx()

    # Injection of Plot Information #

    fig.suptitle("$\\Psi(x,t)$ and $\\tilde{\\Psi}(k,t)$", fontsize=16)
    infoText_tl = fig.text(
        0.01,
        0.925,
        f"$t$ = 0 ${unit_t['symbol']}$\n"
        f"$\\langle\\Psi|\\Psi\\rangle$ = {totalProb_psi_x:>.4G}\n"
        f"$E$ = {totalEnergy*unit_E['conversionFactor']:.7G} ${unit_E['symbol']}$",
        fontsize="large",
    )
    fig.text(
        0.85,
        0.925,
        f"$m$ = {m*unit_m['conversionFactor']:.4G} ${unit_m['symbol']}$\n"
        f"$\\kappa$ = {kappa:.4G}",
        fontsize="large",
    )
    fig.text(
        0.01,
        0.015,
        f"$unit_{{x}}$ = {unit_sl['conversionFactor']:.5G} ${unit_sl['symbol']}$, $dx$ = {dsx:.5G}, "
        f"$x_{{size}}$ = {sxSize}, $x_{{boundary}}$ = {sxBoundary:.5G}, "
        f"$x_{{range}}$ = {sxRange:.5G} = {sxRange*unit_sl['conversionFactor']:.5G} ${unit_sl['symbol']}$\n"
        f"$unit_{{k}}$ = {unit_k['conversionFactor']:.5G} ${unit_k['symbol']}$, $dk$ = {dk:.5G}, "
        f"$k_{{size}}$ = {kSize}, $k_{{boundary}}$ = {kBoundary:.5G}, "
        f"$k_{{range}}$ = {kRange:.5G} = {kRange*unit_k['conversionFactor']:.5G} ${unit_k['symbol']}$\n"
        f"$unit_{{t}}$ = {unit_t['conversionFactor']:.5G} ${unit_t['symbol']}$, $dt$ = {dt:.5G}, $N_t$ = {N_t}, "
        f"$t_{{range}}$ = {tRange:.5G} = {tRange*unit_t['conversionFactor']:.5G} ${unit_t['symbol']}$",
        fontsize="large",
    )
    fig.text(
        0.45,
        0.015,
        " | ".join(
            f"{key}: {value}" if type(value) == str else f"{key}: {value:.5G}"
            for key, value in furtherInfo.items()
        ),
    )

    # Data Plotting #

    # First Subplot

    sx_min, sx_max = 1.1 * sx.min(), 1.1 * sx.max()
    psi_x_min, psi_x_max = (
        -2 * np.absolute(psi_x_frames).max(),
        2 * np.absolute(psi_x_frames).max(),
    )
    if psi_x_min == 0 and psi_x_max == 0:
        psi_x_min, psi_x_max = -1, 1
    ax[0].set(title="Position Space", xlabel="$x$", ylabel="$\\Psi(x,t)$")
    ax[0].axis([sx_min, sx_max, psi_x_min, psi_x_max])
    line0_0 = ax[0].plot(
        sx, psi_x_frames[0].imag, label="$im(\\Psi_x(x,t))$", color="blue"
    )[0]
    line0_1 = ax[0].plot(
        sx, psi_x_frames[0].real, label="$re(\\Psi_x(x,t))$", color="goldenrod"
    )[0]
    line0_2 = ax[0].plot(
        sx, np.absolute(psi_x_frames[0]), label="$|\\Psi_x(x,t)|$", color="green"
    )[0]
    ax[0].legend(loc="upper left")

    V_min, V_max = -(1 / 0.95) * np.absolute(V_frames).max(), (1 /
                                                               0.95) * np.absolute(V_frames).max()
    if V_min == 0 and V_max == 0:
        V_min, V_max = -1, 1
    ax_0_1.set_ylabel("$V(x,t)$")
    ax_0_1.axis([sx_min, sx_max, V_min, V_max])
    line0_4 = ax_0_1.plot(
        sx, V_frames[0], label="$V(x,t)$", color="firebrick")[0]
    ax_0_1.legend(loc="upper right")

    # Second Subplot

    ax[1].set(title="Position Space", xlabel="$x$", ylabel="$\\Psi(x,t)$")
    ax[1].axis([sx_min, sx_max, psi_x_min, psi_x_max])
    line1_0 = ax[1].plot(
        sx, np.absolute(smoothPsi_x_frames[0]), label="Smooth $|\\Psi_x(x,t)|$", color="green"
    )[0]
    ax[1].legend(loc="upper left")

    ax_1_1.set_ylabel("$V(x,t)$")
    ax_1_1.axis([sx_min, sx_max, V_min, V_max])
    line1_1 = ax_1_1.plot(
        sx, V_frames[0], label="$V(x,t)$", color="firebrick")[0]
    ax_1_1.legend(loc="upper right")

    # Third Subplot

    k_min, k_max = 1.1 * k.min(), 1.1 * k.max()
    probDens_k_min, probDens_k_max = 0, 1.25 * probDens_psi_k_frames.max()
    if probDens_k_max == 0:
        probDens_k_max = 1
    ax[2].set(title="k Space", xlabel="$k$",
              ylabel="$|\\tilde{\\Psi}(k,t)|^2$")
    ax[2].axis([k_min, k_max, probDens_k_min, probDens_k_max])
    line2_0 = ax[2].plot(
        k, probDens_psi_k_frames[0], label="$|\\tilde{\\Psi}(k,t)|^2$"
    )[0]
    ax[2].legend(loc="best")

    def nextframe(n_t):
        totalProb_psi_x = computeTotalProbability(
            x, psi_x_frames[skippingFactor * n_t])
        totalEnergy = computeTotalEnergy(
            x,
            psi_x_frames[skippingFactor * n_t],
            V_frames[skippingFactor * n_t],
            kappa,
            m,
        )
        infoText_tl.set_text(
            f"$t$ = {dt*n_t*skippingFactor*unit_t['conversionFactor']:.4G} ${unit_t['symbol']}$\n"
            f"$\\langle\\Psi|\\Psi\\rangle$ = {totalProb_psi_x:>.4G}\n"
            f"$E$ = {totalEnergy*unit_E['conversionFactor']:.7G} ${unit_E['symbol']}$",
        )

        line0_0.set_data(sx, psi_x_frames[skippingFactor * n_t].imag)
        line0_1.set_data(sx, psi_x_frames[skippingFactor * n_t].real)
        line0_2.set_data(sx, np.absolute(psi_x_frames[skippingFactor * n_t]))
        line0_4.set_data(sx, V_frames[skippingFactor * n_t])

        line1_0.set_data(sx, smoothPsi_x_frames[skippingFactor * n_t])
        line1_1.set_data(sx, V_frames[skippingFactor * n_t])

        line2_0.set_data(k, np.absolute(
            psi_k_frames[skippingFactor * n_t]) ** 2)

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
