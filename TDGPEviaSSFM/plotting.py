"""
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from .tools import probDensity
from .tools import computeTotalProbability
from .tools import computeTotalEnergy


def plotState(x, k, psi_x, psi_k, V, kappa, m, title=None, furtherInfo={}):
    from .configs import dt
    from .configs import N_t
    from .configs import unitSystem
    from .configs import smoothingParameter

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

    smoothPsi_x = scipy.signal.savgol_filter(
        np.absolute(psi_x), smoothingParameter*2+1, 3)
    probDens_psi_k = probDensity(psi_k)
    totalProb_psi_x = computeTotalProbability(x, psi_x)
    totalEnergy = computeTotalEnergy(x, psi_x, V, kappa, m)

    # Plot Creation and Configuration #

    fig, ax = plt.subplots(3)
    ax_0_1 = ax[0].twinx()
    ax_1_1 = ax[1].twinx()

    # Injection of Plot Information #

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("$\\Psi(x,t)$ and $\\tilde{\\Psi}(k,t)$", fontsize=16)

    fig.text(
        0.01,
        0.925,
        f"$\\langle\\Psi|\\Psi\\rangle$ = {totalProb_psi_x:>.4G}\n"
        f"$E$ = {totalEnergy*unit_E['conversionFactor']:.4G} ${unit_E['symbol']}$",
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
    psi_x_min, psi_x_max = -2 * \
        np.absolute(psi_x).max(), 2 * np.absolute(psi_x).max()
    if psi_x_min == 0 and psi_x_max == 0:
        psi_x_min, psi_x_max = -1, 1
    ax[0].set(title="Position Space", xlabel="$x$", ylabel="$\\Psi(x,t)$")
    ax[0].axis([sx_min, sx_max, psi_x_min, psi_x_max])
    ax[0].plot(sx, psi_x.imag, label="$im(\\Psi(x,t))$", color="blue")
    ax[0].plot(sx, psi_x.real, label="$re(\\Psi(x,t))$", color="goldenrod")
    ax[0].plot(sx, np.absolute(psi_x), label="$|\\Psi(x,t)|$", color="green")
    ax[0].legend(loc="upper left")

    V_min, V_max = -(1 / 0.95) * np.absolute(V).max(), (1 /
                                                        0.95) * np.absolute(V).max()
    if V_min == 0 and V_max == 0:
        V_min, V_max = -1, 1
    ax_0_1.set_ylabel("$V(x,t)$")
    ax_0_1.axis([sx_min, sx_max, V_min, V_max])
    ax_0_1.plot(sx, V, label="$V(x,t)$", color="firebrick")
    ax_0_1.legend(loc="upper right")

    # Second Subplot

    ax[1].set(title="Position Space", xlabel="$x$", ylabel="$\\Psi(x,t)$")
    ax[1].axis([sx_min, sx_max, psi_x_min, psi_x_max])
    ax[1].plot(sx, smoothPsi_x, label="Smooth $|\\Psi(x,t)|$",
               color="green")
    ax[1].legend(loc="upper left")

    ax_1_1.set_ylabel("$V(x,t)$")
    ax_1_1.axis([sx_min, sx_max, V_min, V_max])
    ax_1_1.plot(sx, V, label="$V(x,t)$", color="firebrick")
    ax_1_1.legend(loc="upper right")

    # Thrid Subplot

    k_min, k_max = 1.1 * k.min(), 1.1 * k.max()
    ax[2].set(title="k Space", xlabel="$k$",
              ylabel="$|\\tilde{\\Psi}(k,t)|^2$")
    probDens_k_min, probDens_k_max = 0, 1.25 * probDens_psi_k.max()
    if probDens_k_max == 0:
        probDens_k_max = 1
    ax[2].axis([k_min, k_max, probDens_k_min, probDens_k_max])
    ax[2].plot(k, probDens_psi_k, label="$|\\tilde{\\Psi}(k,t)|^2$")
    ax[2].legend(loc="best")

    plt.show()
