import numpy as np
import matplotlib.pyplot as plt
import arc
from matplotlib.widgets import Slider

# Visual theme
FIG_BG = "#f5f7fb"
AX_BG = "#ffffff"
GRID_COLOR = "#8a94a6"
DEPTH_COLOR = "#2a6fbb"
AXIAL_COLOR = "#d1495b"
SLIDER_BG = "#e9edf5"
SLIDER_COLOR = "#4a6fa5"

# Physical constants
eps0 = 8.8541878128e-12  # F/m
c = 299792458.0  # m/s
kB = 1.380649e-23  # J/K
h = 6.62607015e-34  # J*s
amu = 1.66053906660e-27  # kg

atom = arc.Rubidium85()
m_Rb85 = 84.911789738 * amu  # kg
pol = arc.DynamicPolarizability(atom, 5, 0, 0.5)

wavelength0 = 594.0e-9  # m
pol.defineBasis(5, 15)
ret = pol.getPolarizability(wavelength0, units="SI")
i_pol = float(ret[0]) + 0 * (3 / 6) * float(ret[1])

waist0 = 250e-6  # m

powers = np.arange(0.1, 1.5, 0.05)
depths = np.zeros_like(powers)
f_axs = np.zeros_like(powers)


def lattice_depth_and_freq(lam, w0, P, alpha_hz, power_is_total=False):
    """
    lam      : wavelength [m]
    w0       : waist [m]
    P        : power [W] (per beam unless power_is_total=True)
    alpha_hz : polarizability [Hz * m^2 / V^2]
    """

    if power_is_total:
        P = P / 2.0  # convert total power -> per-beam power

    k = 2 * np.pi / lam

    # Lattice depth
    U0_Hz = 4 * alpha_hz * P / (np.pi * eps0 * c * w0**2)  # Hz
    U0_J = h * U0_Hz  # J

    # Use absolute depth for trap frequencies
    Uabs = abs(U0_J)

    # Axial lattice frequency
    omega_x = np.sqrt(2 * Uabs * k**2 / m_Rb85)  # rad/s
    f_x = omega_x / (2 * np.pi)  # Hz

    # Radial frequency from Gaussian envelope
    omega_r = np.sqrt(4 * Uabs / (m_Rb85 * w0**2))  # rad/s
    f_r = omega_r / (2 * np.pi)  # Hz

    return {
        "U0_Hz": U0_Hz,
        "U0_J": U0_J,
        "U0_uK": U0_J / kB * 1e6,
        "f_axial_Hz": f_x,
        "f_radial_Hz": f_r,
    }


def compute_traces(wavelength, waist, alpha_hz):
    for i, p in enumerate(powers):
        lattice_params = lattice_depth_and_freq(
            lam=wavelength,
            w0=waist,  # waist [m]
            P=p,  # power [W] per beam
            alpha_hz=alpha_hz,
        )
        depths[i] = lattice_params["U0_uK"]
        f_axs[i] = lattice_params["f_axial_Hz"] * 1e-3  # convert to kHz


def draw_plots():
    axs[0].clear()
    axs[0].set_facecolor(AX_BG)
    axs[0].plot(powers, depths, color=DEPTH_COLOR, linewidth=2.2)
    axs[0].fill_between(powers, depths, color=DEPTH_COLOR, alpha=0.15)
    axs[0].set_title("Lattice Depth", fontsize=13, fontweight="semibold")
    axs[0].set_ylabel("Depth (µK)")
    axs[0].grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    axs[1].clear()
    axs[1].set_facecolor(AX_BG)
    axs[1].plot(powers, f_axs, color=AXIAL_COLOR, linewidth=2.2, label="Axial")
    axs[1].set_title("Trap Frequencies", fontsize=13, fontweight="semibold")
    axs[1].set_xlabel("Power (W)")
    axs[1].set_ylabel("Frequency (kHz)")
    axs[1].grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(frameon=False)


compute_traces(wavelength0, waist0, i_pol)

fig, axs = plt.subplots(2, 1, figsize=(6, 10))
fig.patch.set_facecolor(FIG_BG)
fig.suptitle("Optical Lattice Explorer", fontsize=16, fontweight="bold", y=0.975)
fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.22, hspace=0.35)
draw_plots()

# Create sliders for interactive adjustment
ax_wavelength = plt.axes([0.16, 0.12, 0.72, 0.03], facecolor=SLIDER_BG)
ax_waist = plt.axes([0.16, 0.07, 0.72, 0.03], facecolor=SLIDER_BG)

slider_wavelength = Slider(
    ax_wavelength,
    "Wavelength (nm)",
    500,
    900,
    valinit=594,
    valstep=1,
    color=SLIDER_COLOR,
    initcolor="none",
)
slider_waist = Slider(
    ax_waist,
    "Waist (µm)",
    50,
    1000,
    valinit=250,
    valstep=10,
    color=SLIDER_COLOR,
    initcolor="none",
)

slider_wavelength.label.set_fontsize(10)
slider_waist.label.set_fontsize(10)
slider_wavelength.valtext.set_fontsize(10)
slider_waist.valtext.set_fontsize(10)


def update(val):
    wavelength = slider_wavelength.val * 1e-9
    waist = slider_waist.val * 1e-6
    ret = pol.getPolarizability(wavelength, units="SI")
    i_pol = float(ret[0]) + 0 * (3 / 6) * float(ret[1])

    compute_traces(wavelength, waist, i_pol)
    draw_plots()

    fig.canvas.draw_idle()


slider_wavelength.on_changed(update)
slider_waist.on_changed(update)

plt.show()
