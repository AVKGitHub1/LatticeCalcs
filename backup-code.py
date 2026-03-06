import numpy as np
import matplotlib.pyplot as plt
import arc
from matplotlib.widgets import Slider

# Constants
eps0 = 8.8541878128e-12       # F/m
c = 299792458.0               # m/s
kB = 1.380649e-23             # J/K
h = 6.62607015e-34            # J*s
amu = 1.66053906660e-27       # kg
hbar = h / (2 * np.pi)

atom = arc.Rubidium85()
m_Rb85 = 84.911789738 * amu   # kg
pol = arc.DynamicPolarizability(atom, 5, 0, 0.5)

Wavelength = 594.0 * 1e-9  # m
pol.defineBasis(5, 15)
ret = pol.getPolarizability(Wavelength, units='SI')
i_pol = float(ret[0]) + 0*(3/6)*float(ret[1])

Waist = 250e-6  # m

powers = np.arange(0.1, 1.5, 0.05)
depths = np.zeros_like(powers)
f_axs = np.zeros_like(powers)
f_rads = np.zeros_like(powers)
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
    U0_Hz = 4 * alpha_hz * P / (np.pi * eps0 * c * w0**2)   # Hz
    U0_J = h * U0_Hz                                         # J

    # Use absolute depth for trap frequencies
    Uabs = abs(U0_J)

    # Axial lattice frequency
    omega_x = np.sqrt(2 * Uabs * k**2 / m_Rb85)              # rad/s
    f_x = omega_x / (2 * np.pi)                              # Hz

    # Radial frequency from Gaussian envelope
    omega_r = np.sqrt(4 * Uabs / (m_Rb85 * w0**2))           # rad/s
    f_r = omega_r / (2 * np.pi)                              # Hz

    return {
        "U0_Hz": U0_Hz,
        "U0_J": U0_J,
        "U0_uK": U0_J / kB * 1e6,
        "f_axial_Hz": f_x,
        "f_radial_Hz": f_r,
    }


for i, p in enumerate(powers):
    lattice_params = lattice_depth_and_freq(
        lam=Wavelength,
        w0=Waist, # waist m
        P=p,  # W per beam
        alpha_hz=i_pol)
    depths[i] = lattice_params["U0_uK"] 
    f_axs[i] = lattice_params["f_axial_Hz"] * 10**-3  # convert to kHz
    # f_rads[i] = lattice_params["f_radial_Hz"] * 10**-3  # convert to kHz

fig, axs = plt.subplots(2, 1, figsize=(6, 10))
axs[0].plot(powers, depths)
axs[0].set_title("Lattice Depth")
axs[0].set_ylabel("Depth (µK)")
axs[1].plot(powers, f_axs, label="Axial")
axs[1].set_title("Trap Frequencies")
axs[1].set_xlabel("Power (W)")
axs[1].set_ylabel("Frequency (kHz)")
axs[1].legend()

# Create sliders for interactive adjustment
ax_wavelength = plt.axes([0.2, 0.35, 0.6, 0.03])
ax_waist = plt.axes([0.2, 0.30, 0.6, 0.03])

slider_wavelength = Slider(ax_wavelength, 'Wavelength (nm)', 500, 900, valinit=594, valstep=1)
slider_waist = Slider(ax_waist, 'Waist (µm)', 50, 1000, valinit=250, valstep=10)

def update(val):
    wavelength = slider_wavelength.val * 1e-9
    waist = slider_waist.val * 1e-6
    ret = pol.getPolarizability(wavelength, units='SI')
    i_pol = float(ret[0]) + 0*(3/6)*float(ret[1])
    for i, p in enumerate(powers):
        lattice_params = lattice_depth_and_freq(wavelength, waist, p, i_pol)
        depths[i] = lattice_params["U0_uK"]
        f_axs[i] = lattice_params["f_axial_Hz"] * 1e-3
    
    axs[0].clear()
    axs[0].plot(powers, depths)
    axs[0].set_title("Lattice Depth")
    axs[0].set_ylabel("Depth (µK)")
    
    axs[1].clear()
    axs[1].plot(powers, f_axs, label="Axial")
    axs[1].set_title("Trap Frequencies")
    axs[1].set_xlabel("Power (W)")
    axs[1].set_ylabel("Frequency (kHz)")
    axs[1].legend()
    
    fig.canvas.draw_idle()

slider_wavelength.on_changed(update)
slider_waist.on_changed(update)


plt.tight_layout()
plt.show()