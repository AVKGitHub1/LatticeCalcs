import numpy as np
import matplotlib.pyplot as plt
import arc
atom = arc.Rubidium85()
pol = arc.DynamicPolarizability(atom, 5, 0, 0.5)
# Constants
eps0 = 8.8541878128e-12       # F/m
c = 299792458.0               # m/s
kB = 1.380649e-23             # J/K
h = 6.62607015e-34            # J*s
amu = 1.66053906660e-27       # kg
m_Rb85 = 84.911789738 * amu   # kg
hbar = h / (2 * np.pi)

pol.defineBasis(5, 15)
wavelengths = np.linspace(770, 800, 1000)
pols = np.zeros_like(wavelengths)
depths = np.zeros_like(wavelengths)
f_axs = np.zeros_like(wavelengths)
f_rads = np.zeros_like(wavelengths)
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


for i, w in enumerate(wavelengths):
    ret = pol.getPolarizability(w*(10**(-9)), units='SI')
    i_pol = float(ret[0]) + 0*(3/6)*float(ret[1])
    pols[i] = i_pol
    lattice_params = lattice_depth_and_freq(
        lam=w*(10**(-9)),
        w0=250e-6, # waist m
        P=0.130,  # W per beam
        alpha_hz=i_pol)
    depths[i] = lattice_params["U0_uK"] 
    f_axs[i] = lattice_params["f_axial_Hz"] * 10**-3  # convert to kHz
    f_rads[i] = lattice_params["f_radial_Hz"] * 10**-3  # convert to kHz

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(wavelengths, pols)
axs[0].set_title("Dynamic Polarizability")
axs[0].set_ylabel("Polarizability (SI units)")
axs[1].plot(wavelengths, depths)
axs[1].set_title("Lattice Depth")
axs[1].set_ylabel("Depth (µK)")
axs[2].plot(wavelengths, f_axs, label="Axial")
axs[2].plot(wavelengths, f_rads, label="Radial")
axs[2].set_title("Trap Frequencies")
axs[2].set_xlabel("Wavelength (nm)")
axs[2].set_ylabel("Frequency (kHz)")
axs[2].legend()
plt.tight_layout()
plt.show()