try:
    import marimo
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing notebook dependency. Install with: pip install marimo"
    ) from exc

__generated_with = "0.11.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from hf_pol import HFPolarizabilityCalculator

    return HFPolarizabilityCalculator, mo, np, plt


@app.cell
def __(HFPolarizabilityCalculator, np):
    # Choices
    ATOM = "Rb85"
    n = 5
    L = 0
    J = 0.5
    F = 3
    mF = 3
    q = 0

    # Plot palette
    FIG_BG = "#f6f8fc"
    AX_BG = "#ffffff"
    GRID_COLOR = "#8a94a6"
    DEPTH_COLOR = "#2a6fbb"
    AXIAL_COLOR = "#d1495b"
    SCATTER_COLOR = "#17643d"

    # Physical constants
    eps0 = 8.8541878128e-12  # F/m
    c = 299792458.0  # m/s
    kB = 1.380649e-23  # J/K
    h = 6.62607015e-34  # J*s
    hbar = h / (2 * np.pi)

    class LatticeModel:
        def __init__(self, n=n, L=L, J=J, F=F, mF=mF, q=q):
            self.n = n
            self.L = L
            self.J = J
            self.F = F
            self.mF = mF
            self.q = q
            self.hfpol = HFPolarizabilityCalculator(
                atom_name=ATOM,
                n=self.n,
                L=self.L,
                J=self.J,
                F=self.F,
                mF=self.mF,
                q=self.q,
            )
            self.m_atom = self.hfpol.get_atom_mass()

        def get_polarizability(self, wavelength_m):
            return self.hfpol.calculate(wavelength_m)

        def lattice_depth_and_freq(
            self, lam, w0, power, alpha_hz, power_is_total=False
        ):
            if power_is_total:
                power = power / 2.0

            k = 2 * np.pi / lam
            u0_hz = 4 * alpha_hz * power / (np.pi * eps0 * c * w0**2)
            u0_j = h * u0_hz
            u_abs = abs(u0_j)

            omega_x = np.sqrt(2 * u_abs * k**2 / self.m_atom)
            omega_r = np.sqrt(4 * u_abs / (self.m_atom * w0**2))

            return {
                "U0_uK": u0_j / kB * 1e6,
                "f_axial_Hz": omega_x / (2 * np.pi),
                "f_radial_Hz": omega_r / (2 * np.pi),
            }

        def estimate_scattering_rate(
            self, lam, w0, power, alpha_hz, power_is_total=False
        ):
            if power_is_total:
                power = power / 2.0

            # Peak standing-wave intensity for two equal counter-propagating Gaussian beams.
            intensity_peak = 8 * power / (np.pi * w0**2)

            # ARC "SI" polarizability used here is in Hz/(V/m)^2; convert to SI dipole units.
            alpha_si = h * alpha_hz
            k = 2 * np.pi / lam
            omega = 2 * np.pi * c / lam

            # Rayleigh-scattering estimate from dipole cross-section:
            # sigma = k^4 |alpha|^2 / (6*pi*eps0^2), Gamma = sigma*I/(hbar*omega)
            sigma = (k**4) * (abs(alpha_si) ** 2) / (6 * np.pi * eps0**2)
            return sigma * intensity_peak / (hbar * omega)

        def compute_traces(self, wavelength_m, waist_m, alpha_hz, powers):
            depths = np.zeros_like(powers)
            f_axs = np.zeros_like(powers)
            scatters_angular = np.zeros_like(powers)
            for i, power in enumerate(powers):
                result = self.lattice_depth_and_freq(
                    wavelength_m, waist_m, power, alpha_hz
                )
                depths[i] = result["U0_uK"]
                f_axs[i] = result["f_axial_Hz"] * 1e-3  # kHz
                scatter_hz = self.estimate_scattering_rate(
                    wavelength_m, waist_m, power, alpha_hz
                )
                scatters_angular[i] = 2 * np.pi * scatter_hz  # rad/s
            return depths, f_axs, scatters_angular

    model = LatticeModel()

    return (
        AXIAL_COLOR,
        AX_BG,
        DEPTH_COLOR,
        FIG_BG,
        GRID_COLOR,
        LatticeModel,
        SCATTER_COLOR,
        model,
    )


@app.cell
def __(mo):
    mo.md("""# Optical Lattice Explorer (Marimo)""")
    return


@app.cell
def __(mo):
    wavelength_nm = mo.ui.number(
        start=300.0,
        stop=2000.0,
        step=0.01,
        value=594.0,
        label="Wavelength (nm)",
    )
    waist_um = mo.ui.number(
        start=50.0,
        stop=1000.0,
        step=10.0,
        value=250.0,
        label="Waist (um)",
    )
    min_power_w = mo.ui.number(
        start=0.01,
        stop=50.0,
        step=0.05,
        value=0.05,
        label="Min Power (W)",
    )
    max_power_w = mo.ui.number(
        start=0.02,
        stop=50.0,
        step=0.05,
        value=3.0,
        label="Max Power (W)",
    )

    return max_power_w, min_power_w, waist_um, wavelength_nm


@app.cell
def __(max_power_w, min_power_w, mo, waist_um, wavelength_nm):
    controls = mo.vstack(
        [
            wavelength_nm,
            waist_um,
            min_power_w,
            max_power_w,
        ],
        gap=1,
    )
    controls
    return


@app.cell
def __(max_power_w, min_power_w, model, np, waist_um, wavelength_nm):
    wavelength_m = wavelength_nm.value * 1e-9
    waist_m = waist_um.value * 1e-6

    power_step = 0.05
    min_power = float(min_power_w.value)
    max_power = float(max_power_w.value)
    if min_power >= max_power:
        max_power = min_power + power_step

    powers = np.arange(min_power, max_power + 0.5 * power_step, power_step)
    powers = np.round(powers, 6)

    alpha_hz = model.get_polarizability(wavelength_m)
    depths, f_axs, scatters = model.compute_traces(
        wavelength_m, waist_m, alpha_hz, powers
    )

    return alpha_hz, depths, f_axs, max_power, min_power, powers, scatters


@app.cell
def __(alpha_hz, max_power, min_power, mo):
    mo.md(
        f"""
        **Polarizability (SI):** `{alpha_hz:.3e}`

        **Power Sweep Used:** `{min_power:.2f} W` to `{max_power:.2f} W` (step `0.05 W`)
        """
    )
    return


@app.cell
def __(
    AXIAL_COLOR,
    AX_BG,
    DEPTH_COLOR,
    FIG_BG,
    GRID_COLOR,
    SCATTER_COLOR,
    depths,
    f_axs,
    plt,
    powers,
    scatters,
):
    fig, (ax_depth, ax_freq, ax_scatter) = plt.subplots(
        3, 1, figsize=(7.5, 9.6), facecolor=FIG_BG
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.07, hspace=0.42)

    ax_depth.clear()
    ax_depth.set_facecolor(AX_BG)
    ax_depth.plot(powers, depths, color=DEPTH_COLOR, linewidth=2.4)
    ax_depth.fill_between(powers, depths, color=DEPTH_COLOR, alpha=0.15)
    ax_depth.set_title("Lattice Depth", fontsize=13, fontweight="semibold")
    ax_depth.set_ylabel("Depth (uK)")
    ax_depth.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
    ax_depth.spines["top"].set_visible(False)
    ax_depth.spines["right"].set_visible(False)

    ax_freq.clear()
    ax_freq.set_facecolor(AX_BG)
    ax_freq.plot(powers, f_axs, color=AXIAL_COLOR, linewidth=2.4, label="Axial")
    ax_freq.set_title("Trap Frequencies", fontsize=13, fontweight="semibold")
    ax_freq.set_ylabel("Frequency (kHz)")
    ax_freq.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
    ax_freq.spines["top"].set_visible(False)
    ax_freq.spines["right"].set_visible(False)
    ax_freq.legend(frameon=False)

    ax_scatter.clear()
    ax_scatter.set_facecolor(AX_BG)
    ax_scatter.plot(powers, scatters, color=SCATTER_COLOR, linewidth=2.4)
    ax_scatter.fill_between(powers, scatters, color=SCATTER_COLOR, alpha=0.12)
    ax_scatter.set_title(
        "Photon Scattering Rate (Estimated, Angular)",
        fontsize=13,
        fontweight="semibold",
    )
    ax_scatter.set_xlabel("Power (W)")
    ax_scatter.set_ylabel("Rate (rad/s) = 2pi x Hz")
    ax_scatter.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    fig
    return fig,


if __name__ == "__main__":
    app.run()
