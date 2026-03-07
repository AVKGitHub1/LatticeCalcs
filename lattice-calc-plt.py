import numpy as np
import matplotlib.pyplot as plt
import arc
from matplotlib.widgets import Button, TextBox

# Visual theme
FIG_BG = "#f5f7fb"
AX_BG = "#ffffff"
GRID_COLOR = "#8a94a6"
DEPTH_COLOR = "#2a6fbb"
AXIAL_COLOR = "#d1495b"
SLIDER_BG = "#e9edf5"
BUTTON_COLOR = "#d7dfed"
BUTTON_HOVER = "#c7d3e8"
TEXTBOX_FG = "#1f2a44"

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

controls = {
    "wavelength_nm": {
        "label": "Wavelength (nm)",
        "min": 300.0,
        "max": 2000.0,
        "step": 1.0,
        "value": wavelength0 * 1e9,
        "y": 0.12,
    },
    "waist_um": {
        "label": "Waist (µm)",
        "min": 50.0,
        "max": 1000.0,
        "step": 10.0,
        "value": waist0 * 1e6,
        "y": 0.07,
    },
}

text_sync_in_progress = False


def format_control_value(value, step):
    if abs(step - round(step)) < 1e-12:
        return f"{int(round(value))}"
    decimals = max(0, int(np.ceil(-np.log10(step))))
    return f"{value:.{decimals}f}"


def snap_control_value(value, cfg):
    snapped = round((value - cfg["min"]) / cfg["step"]) * cfg["step"] + cfg["min"]
    return min(cfg["max"], max(cfg["min"], snapped))


def update_from_controls():
    wavelength = controls["wavelength_nm"]["value"] * 1e-9
    waist = controls["waist_um"]["value"] * 1e-6
    ret = pol.getPolarizability(wavelength, units="SI")
    i_pol = float(ret[0]) + 0 * (3 / 6) * float(ret[1])

    compute_traces(wavelength, waist, i_pol)
    draw_plots()
    fig.canvas.draw_idle()

def set_control_value(name, raw_value):
    global text_sync_in_progress
    cfg = controls[name]
    cfg["value"] = snap_control_value(raw_value, cfg)
    text_sync_in_progress = True
    cfg["textbox"].set_val(format_control_value(cfg["value"], cfg["step"]))
    text_sync_in_progress = False
    update_from_controls()


def make_text_submit_handler(name):
    def _handler(text):
        if text_sync_in_progress:
            return
        cfg = controls[name]
        try:
            raw_value = float(text)
        except ValueError:
            raw_value = cfg["value"]
        set_control_value(name, raw_value)

    return _handler


def make_step_handler(name, direction):
    def _handler(_event):
        cfg = controls[name]
        set_control_value(name, cfg["value"] + direction * cfg["step"])

    return _handler


for name, cfg in controls.items():
    fig.text(
        0.16,
        cfg["y"] + 0.02,
        cfg["label"],
        fontsize=10,
        color=TEXTBOX_FG,
        ha="left",
        va="center",
    )

    ax_box = plt.axes([0.56, cfg["y"], 0.15, 0.04], facecolor=SLIDER_BG)
    ax_down = plt.axes([0.73, cfg["y"], 0.06, 0.04], facecolor=SLIDER_BG)
    ax_up = plt.axes([0.81, cfg["y"], 0.06, 0.04], facecolor=SLIDER_BG)

    textbox = TextBox(
        ax_box,
        "",
        initial=format_control_value(cfg["value"], cfg["step"]),
        textalignment="center",
    )
    textbox.label.set_visible(False)
    textbox.text_disp.set_fontsize(10)
    textbox.text_disp.set_color(TEXTBOX_FG)
    textbox.on_submit(make_text_submit_handler(name))

    btn_down = Button(ax_down, "▼", color=BUTTON_COLOR, hovercolor=BUTTON_HOVER)
    btn_up = Button(ax_up, "▲", color=BUTTON_COLOR, hovercolor=BUTTON_HOVER)
    btn_down.label.set_fontsize(9)
    btn_up.label.set_fontsize(9)
    btn_down.on_clicked(make_step_handler(name, -1))
    btn_up.on_clicked(make_step_handler(name, 1))

    cfg["textbox"] = textbox
    cfg["btn_down"] = btn_down
    cfg["btn_up"] = btn_up

plt.show()
