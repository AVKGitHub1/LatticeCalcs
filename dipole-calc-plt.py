import sys

import arc
import numpy as np
from matplotlib.figure import Figure

from hf_pol import HFPolarizabilityCalculator

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QPainter
    from PyQt6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QSizePolicy,
        QStyle,
        QStyleOptionSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing GUI dependency. Install with: pip install PyQt6"
    ) from exc

# Choices
ATOM = "Rb85"
F = 3
mF = 3
q = 0


def return_atom():
    if ATOM == "Rb85":
        return arc.Rubidium85(), 84.911789738 * amu
    if ATOM == "Rb87":
        return arc.Rubidium87(), 86.909180527 * amu
    print(f"Unknown atom choice: {ATOM}. Defaulting to Rb85.")
    return arc.Rubidium85(), 84.911789738 * amu


def check_validity():
    if F < 0 or mF < -F or mF > F:
        raise ValueError(f"Invalid quantum numbers: F={F}, mF={mF}. Must satisfy -F <= mF <= F.")
    if q not in [-1, 0, 1]:
        raise ValueError(f"Invalid polarization q={q}. Must be -1, 0, or 1.")
    if ATOM == "Rb85" and F not in [2, 3]:
        raise ValueError(f"Invalid F={F} for Rb85. Allowed values are 2 or 3.")
    if ATOM == "Rb87" and F not in [1, 2]:
        raise ValueError(f"Invalid F={F} for Rb87. Allowed values are 1 or 2.")


# Plot palette
FIG_BG = "#f6f8fc"
AX_BG = "#ffffff"
GRID_COLOR = "#8a94a6"
DEPTH_COLOR = "#2a6fbb"
RADIAL_COLOR = "#17643d"
AXIAL_COLOR = "#d1495b"

# Physical constants
eps0 = 8.8541878128e-12  # F/m
c = 299792458.0  # m/s
kB = 1.380649e-23  # J/K
h = 6.62607015e-34  # J*s
amu = 1.66053906660e-27  # kg


class DipoleModel:
    def __init__(self):
        self.atom, self.m_atom = return_atom()
        check_validity()
        self.powers = np.arange(0.2, 1.5, 0.05)

    def get_polarizability(self, wavelength_m):
        hfpol = HFPolarizabilityCalculator(
            atom_name=ATOM,
            n=5,
            L=0,
            J=0.5,
            F=F,
            mF=mF,
            q=q)
        return hfpol.calculate(wavelength_m)

    def dipole_depth_and_freq(self, lam, w0, power, alpha_hz):
        # Gaussian beam trap depth at focus (single beam).
        u0_hz = alpha_hz * power / (np.pi * eps0 * c * w0**2)
        u0_j = h * u0_hz
        u_abs = abs(u0_j)

        z_r = np.pi * w0**2 / lam

        # Harmonic approximation around beam focus.
        omega_r = np.sqrt(4 * u_abs / (self.m_atom * w0**2))
        omega_z = np.sqrt(2 * u_abs / (self.m_atom * z_r**2))

        return {
            "U0_uK": u0_j / kB * 1e6,
            "f_radial_Hz": omega_r / (2 * np.pi),
            "f_axial_Hz": omega_z / (2 * np.pi),
            "zR_mm": z_r * 1e3,
        }

    def compute_traces(self, wavelength_m, waist_m, alpha_hz):
        depths = np.zeros_like(self.powers)
        f_rads = np.zeros_like(self.powers)
        f_axs = np.zeros_like(self.powers)
        for i, power in enumerate(self.powers):
            result = self.dipole_depth_and_freq(wavelength_m, waist_m, power, alpha_hz)
            depths[i] = result["U0_uK"]
            f_rads[i] = result["f_radial_Hz"] * 1e-3  # kHz
            f_axs[i] = result["f_axial_Hz"] * 1e-3  # kHz
        return depths, f_rads, f_axs


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        figure = Figure(figsize=(7.5, 8.5), facecolor=FIG_BG)
        super().__init__(figure)
        self.setParent(parent)
        self.ax_depth, self.ax_freq = self.figure.subplots(2, 1)
        self.figure.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.08, hspace=0.35)

    def draw_traces(self, powers, depths, f_rads, f_axs):
        self.ax_depth.clear()
        self.ax_depth.set_facecolor(AX_BG)
        self.ax_depth.plot(powers, depths, color=DEPTH_COLOR, linewidth=2.4)
        self.ax_depth.fill_between(powers, depths, color=DEPTH_COLOR, alpha=0.15)
        self.ax_depth.set_title("Dipole Trap Depth", fontsize=13, fontweight="semibold")
        self.ax_depth.set_ylabel("Depth (uK)")
        self.ax_depth.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
        self.ax_depth.spines["top"].set_visible(False)
        self.ax_depth.spines["right"].set_visible(False)

        self.ax_freq.clear()
        self.ax_freq.set_facecolor(AX_BG)
        self.ax_freq.plot(powers, f_rads, color=RADIAL_COLOR, linewidth=2.4, label="Radial")
        self.ax_freq.plot(powers, f_axs, color=AXIAL_COLOR, linewidth=2.4, label="Axial")
        self.ax_freq.set_title("Trap Frequencies", fontsize=13, fontweight="semibold")
        self.ax_freq.set_xlabel("Power (W)")
        self.ax_freq.set_ylabel("Frequency (kHz)")
        self.ax_freq.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
        self.ax_freq.spines["top"].set_visible(False)
        self.ax_freq.spines["right"].set_visible(False)
        self.ax_freq.legend(frameon=False)

        self.draw_idle()


class ArrowSpinBox(QDoubleSpinBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.buttonSymbols() == QDoubleSpinBox.ButtonSymbols.NoButtons:
            return

        option = QStyleOptionSpinBox()
        self.initStyleOption(option)

        up_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxUp,
            self,
        )
        down_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxDown,
            self,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(QColor("#31446e"))
        painter.drawText(up_rect, int(Qt.AlignmentFlag.AlignCenter), "^")
        painter.drawText(down_rect, int(Qt.AlignmentFlag.AlignCenter), "v")
        painter.end()


class DipoleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = DipoleModel()
        self.setWindowTitle("Optical Dipole Trap Explorer")
        self.resize(1180, 820)
        self.setMinimumSize(980, 700)
        self._apply_style()
        self._build_ui()
        self.refresh_plots()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eef2fb, stop:1 #e9edf8
                );
            }
            QFrame#PanelCard {
                background: #ffffff;
                border: 1px solid #dce3f3;
                border-radius: 14px;
            }
            QLabel#AppTitle {
                color: #1f2a44;
                font-size: 26px;
                font-weight: 700;
            }
            QLabel#SectionTitle {
                color: #2a3658;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#MetaLabel {
                color: #5b6b87;
                font-size: 12px;
                font-weight: 500;
            }
            QLabel#MetaValue {
                color: #223252;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#ControlLabel {
                color: #2a3658;
                font-size: 13px;
                font-weight: 600;
            }
            QDoubleSpinBox {
                background: #f5f7fc;
                border: 1px solid #cfd8ec;
                border-radius: 10px;
                padding: 6px 8px;
                min-height: 34px;
                font-size: 14px;
                color: #1f2a44;
            }
            QDoubleSpinBox:focus {
                border: 1px solid #6a8fd8;
                background: #ffffff;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 24px;
                border: none;
                background: #e7edf9;
                margin: 1px;
                border-radius: 6px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #d7e2f6;
            }
            """
        )

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        left_panel = QFrame()
        left_panel.setObjectName("PanelCard")
        left_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(12)

        title = QLabel("Optical Dipole Trap Explorer")
        title.setObjectName("AppTitle")
        title.setWordWrap(True)
        left_layout.addWidget(title)

        controls_title = QLabel("Controls")
        controls_title.setObjectName("SectionTitle")
        left_layout.addWidget(controls_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        wavelength_label = QLabel("Wavelength (nm)")
        wavelength_label.setObjectName("ControlLabel")
        wavelength_label.setMinimumWidth(130)
        self.wavelength_spin = ArrowSpinBox()
        self.wavelength_spin.setRange(300.0, 2000.0)
        self.wavelength_spin.setSingleStep(1.0)
        self.wavelength_spin.setDecimals(0)
        self.wavelength_spin.setValue(1064.0)

        waist_label = QLabel("Waist (um)")
        waist_label.setObjectName("ControlLabel")
        waist_label.setMinimumWidth(130)
        self.waist_spin = ArrowSpinBox()
        self.waist_spin.setRange(10.0, 1000.0)
        self.waist_spin.setSingleStep(5.0)
        self.waist_spin.setDecimals(0)
        self.waist_spin.setValue(50.0)

        grid.addWidget(wavelength_label, 0, 0)
        grid.addWidget(self.wavelength_spin, 0, 1)
        grid.addWidget(waist_label, 1, 0)
        grid.addWidget(self.waist_spin, 1, 1)
        left_layout.addLayout(grid)

        self.alpha_label = QLabel("...")
        self.alpha_label.setObjectName("MetaValue")
        self.zr_label = QLabel("...")
        self.zr_label.setObjectName("MetaValue")

        left_layout.addWidget(self._meta_row("Polarizability (SI)", self.alpha_label))
        left_layout.addWidget(self._meta_row("Rayleigh Range (mm)", self.zr_label))
        left_layout.addStretch(1)

        plot_card = QFrame()
        plot_card.setObjectName("PanelCard")
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(12, 12, 12, 12)
        plot_layout.setSpacing(8)
        self.canvas = PlotCanvas(plot_card)
        plot_layout.addWidget(self.canvas)

        root_layout.addWidget(left_panel, stretch=0)
        root_layout.addWidget(plot_card, stretch=1)

        self.wavelength_spin.valueChanged.connect(self.refresh_plots)
        self.waist_spin.valueChanged.connect(self.refresh_plots)

    def _meta_row(self, label_text, value_widget):
        row = QFrame()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        label = QLabel(label_text)
        label.setObjectName("MetaLabel")
        row_layout.addWidget(label)
        row_layout.addWidget(value_widget)
        return row

    def refresh_plots(self):
        wavelength_m = self.wavelength_spin.value() * 1e-9
        waist_m = self.waist_spin.value() * 1e-6

        alpha_hz = self.model.get_polarizability(wavelength_m)
        depths, f_rads, f_axs = self.model.compute_traces(wavelength_m, waist_m, alpha_hz)
        self.canvas.draw_traces(self.model.powers, depths, f_rads, f_axs)

        params = self.model.dipole_depth_and_freq(
            wavelength_m,
            waist_m,
            self.model.powers[-1],
            alpha_hz,
        )
        self.alpha_label.setText(f"{alpha_hz:.3e}")
        self.zr_label.setText(f"{params['zR_mm']:.3f}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DipoleWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
