from __future__ import annotations

import csv
import inspect
import math
import sys
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

# ARC package: https://arc-alkali-rydberg-calculator.readthedocs.io/
try:
    from arc import DynamicPolarizability, Rubidium85
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: arc. Install it with:\n"
        "  pip install arc-alkali-rydberg-calculator"
    ) from exc

try:
    from PyQt5.QtCore import QRegularExpression
    from PyQt5.QtGui import QDoubleValidator, QIntValidator, QRegularExpressionValidator
    from PyQt5.QtWidgets import (
        QApplication,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: PyQt5. Install it with:\n"
        "  pip install PyQt5"
    ) from exc

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install it with:\n"
        "  pip install matplotlib"
    ) from exc


# Physical constants (SI)
EPS0 = 8.854_187_812_8e-12
C0 = 299_792_458.0
A0 = 5.291_772_109_03e-11
H = 6.626_070_15e-34
HBAR = 1.054_571_817e-34
KB = 1.380_649e-23
AMU = 1.660_539_066_60e-27
MASS_RB85 = 84.911_789_738 * AMU


@dataclass(frozen=True)
class HyperfineState:
    F: int
    mF: int


@dataclass(frozen=True)
class LatticeResult:
    wavelength_nm: float
    alpha_au: float
    depth_joule: float
    depth_uK: float
    depth_Er: float
    axial_freq_hz: float
    radial_freq_hz: float
    intensity_single_beam_w_m2: float
    intensity_standing_peak_w_m2: float


class Rb85GroundStatePolarizability:
    def __init__(self) -> None:
        self.atom = Rubidium85()
        self.n = 5
        self.l = 0
        self.j = 0.5
        self._direct_get_polarizability = getattr(self.atom, "getPolarizability", None)
        self._dynamic_polarizability = None

    @staticmethod
    def validate_state(F: int, mF: int) -> None:
        if F not in (2, 3):
            raise ValueError("For Rb-85 5S1/2, F must be 2 or 3.")
        if abs(mF) > F:
            raise ValueError(f"Invalid mF={mF} for F={F}; requires |mF| <= F.")

    def alpha_with_hyperfine(
        self, wavelength_nm: float, F: int, mF: int, polarization_q: int = 0
    ) -> float:
        self.validate_state(F=F, mF=mF)
        if polarization_q not in (-1, 0, 1):
            raise ValueError("polarization_q must be -1, 0, or 1.")

        wavelength_m = 1e-9 * wavelength_nm
        if callable(self._direct_get_polarizability):
            raw = self._call_get_polarizability_direct(
                wavelength_m=wavelength_m, F=F, mF=mF, q=polarization_q
            )
            return self._extract_alpha(raw)

        alpha0, alpha1, _alpha2 = self._get_alpha_components_dynamic(wavelength_m)
        vector_factor = polarization_q * (mF / (2.0 * F))
        return float(alpha0 + vector_factor * alpha1)

    def _call_get_polarizability_direct(
        self, wavelength_m: float, F: int, mF: int, q: int
    ):
        fn = self._direct_get_polarizability
        if not callable(fn):
            raise RuntimeError("Direct ARC getPolarizability API is not available.")

        params = set(inspect.signature(fn).parameters.keys())
        kwargs = {}
        if "F" in params:
            kwargs["F"] = F
        if "mF" in params:
            kwargs["mF"] = mF
        if "q" in params:
            kwargs["q"] = q

        try:
            return fn(self.n, self.l, self.j, wavelength_m, **kwargs)
        except TypeError:
            kwargs.pop("q", None)
            return fn(self.n, self.l, self.j, wavelength_m, **kwargs)

    def _ensure_dynamic_backend(self) -> None:
        if self._dynamic_polarizability is not None:
            return
        self._dynamic_polarizability = DynamicPolarizability(
            self.atom, self.n, self.l, self.j
        )
        self._dynamic_polarizability.defineBasis(5, 25)

    def _get_alpha_components_dynamic(
        self, wavelength_m: float
    ) -> tuple[float, float, float]:
        self._ensure_dynamic_backend()
        raw = self._dynamic_polarizability.getPolarizability(wavelength_m, units="au")
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            return float(raw[0]), float(raw[1]), float(raw[2])
        raise TypeError(
            "DynamicPolarizability.getPolarizability returned unexpected type/shape."
        )

    @staticmethod
    def _extract_alpha(raw) -> float:
        if np.isscalar(raw):
            return float(raw)
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            return float(raw[0])
        raise TypeError(f"Unexpected polarizability return type: {type(raw)!r}")


class LatticeStrengthCalculator:
    """Computes standing-wave lattice depth and trap frequencies."""

    def __init__(self) -> None:
        self.pol = Rb85GroundStatePolarizability()

    @staticmethod
    def _alpha_au_to_si(alpha_au: float) -> float:
        return 4.0 * math.pi * EPS0 * (A0**3) * alpha_au

    @staticmethod
    def _peak_intensity_single_beam(power_w: float, waist_m: float) -> float:
        # Gaussian beam peak intensity at focus.
        return 2.0 * power_w / (math.pi * waist_m**2)

    def evaluate(
        self, wavelength_nm: float, F: int, mF: int, q: int, power_mw: float, waist_um: float
    ) -> LatticeResult:
        if wavelength_nm <= 0:
            raise ValueError("Wavelength must be positive.")
        if power_mw <= 0:
            raise ValueError("Power per beam must be positive.")
        if waist_um <= 0:
            raise ValueError("Spot size must be positive.")

        wavelength_m = 1e-9 * wavelength_nm
        power_w = 1e-3 * power_mw
        waist_m = 1e-6 * waist_um
        k = 2.0 * math.pi / wavelength_m

        alpha_au = self.pol.alpha_with_hyperfine(
            wavelength_nm=wavelength_nm,
            F=F,
            mF=mF,
            polarization_q=q,
        )
        alpha_si = self._alpha_au_to_si(alpha_au)

        # Counter-propagating equal-power beams -> standing wave with I_max = 4*I0.
        intensity_single = self._peak_intensity_single_beam(power_w, waist_m)
        intensity_standing_peak = 4.0 * intensity_single

        # Dipole potential U = -alpha * I / (2*eps0*c). Lattice depth is |U_peak - U_node|.
        depth_joule = abs(alpha_si * intensity_standing_peak / (2.0 * EPS0 * C0))
        depth_uK = 1e6 * depth_joule / KB

        recoil_energy = (HBAR * k) ** 2 / (2.0 * MASS_RB85)
        depth_Er = depth_joule / recoil_energy if recoil_energy > 0 else float("nan")

        # Axial trap frequency around a lattice minimum: omega = sqrt(2*U0*k^2/m).
        axial_omega = math.sqrt(max(0.0, 2.0 * depth_joule * (k**2) / MASS_RB85))
        axial_freq_hz = axial_omega / (2.0 * math.pi)

        # Radial confinement from Gaussian envelope (red detuning approximation near antinode).
        radial_omega = math.sqrt(max(0.0, 4.0 * depth_joule / (MASS_RB85 * waist_m**2)))
        radial_freq_hz = radial_omega / (2.0 * math.pi)

        return LatticeResult(
            wavelength_nm=wavelength_nm,
            alpha_au=alpha_au,
            depth_joule=depth_joule,
            depth_uK=depth_uK,
            depth_Er=depth_Er,
            axial_freq_hz=axial_freq_hz,
            radial_freq_hz=radial_freq_hz,
            intensity_single_beam_w_m2=intensity_single,
            intensity_standing_peak_w_m2=intensity_standing_peak,
        )


def build_wavelength_array(start_nm: float, stop_nm: float, num_points: int) -> np.ndarray:
    if start_nm <= 0 or stop_nm <= 0:
        raise ValueError("Wavelength bounds must be positive.")
    if stop_nm <= start_nm:
        raise ValueError("scan max must be greater than scan min.")
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")
    return np.linspace(start_nm, stop_nm, num_points)


def write_csv(path: str, results: Sequence[LatticeResult]) -> None:
    with open(path, "w", newline="", encoding="ascii") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "wavelength_nm",
                "alpha_au",
                "depth_joule",
                "depth_uK",
                "depth_Er",
                "axial_freq_hz",
                "axial_freq_khz",
                "radial_freq_hz",
                "radial_freq_khz",
                "single_beam_peak_intensity_w_m2",
                "standing_wave_peak_intensity_w_m2",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    f"{item.wavelength_nm:.9f}",
                    f"{item.alpha_au:.12e}",
                    f"{item.depth_joule:.12e}",
                    f"{item.depth_uK:.12e}",
                    f"{item.depth_Er:.12e}",
                    f"{item.axial_freq_hz:.12e}",
                    f"{item.axial_freq_hz/1e3:.12e}",
                    f"{item.radial_freq_hz:.12e}",
                    f"{item.radial_freq_hz/1e3:.12e}",
                    f"{item.intensity_single_beam_w_m2:.12e}",
                    f"{item.intensity_standing_peak_w_m2:.12e}",
                ]
            )


class LatticeStrengthWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.calc = LatticeStrengthCalculator()
        self.results: List[LatticeResult] = []

        self.setWindowTitle("Rb-85 Lattice Strength (Trap Frequency)")
        self._build_ui()
        self._wire_signals()
        self._sync_mode_inputs()
        self._sync_mf_validator()
        self._clear_plot("Run a scan to display axial trap frequency vs wavelength.")

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        self.single_mode_radio = QRadioButton("Specific wavelength")
        self.scan_mode_radio = QRadioButton("Scan")
        self.single_mode_radio.setChecked(True)
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.scan_mode_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        wavelength_group = QGroupBox("Wavelength Inputs (nm)")
        wavelength_grid = QGridLayout()
        self.single_wavelength_edit = QLineEdit("813")
        self.scan_min_edit = QLineEdit("500")
        self.scan_max_edit = QLineEdit("850")
        wl_validator = QDoubleValidator(0.0, 1e6, 6)
        wl_validator.setNotation(QDoubleValidator.StandardNotation)
        self.single_wavelength_edit.setValidator(wl_validator)
        self.scan_min_edit.setValidator(wl_validator)
        self.scan_max_edit.setValidator(wl_validator)
        wavelength_grid.addWidget(QLabel("Specific wavelength"), 0, 0)
        wavelength_grid.addWidget(self.single_wavelength_edit, 0, 1)
        wavelength_grid.addWidget(QLabel("Scan min"), 1, 0)
        wavelength_grid.addWidget(self.scan_min_edit, 1, 1)
        wavelength_grid.addWidget(QLabel("Scan max"), 2, 0)
        wavelength_grid.addWidget(self.scan_max_edit, 2, 1)
        wavelength_group.setLayout(wavelength_grid)
        layout.addWidget(wavelength_group)

        hfs_group = QGroupBox("Hyperfine Inputs")
        hfs_grid = QGridLayout()
        self.f_edit = QLineEdit("3")
        self.mf_edit = QLineEdit("0")
        self.q_edit = QLineEdit("0")
        self.f_edit.setValidator(QRegularExpressionValidator(QRegularExpression("[23]")))
        self.mf_edit.setValidator(QIntValidator(-3, 3))
        self.q_edit.setValidator(QRegularExpressionValidator(QRegularExpression("-1|0|1")))
        hfs_grid.addWidget(QLabel("F (2 or 3)"), 0, 0)
        hfs_grid.addWidget(self.f_edit, 0, 1)
        hfs_grid.addWidget(QLabel("mF (|mF| <= F)"), 1, 0)
        hfs_grid.addWidget(self.mf_edit, 1, 1)
        hfs_grid.addWidget(QLabel("q (-1, 0, +1)"), 2, 0)
        hfs_grid.addWidget(self.q_edit, 2, 1)
        hfs_group.setLayout(hfs_grid)
        layout.addWidget(hfs_group)

        lattice_group = QGroupBox("Lattice Geometry")
        lattice_grid = QGridLayout()
        self.power_mw_edit = QLineEdit("100")
        self.waist_um_edit = QLineEdit("200")
        geom_validator = QDoubleValidator(0.0, 1e9, 6)
        geom_validator.setNotation(QDoubleValidator.StandardNotation)
        self.power_mw_edit.setValidator(geom_validator)
        self.waist_um_edit.setValidator(geom_validator)
        lattice_grid.addWidget(QLabel("Power per beam (mW)"), 0, 0)
        lattice_grid.addWidget(self.power_mw_edit, 0, 1)
        lattice_grid.addWidget(QLabel("Beam waist (um)"), 1, 0)
        lattice_grid.addWidget(self.waist_um_edit, 1, 1)
        lattice_group.setLayout(lattice_grid)
        layout.addWidget(lattice_group)

        button_row = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate")
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setEnabled(False)
        button_row.addWidget(self.calculate_button)
        button_row.addWidget(self.save_csv_button)
        layout.addLayout(button_row)

        self.status_label = QLabel(
            "Assumptions: counter-propagating equal-power beams, standing-wave depth from peak-to-node."
        )
        layout.addWidget(self.status_label)

        plot_group = QGroupBox("Scan Plot (Axial Trap Frequency)")
        plot_layout = QVBoxLayout()
        self.figure = Figure(figsize=(6.5, 3.4), tight_layout=True)
        self.plot_canvas = FigureCanvas(self.figure)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.plot_axes = self.figure.add_subplot(111)
        plot_layout.addWidget(self.plot_toolbar)
        plot_layout.addWidget(self.plot_canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        self.setLayout(layout)
        self.resize(850, 820)

    def _wire_signals(self) -> None:
        self.single_mode_radio.toggled.connect(self._sync_mode_inputs)
        self.scan_mode_radio.toggled.connect(self._sync_mode_inputs)
        self.f_edit.textChanged.connect(self._sync_mf_validator)
        self.calculate_button.clicked.connect(self._on_calculate)
        self.save_csv_button.clicked.connect(self._on_save_csv)

    def _sync_mode_inputs(self) -> None:
        single_mode = self.single_mode_radio.isChecked()
        self.single_wavelength_edit.setEnabled(single_mode)
        self.scan_min_edit.setEnabled(not single_mode)
        self.scan_max_edit.setEnabled(not single_mode)
        if single_mode:
            self._clear_plot("Scan mode required for plotting.")
        elif not self.results:
            self._clear_plot("Run a scan to display axial trap frequency vs wavelength.")

    def _sync_mf_validator(self) -> None:
        f_value = self.f_edit.text().strip()
        if f_value in ("2", "3"):
            self.mf_edit.setValidator(QIntValidator(-int(f_value), int(f_value)))
        else:
            self.mf_edit.setValidator(QIntValidator(-3, 3))

    def _read_hyperfine(self) -> tuple[int, int, int]:
        f_text = self.f_edit.text().strip()
        mf_text = self.mf_edit.text().strip()
        q_text = self.q_edit.text().strip()
        if f_text not in ("2", "3"):
            raise ValueError("F must be 2 or 3.")
        if q_text not in ("-1", "0", "1"):
            raise ValueError("q must be -1, 0, or 1.")
        if mf_text in ("", "-", "+"):
            raise ValueError("mF is required.")
        F = int(f_text)
        mF = int(mf_text)
        q = int(q_text)
        self.calc.pol.validate_state(F=F, mF=mF)
        return F, mF, q

    def _read_geometry(self) -> tuple[float, float]:
        power_text = self.power_mw_edit.text().strip()
        waist_text = self.waist_um_edit.text().strip()
        if not power_text:
            raise ValueError("Power per beam is required.")
        if not waist_text:
            raise ValueError("Beam waist is required.")
        power_mw = float(power_text)
        waist_um = float(waist_text)
        if power_mw <= 0:
            raise ValueError("Power per beam must be positive.")
        if waist_um <= 0:
            raise ValueError("Beam waist must be positive.")
        return power_mw, waist_um

    def _on_calculate(self) -> None:
        try:
            F, mF, q = self._read_hyperfine()
            power_mw, waist_um = self._read_geometry()

            if self.single_mode_radio.isChecked():
                wl_text = self.single_wavelength_edit.text().strip()
                if not wl_text:
                    raise ValueError("Specific wavelength is required.")
                result = self.calc.evaluate(
                    wavelength_nm=float(wl_text),
                    F=F,
                    mF=mF,
                    q=q,
                    power_mw=power_mw,
                    waist_um=waist_um,
                )
                self.results = [result]
                self._clear_plot("Scan mode required for plotting.")
                self.status_label.setText(
                    "Single point: "
                    f"lambda={result.wavelength_nm:.3f} nm, "
                    f"f_ax={result.axial_freq_hz/1e3:.3f} kHz, "
                    f"f_rad={result.radial_freq_hz/1e3:.3f} kHz, "
                    f"U={result.depth_uK:.3f} uK, "
                    f"alpha={result.alpha_au:.4e} a.u."
                )
            else:
                min_text = self.scan_min_edit.text().strip()
                max_text = self.scan_max_edit.text().strip()
                if not min_text or not max_text:
                    raise ValueError("Scan min and scan max are required.")
                scan_min = float(min_text)
                scan_max = float(max_text)

                wavelengths = build_wavelength_array(scan_min, scan_max, num_points=1000)
                self.results = [
                    self.calc.evaluate(
                        wavelength_nm=float(wl),
                        F=F,
                        mF=mF,
                        q=q,
                        power_mw=power_mw,
                        waist_um=waist_um,
                    )
                    for wl in wavelengths
                ]
                self._plot_scan(self.results, F, mF, q)
                self.status_label.setText(
                    f"Scan complete: {len(self.results)} points, "
                    f"{self.results[0].wavelength_nm:.3f}-{self.results[-1].wavelength_nm:.3f} nm, "
                    f"P={power_mw:.3f} mW/beam, waist={waist_um:.3f} um."
                )

            self.save_csv_button.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Input/Calculation Error", str(exc))

    def _clear_plot(self, message: str) -> None:
        self.plot_axes.clear()
        self.plot_axes.set_xlabel("Wavelength (nm)")
        self.plot_axes.set_ylabel("Axial trap frequency (kHz)")
        self.plot_axes.text(
            0.5, 0.5, message, ha="center", va="center", transform=self.plot_axes.transAxes
        )
        self.plot_axes.grid(True, alpha=0.25)
        self.plot_canvas.draw()

    def _plot_scan(self, results: Sequence[LatticeResult], F: int, mF: int, q: int) -> None:
        x_nm = [r.wavelength_nm for r in results]
        f_ax_khz = [r.axial_freq_hz / 1e3 for r in results]
        f_rad_khz = [r.radial_freq_hz / 1e3 for r in results]

        self.plot_axes.clear()
        self.plot_axes.plot(x_nm, f_ax_khz, linewidth=1.9, label="Axial")
        self.plot_axes.plot(x_nm, f_rad_khz, linewidth=1.2, linestyle="--", label="Radial")
        self.plot_axes.set_xlabel("Wavelength (nm)")
        self.plot_axes.set_ylabel("Trap frequency (kHz)")
        self.plot_axes.set_title(f"Rb-85 lattice frequencies, F={F}, mF={mF}, q={q}")
        self.plot_axes.grid(True, alpha=0.3)
        self.plot_axes.legend()
        self.plot_canvas.draw()

    def _on_save_csv(self) -> None:
        if not self.results:
            QMessageBox.information(self, "No Results", "Run a calculation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save lattice-strength table",
            "rb85_lattice_strength.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            write_csv(path, self.results)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Error", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    window = LatticeStrengthWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
