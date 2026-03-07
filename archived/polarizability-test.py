from __future__ import annotations

import csv
import inspect
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


@dataclass(frozen=True)
class HyperfineState:
    """Represents a hyperfine sublevel in 5S1/2 of Rb-85."""

    F: int
    mF: int


class Rb85GroundStatePolarizability:
    """ARC-backed calculator for dynamic polarizability of Rb-85 5S1/2."""

    def __init__(self) -> None:
        self.atom = Rubidium85()
        self.n = 5
        self.l = 0
        self.j = 0.5
        self._direct_get_polarizability = getattr(self.atom, "getPolarizability", None)
        self._dynamic_polarizability = None

    @staticmethod
    def allowed_hyperfine_states() -> List[HyperfineState]:
        states: List[HyperfineState] = []
        for F in (2, 3):
            for mF in range(-F, F + 1):
                states.append(HyperfineState(F=F, mF=mF))
        return states

    @staticmethod
    def validate_state(F: int, mF: int) -> None:
        if F not in (2, 3):
            raise ValueError("For Rb-85 5S1/2, F must be 2 or 3.")
        if abs(mF) > F:
            raise ValueError(f"Invalid mF={mF} for F={F}; requires |mF| <= F.")

    def alpha_with_hyperfine(
        self,
        wavelength_nm: float,
        F: int,
        mF: int,
        polarization_q: int = 0,
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
        # Hyperfine-resolved shift approximation for 5S1/2:
        # alpha(F,mF,q) ~= alpha0 + q * (mF / (2F)) * alpha1
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
        # Covers dominant 5S -> nP lines over 500-850 nm and nearby contributions.
        self._dynamic_polarizability.defineBasis(5, 25)

    def _get_alpha_components_dynamic(self, wavelength_m: float) -> tuple[float, float, float]:
        self._ensure_dynamic_backend()
        raw = self._dynamic_polarizability.getPolarizability(
            wavelength_m, units="au"
        )
        return self._extract_alpha_components(raw)

    @staticmethod
    def _extract_alpha(raw) -> float:
        if np.isscalar(raw):
            return float(raw)
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            return float(raw[0])
        raise TypeError(f"Unexpected polarizability return type: {type(raw)!r}")

    @staticmethod
    def _extract_alpha_components(raw) -> tuple[float, float, float]:
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            return float(raw[0]), float(raw[1]), float(raw[2])
        raise TypeError(
            "DynamicPolarizability.getPolarizability returned unexpected type/shape."
        )


def build_wavelength_array(start_nm: float, stop_nm: float, num_points: int) -> np.ndarray:
    if start_nm <= 0 or stop_nm <= 0:
        raise ValueError("Wavelength bounds must be positive.")
    if stop_nm <= start_nm:
        raise ValueError("scan max must be greater than scan min.")
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")
    return np.linspace(start_nm, stop_nm, num_points)


def write_csv(path: str, wavelengths_nm: Sequence[float], alphas: Sequence[float]) -> None:
    with open(path, "w", newline="", encoding="ascii") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["wavelength_nm", "alpha_au"])
        for wl, alpha in zip(wavelengths_nm, alphas):
            writer.writerow([f"{wl:.9f}", f"{alpha:.12e}"])


class PolarizabilityWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.calc = Rb85GroundStatePolarizability()
        self.wavelengths_nm: List[float] = []
        self.alphas_au: List[float] = []

        self.setWindowTitle("Rb-85 Ground-State Polarizability (ARC)")
        self._build_ui()
        self._wire_signals()
        self._sync_mode_inputs()
        self._sync_mf_validator()
        self._clear_plot("Run a scan to display wavelength vs polarizability.")

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
        self.single_wavelength_edit = QLineEdit("780")
        self.scan_min_edit = QLineEdit("500")
        self.scan_max_edit = QLineEdit("850")

        wl_validator = QDoubleValidator(0.0, 1000000.0, 6)
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

        button_row = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate")
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setEnabled(False)
        button_row.addWidget(self.calculate_button)
        button_row.addWidget(self.save_csv_button)
        layout.addLayout(button_row)

        self.status_label = QLabel("No calculation yet.")
        layout.addWidget(self.status_label)

        plot_group = QGroupBox("Scan Plot")
        plot_layout = QVBoxLayout()
        self.figure = Figure(figsize=(6.0, 3.2), tight_layout=True)
        self.plot_canvas = FigureCanvas(self.figure)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.plot_axes = self.figure.add_subplot(111)
        plot_layout.addWidget(self.plot_toolbar)
        plot_layout.addWidget(self.plot_canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        self.setLayout(layout)
        self.resize(780, 760)

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
        elif not self.wavelengths_nm:
            self._clear_plot("Run a scan to display wavelength vs polarizability.")

    def _sync_mf_validator(self) -> None:
        f_value = self.f_edit.text().strip()
        if f_value in ("2", "3"):
            bound = int(f_value)
            self.mf_edit.setValidator(QIntValidator(-bound, bound))
        else:
            self.mf_edit.setValidator(QIntValidator(-3, 3))

    def _read_hyperfine_inputs(self) -> tuple[int, int, int]:
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
        self.calc.validate_state(F=F, mF=mF)
        return F, mF, q

    def _on_calculate(self) -> None:
        try:
            F, mF, q = self._read_hyperfine_inputs()
            if self.single_mode_radio.isChecked():
                wl_text = self.single_wavelength_edit.text().strip()
                if not wl_text:
                    raise ValueError("Specific wavelength is required.")
                wavelength_nm = float(wl_text)
                if wavelength_nm <= 0:
                    raise ValueError("Wavelength must be positive.")

                alpha = self.calc.alpha_with_hyperfine(
                    wavelength_nm=wavelength_nm,
                    F=F,
                    mF=mF,
                    polarization_q=q,
                )
                self.wavelengths_nm = [wavelength_nm]
                self.alphas_au = [alpha]
                self._clear_plot("Scan mode required for plotting.")
                self.status_label.setText(
                    f"Single wavelength result: lambda={wavelength_nm:.6f} nm, "
                    f"alpha={alpha:.10e} a.u. (F={F}, mF={mF}, q={q})"
                )
            else:
                min_text = self.scan_min_edit.text().strip()
                max_text = self.scan_max_edit.text().strip()
                if not min_text or not max_text:
                    raise ValueError("Scan min and scan max are required.")
                scan_min_nm = float(min_text)
                scan_max_nm = float(max_text)

                wavelengths = build_wavelength_array(scan_min_nm, scan_max_nm, num_points=351)
                alphas = [
                    self.calc.alpha_with_hyperfine(
                        wavelength_nm=float(wavelength),
                        F=F,
                        mF=mF,
                        polarization_q=q,
                    )
                    for wavelength in wavelengths
                ]
                self.wavelengths_nm = [float(value) for value in wavelengths]
                self.alphas_au = alphas
                self._plot_scan_results(self.wavelengths_nm, self.alphas_au, F, mF, q)
                self.status_label.setText(
                    f"Scan complete: {len(self.wavelengths_nm)} points from "
                    f"{self.wavelengths_nm[0]:.3f} to {self.wavelengths_nm[-1]:.3f} nm "
                    f"(F={F}, mF={mF}, q={q})."
                )
            self.save_csv_button.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Input/Calculation Error", str(exc))

    def _clear_plot(self, message: str) -> None:
        self.plot_axes.clear()
        self.plot_axes.set_xlabel("Wavelength (nm)")
        self.plot_axes.set_ylabel("Polarizability (a.u.)")
        self.plot_axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.plot_axes.transAxes,
        )
        self.plot_axes.grid(True, alpha=0.25)
        self.plot_canvas.draw()

    def _plot_scan_results(
        self,
        wavelengths_nm: Sequence[float],
        alphas_au: Sequence[float],
        F: int,
        mF: int,
        q: int,
    ) -> None:
        self.plot_axes.clear()
        self.plot_axes.plot(wavelengths_nm, alphas_au, linewidth=1.8)
        self.plot_axes.set_xlabel("Wavelength (nm)")
        self.plot_axes.set_ylabel("Polarizability (a.u.)")
        self.plot_axes.set_title(f"Rb-85 5S1/2 scan, F={F}, mF={mF}, q={q}")
        self.plot_axes.grid(True, alpha=0.3)
        self.plot_canvas.draw()

    def _on_save_csv(self) -> None:
        if not self.wavelengths_nm:
            QMessageBox.information(self, "No Results", "Run a calculation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save polarizability table",
            "rb85_polarizability.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            write_csv(path, self.wavelengths_nm, self.alphas_au)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Error", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    window = PolarizabilityWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
