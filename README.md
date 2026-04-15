# LatticeCalcs

Small utilities for Rubidium optical lattice calculations and visualization using ARC.

## Requirements

- Python 3.10+
- `arc-alkali-rydberg-calculator`
- `numpy`
- `matplotlib`
- `PyQt6`
- `marimo`

## Install

From within the directory:

```bash
pip install -r requirements.txt
```

## Changing Isotope and Ground Hyperfine State

You can edit the isotope and the ground state of interest and the q of the lattice photon:

```python
ATOM = "Rb85"
F = 3
mF = 3
q = 0
```

## Run

`python lattice-calc-plt.py`

For the notebook-style Marimo version:

`marimo run lattice-calc-mo.py`
