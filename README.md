# lattice_calcs

Small utilities for Rubidium optical lattice calculations and visualization using ARC.

## Requirements

- Python 3.10+
- `arc-alkali-rydberg-calculator`
- `numpy`
- `matplotlib`
- `PyQt6`

## Install

From within the directory:

```bash
pip install -r requirements.txt
```

## Run

You can edit the isotope and the ground state of interest and the q of the lattice photon:

```python
ATOM = "Rb85"
F = 3
mF = 3
q = 0
```

`python lattice-calc-plt.py`