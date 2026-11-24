"""Frozen grid, time and dataset constants for LQM-1e.

Atomic units throughout (Hartree energy, Bohr length, ħ = mₑ = e = 1).
Do not change these without bumping ``SCHEMA_VERSION`` and regenerating all data.
"""

import numpy as np

SCHEMA_VERSION: int = 1

# Space: hard-wall box x ∈ [-L, L], NX points including both walls (iDEA convention).
L: float = 10.0
NX: int = 128
X: np.ndarray = np.linspace(-L, L, NX)
X.flags.writeable = False
DX: float = float(X[1] - X[0])  # = 20/127
NFACE: int = NX - 1  # interior cell faces i+1/2, i = 0..NX-2; walls carry zero current
STENCIL: int = 13  # iDEA finite-difference stencil for the kinetic energy

# Time: frames t_k = k * DT, k = 0..NT-1 (NT-1 = 200 input -> target pairs).
DT: float = 0.2
NT: int = 201
T: np.ndarray = DT * np.arange(NT)
T.flags.writeable = False

# Numerics.
N_FLOOR: float = 1e-12  # floor for log(n) in the softmax head
N_SPEC: int = 16  # number of most-populated H1 eigenlevels stored per trajectory

# Splits and potential families. Integer ids are part of the schema.
SPLITS: tuple[str, ...] = ("train", "val", "test", "ood_field", "ood_Z", "ood_family")
SPLIT_IDS: dict[str, int] = {name: i for i, name in enumerate(SPLITS)}
FAMILIES: tuple[str, ...] = ("atom_fourier", "double_well", "harmonic", "pure_fourier")
FAMILY_IDS: dict[str, int] = {name: i for i, name in enumerate(FAMILIES)}
