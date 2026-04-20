"""
Out-of-plane metrics from XYZ geometries for EXESS CSV columns.

The mean molecular plane is a weighted least-squares fit: each atom is weighted by
its atomic number Z. The reference point is the Z-weighted centroid. Distances are
signed lengths along the unit normal; max is the largest |d_i|; mean is the Z-weighted
mean of |d_i|.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Atomic numbers for elements that may appear in dataset XYZ files (extend as needed).
_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}


def _normalize_element(symbol: str) -> str:
    raw = symbol.strip()
    raw = re.sub(r"[^A-Za-z].*", "", raw)
    if not raw:
        raise ValueError(f"Empty element symbol from {symbol!r}")
    if len(raw) == 1:
        return raw.upper()
    return raw[0].upper() + raw[1:].lower()


def atomic_number(symbol: str) -> int:
    key = _normalize_element(symbol)
    if key not in _ATOMIC_NUMBER:
        raise ValueError(f"Unknown element {key!r}; add it to _ATOMIC_NUMBER in exess_plane_displacement.py")
    return _ATOMIC_NUMBER[key]


def read_xyz_symbols_positions_angstrom(path: Path) -> tuple[list[str], np.ndarray]:
    """Return element symbols and an (N, 3) array of Cartesian coordinates in Å."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ too short: {path}")
    n = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(coords) != n:
        raise ValueError(f"Expected {n} atom lines in {path}, got {len(coords)}")
    return symbols, np.asarray(coords, dtype=float)


def max_mean_abs_plane_displacement_z_weighted(
    symbols: list[str], coords: np.ndarray
) -> tuple[float, float]:
    """
    Z-weighted least-squares plane (atomic number weights).

    Reference point: Z-weighted centroid. Plane: minimizes sum_i Z_i * d_i^2 where
    d_i is the distance to the plane. Obtained by SVD on rows sqrt(Z_i) * (x_i - c).

    Returns:
        max_i |d_i| over atoms,
        sum_i Z_i |d_i| / sum_i Z_i  (Z-weighted mean absolute distance).
    """
    if coords.shape[0] < 3:
        raise ValueError("Need at least 3 atoms for a plane fit")
    z = np.array([atomic_number(s) for s in symbols], dtype=float)
    wsum = float(z.sum())
    if wsum <= 0:
        raise ValueError("Non-positive total atomic-number weight")
    c = (z[:, None] * coords).sum(axis=0) / wsum
    x = coords - c
    xw = np.sqrt(z)[:, None] * x
    _, _, vt = np.linalg.svd(xw, full_matrices=False)
    normal = vt[-1].astype(float)
    norm = np.linalg.norm(normal)
    if norm < 1e-14:
        raise ValueError("Degenerate geometry (zero normal)")
    normal /= norm
    signed = x @ normal
    abs_d = np.abs(signed)
    max_abs = float(abs_d.max())
    mean_z_weighted = float((z * abs_d).sum() / wsum)
    return max_abs, mean_z_weighted


def plane_metrics_from_xyz(path: Path) -> tuple[float, float]:
    """Read XYZ and return (max_z_displacement, mean_z_displacement)."""
    symbols, coords = read_xyz_symbols_positions_angstrom(path)
    return max_mean_abs_plane_displacement_z_weighted(symbols, coords)
