#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
from dftd4.interface import DampingParam, DispersionModel
import common

OUT = Path(__file__).resolve().parent.parent / "outputs" / "dftd4" / "dftd4_results.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

BOHR_RADIUS = 0.52917721067
Z = {"H": 1, "C": 6}
PARAM = DampingParam(s6=0.4612, s8=0.0, a1=0.44, a2=3.60)


def read_xyz(p: Path):
    L = p.read_text().splitlines()
    n = int(L[0])
    lines = L[2 : 2 + n]
    syms = [t.split()[0] for t in lines]
    coords = np.array([[float(x) for x in t.split()[1:4]] for t in lines], float)
    return syms, coords


rows = []
for item in common.all_structures:
    name, xyz = item.name, item.xyz_path
    syms, pos_ang = read_xyz(xyz)
    nums = np.array([Z[s] for s in syms], int)
    pos_bohr = pos_ang / BOHR_RADIUS
    e = float(
        DispersionModel(numbers=nums, positions=pos_bohr, charge=None).get_dispersion(
            PARAM, grad=False
        )["energy"]
    )
    rows.append(
        {
            "system": name,
            "n_atoms": len(syms),
            "d4_energy_hartree": f"{e:.12f}",
        }
    )

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["system", "n_atoms", "d4_energy_hartree"])
    w.writeheader()
    w.writerows(rows)
