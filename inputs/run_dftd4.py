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

# Parameters for revDSD-PBEP86-D4 (current parameters)
PARAM_REVDSD = DampingParam(s6=0.4612, s8=0.0, a1=0.44, a2=3.60)

# Parameters for PBE0 (using built-in PBE0 parameters)
PARAM_PBE0 = DampingParam(method="pbe0")

# Mapping of GGA functional names to dftd4 method names
# For functionals not directly supported, we'll use the closest match or default parameters
GGA_D4_PARAMS = {
    "PBE": DampingParam(method="pbe"),
    "BLYP": DampingParam(method="blyp"),
    "revPBE": DampingParam(method="rpbe"),
    "BP86": DampingParam(method="bp86"),
    "BPW91": DampingParam(method="bpw"),
    "B97-D": DampingParam(method="b97d"),
    # HCTH407 is not supported by dftd4, so it will be skipped with a warning
}

# Mapping of mGGA functional names to dftd4 method names
# For functionals not directly supported, we'll skip with a warning
MGGA_D4_PARAMS = {
    "TPSS": DampingParam(method="tpss"),
    "SCAN": DampingParam(method="scan"),
    "rSCAN": DampingParam(method="rscan"),
    "r2SCAN": DampingParam(method="r2scan"),
    "revTPSS": DampingParam(method="revtpss"),
    "M06-L": DampingParam(method="m06l"),
    # M11-L, B97M-V, MN15L, and t-HCTH are not supported by dftd4, so they will be skipped with a warning
}


def read_xyz(p: Path):
    L = p.read_text().splitlines()
    n = int(L[0])
    lines = L[2 : 2 + n]
    syms = [t.split()[0] for t in lines]
    coords = np.array([[float(x) for x in t.split()[1:4]] for t in lines], float)
    return syms, coords


# Determine which structures are used with which functional
# Structures in revDSD-PBEP86-D4 batches
revdsd_structures = set()
for batch in common.exess_batches:
    for isomer in batch.isomers:
        revdsd_structures.add(isomer.name)
for batch in common.exess_pah335_batches:
    for isomer in batch.isomers:
        revdsd_structures.add(isomer.name)

# Structures in PBE0 batches
pbe0_structures = set()
for batch in common.exess_pah335_pbe_batches:
    for isomer in batch.isomers:
        pbe0_structures.add(isomer.name)

# Structures in GGA batches - map structure name to list of functionals
# Each structure appears in multiple batches (one per functional)
gga_structures = {}  # {structure_name: [functional_name, ...]}
for batch in common.exess_gga_batches:
    functional = batch.functional
    for isomer in batch.isomers:
        if isomer.name not in gga_structures:
            gga_structures[isomer.name] = []
        gga_structures[isomer.name].append(functional)

# Structures in mGGA batches - map structure name to list of functionals
# Each structure appears in multiple batches (one per functional)
mgga_structures = {}  # {structure_name: [functional_name, ...]}
for batch in common.exess_mgga_batches:
    functional = batch.functional
    for isomer in batch.isomers:
        if isomer.name not in mgga_structures:
            mgga_structures[isomer.name] = []
        mgga_structures[isomer.name].append(functional)

rows = []
# Cache XYZ data to avoid reading multiple times for structures used with both functionals
xyz_cache = {}

for item in common.all_structures:
    name, xyz = item.name, item.xyz_path
    
    # Read XYZ data (cache it if structure is used with both functionals)
    if name not in xyz_cache:
        syms, pos_ang = read_xyz(xyz)
        nums = np.array([Z[s] for s in syms], int)
        pos_bohr = pos_ang / BOHR_RADIUS
        xyz_cache[name] = (syms, nums, pos_bohr)
    else:
        syms, nums, pos_bohr = xyz_cache[name]
    
    # Calculate D4 energy with revDSD-PBEP86-D4 parameters if structure is used with that functional
    if name in revdsd_structures:
        e_revdsd = float(
            DispersionModel(numbers=nums, positions=pos_bohr, charge=None).get_dispersion(
                PARAM_REVDSD, grad=False
            )["energy"]
        )
        rows.append(
            {
                "system": name,
                "functional": "revDSD-PBEP86-D4",
                "n_atoms": len(syms),
                "d4_energy_hartree": f"{e_revdsd:.12f}",
            }
        )
    
    # Calculate D4 energy with PBE0 parameters if structure is used with that functional
    if name in pbe0_structures:
        e_pbe0 = float(
            DispersionModel(numbers=nums, positions=pos_bohr, charge=None).get_dispersion(
                PARAM_PBE0, grad=False
            )["energy"]
        )
        rows.append(
            {
                "system": name,
                "functional": "PBE0",
                "n_atoms": len(syms),
                "d4_energy_hartree": f"{e_pbe0:.12f}",
            }
        )
    
    # Calculate D4 energy for GGA functionals (each structure can have multiple GGA functionals)
    if name in gga_structures:
        for functional in gga_structures[name]:
            if functional in GGA_D4_PARAMS:
                d4_param = GGA_D4_PARAMS[functional]
                e_gga = float(
                    DispersionModel(numbers=nums, positions=pos_bohr, charge=None).get_dispersion(
                        d4_param, grad=False
                    )["energy"]
                )
                rows.append(
                    {
                        "system": name,
                        "functional": functional,
                        "n_atoms": len(syms),
                        "d4_energy_hartree": f"{e_gga:.12f}",
                    }
                )
            else:
                print(f"Warning: No D4 parameters found for functional {functional}, skipping {name}")
    
    # Calculate D4 energy for mGGA functionals (each structure can have multiple mGGA functionals)
    if name in mgga_structures:
        for functional in mgga_structures[name]:
            if functional in MGGA_D4_PARAMS:
                d4_param = MGGA_D4_PARAMS[functional]
                e_mgga = float(
                    DispersionModel(numbers=nums, positions=pos_bohr, charge=None).get_dispersion(
                        d4_param, grad=False
                    )["energy"]
                )
                rows.append(
                    {
                        "system": name,
                        "functional": functional,
                        "n_atoms": len(syms),
                        "d4_energy_hartree": f"{e_mgga:.12f}",
                    }
                )
            else:
                print(f"Warning: No D4 parameters found for functional {functional}, skipping {name}")
    
    # If structure is not in any batch, default to revDSD-PBEP86-D4
    if (name not in revdsd_structures and name not in pbe0_structures 
        and name not in gga_structures and name not in mgga_structures):
        e = float(
            DispersionModel(numbers=nums, positions=pos_bohr, charge=None).get_dispersion(
                PARAM_REVDSD, grad=False
            )["energy"]
        )
        rows.append(
            {
                "system": name,
                "functional": "revDSD-PBEP86-D4",
                "n_atoms": len(syms),
                "d4_energy_hartree": f"{e:.12f}",
            }
        )

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["system", "functional", "n_atoms", "d4_energy_hartree"])
    w.writeheader()
    w.writerows(rows)
