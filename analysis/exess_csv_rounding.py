"""Column-specific numeric precision for EXESS-style CSV export."""

from __future__ import annotations

# Hartree energies and related orbital quantities (hartree)
ENERGY_HARTREE_FIELDS = frozenset(
    {
        "total_energy_hartree",
        "isomerization_energy_hartree",
        "scf_energy_hartree",
        "pt2_os_correction_hartree",
        "pt2_ss_correction_hartree",
        "d4_energy_hartree",
        "xc_energy_hartree",
        "nuc_repulsion_energy_hartree",
        "elec_energy_hartree",
        "homo_hartree",
        "lumo_hartree",
        "hlg_hartree",
    }
)

Z_DISPLACEMENT_FIELDS = frozenset({"max_z_displacement", "mean_z_displacement"})

ENERGY_DECIMALS = 8
Z_DECIMALS = 4
TIME_DECIMALS = 4
TFLOP_DECIMALS = 3


def round_exess_csv_row(row: dict) -> dict:
    """Return a copy of row with numeric fields rounded for CSV export."""
    out = {}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = v
            continue
        if k in ENERGY_HARTREE_FIELDS:
            out[k] = round(float(v), ENERGY_DECIMALS)
        elif k in Z_DISPLACEMENT_FIELDS:
            out[k] = round(float(v), Z_DECIMALS)
        elif k.endswith("_time_s"):
            out[k] = round(float(v), TIME_DECIMALS)
        elif "tflop" in k:
            out[k] = round(float(v), TFLOP_DECIMALS)
        else:
            out[k] = v
    return out
