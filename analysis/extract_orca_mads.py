import pandas as pd
import numpy as np

HARTREE_TO_KJMOL = 2625.5  # conversion factor


def compute_mad_vs_qz(
    csv_path: str, num_carbons: int = 24, num_hydrogens: int = 14
) -> pd.DataFrame:
    """
    Compute deviations per basis combination relative to 'qz_ri' for:
      - Absolute energy (total_energy_hartree)
      - Relative energy (relative_energy_hartree)

    Returns mean absolute deviation (MAD) in Hartree and kJ/mol,
    and RMSD in kJ/mol, averaged over isomers that have a qz_ri reference.
    Only molecules matching the specified carbon/hydrogen counts are considered.
    """

    df = pd.read_csv(csv_path)

    # Filter to target stoichiometry
    sub = df[
        (df["num_carbons"] == num_carbons) & (df["num_hydrogens"] == num_hydrogens)
    ].copy()
    if sub.empty:
        print(
            f"No rows found for {num_carbons}C/{num_hydrogens}H — result will be empty."
        )
        return pd.DataFrame()

    metrics = [
        "total_energy_hartree",
        "relative_energy_hartree",
    ]

    # Keep only relevant columns
    sub_slim = sub[["isomer", "basis_combo_id"] + metrics].drop_duplicates()

    # Retain only isomers that have a qz_ri reference
    has_qz = sub_slim.query("basis_combo_id == 'qz_ri'")[["isomer"]].drop_duplicates()
    sub_slim = sub_slim.merge(has_qz, on="isomer", how="inner")

    # Create wide table (multi-index columns: metric, basis_combo_id)
    wide = sub_slim.set_index(["isomer", "basis_combo_id"])[metrics].unstack(
        "basis_combo_id"
    )

    def mad_vs_qz(wide_df: pd.DataFrame, metric: str) -> pd.Series:
        """Mean absolute deviation (Hartree) vs qz_ri."""
        if metric not in wide_df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        w = wide_df[metric]
        if "qz_ri" not in w.columns:
            return pd.Series(dtype=float)
        ref = w["qz_ri"]
        diffs = (w.sub(ref, axis=0)).abs()
        mad = diffs.mean(skipna=True)
        mad.name = f"MAD_{metric}"
        return mad

    def rmsd_vs_qz(wide_df: pd.DataFrame, metric: str) -> pd.Series:
        """RMSD (Hartree) vs qz_ri."""
        if metric not in wide_df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        w = wide_df[metric]
        if "qz_ri" not in w.columns:
            return pd.Series(dtype=float)
        ref = w["qz_ri"]
        diffs = w.sub(ref, axis=0)
        rmsd = np.sqrt((diffs.pow(2)).mean(skipna=True))
        rmsd.name = f"RMSD_{metric}"
        return rmsd

    # Compute stats in Hartree
    mad_absE = mad_vs_qz(wide, "total_energy_hartree")
    mad_relE = mad_vs_qz(wide, "relative_energy_hartree")
    rmsd_absE = rmsd_vs_qz(wide, "total_energy_hartree")
    rmsd_relE = rmsd_vs_qz(wide, "relative_energy_hartree")

    # Combine into a single DataFrame
    result = pd.concat([mad_absE, mad_relE, rmsd_absE, rmsd_relE], axis=1).reset_index()
    result = result.rename(
        columns={
            "basis_combo_id": "basis",
            "MAD_total_energy_hartree": "MAD_abs_energy_hartree",
            "MAD_relative_energy_hartree": "MAD_relative_energy_hartree",
            "RMSD_total_energy_hartree": "RMSD_abs_energy_hartree",
            "RMSD_relative_energy_hartree": "RMSD_relative_energy_hartree",
        }
    )

    # Convert to kJ/mol
    result["MAD_abs_energy_kJmol"] = result["MAD_abs_energy_hartree"] * HARTREE_TO_KJMOL
    result["MAD_relative_energy_kJmol"] = (
        result["MAD_relative_energy_hartree"] * HARTREE_TO_KJMOL
    )
    result["RMSD_abs_energy_kJmol"] = (
        result["RMSD_abs_energy_hartree"] * HARTREE_TO_KJMOL
    )
    result["RMSD_relative_energy_kJmol"] = (
        result["RMSD_relative_energy_hartree"] * HARTREE_TO_KJMOL
    )

    # Sort so qz_ri appears first
    if "qz_ri" in result["basis"].values:
        result["basis_rank"] = (result["basis"] != "qz_ri").astype(int)
    else:
        result["basis_rank"] = 1

    result = result.sort_values(["basis_rank", "MAD_abs_energy_hartree"]).drop(
        columns="basis_rank"
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MAD and RMSD vs qz_ri by basis (Hartree + kJ/mol)."
    )
    parser.add_argument("csv_path", help="Path to input CSV (e.g., orca_data.csv)")
    parser.add_argument(
        "--carbons",
        type=int,
        default=24,
        help="Number of carbons to filter (default: 24)",
    )
    parser.add_argument(
        "--hydrogens",
        type=int,
        default=14,
        help="Number of hydrogens to filter (default: 14)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="mad_by_basis_24C_14H.csv",
        help="Output CSV filename",
    )
    args = parser.parse_args()

    result = compute_mad_vs_qz(args.csv_path, args.carbons, args.hydrogens)
    if result.empty:
        print("No data found for the specified stoichiometry.")
    else:
        result.to_csv(args.out, index=False)
        print(f"Wrote {args.out} with {len(result)} rows.")
