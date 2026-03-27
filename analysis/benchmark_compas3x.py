#!/usr/bin/env python3
"""Benchmark SVWN5 and GGAs against revDSD-PBEP86-D4(noFC) for COMPAS-3x geometries."""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import re
from contextlib import contextmanager
from pathlib import Path
from scipy.stats import linregress
from plotting_utils import (
    create_scatter_plot,
    HARTREE_TO_KJ_PER_MOL,
    calculate_stats,
    extract_common_id,
    format_functional_name,
)


def _maxz_common_ids(df: pd.DataFrame, threshold: float) -> tuple[set[str], set[str]]:
    """Return (lt_ids, ge_ids) for COMPAS-3x GFN2-xTB rows based on max_z_displacement."""
    if "max_z_displacement" not in df.columns:
        raise ValueError(
            "EXESS CSV missing max_z_displacement (run extract_exess_data.py)."
        )
    sub = df[
        (df["isomer_name"].str.contains("compas3x", case=False, na=False))
        & (df["optimizer"] == "GFN2-xTB")
    ].copy()
    sub["common_id"] = sub["isomer_name"].apply(extract_common_id)
    sub = sub.dropna(subset=["common_id", "max_z_displacement"])
    g = sub.groupby("common_id", as_index=False)["max_z_displacement"].first()
    lt = set(g.loc[g["max_z_displacement"] < threshold, "common_id"].astype(str))
    ge = set(g.loc[g["max_z_displacement"] >= threshold, "common_id"].astype(str))
    return lt, ge


def _subset_result(result: dict, allowed_ids: set[str]) -> dict | None:
    """Filter a benchmark result down to allowed common_ids and recompute stats."""
    merged = result.get("merged")
    if merged is None or "common_id" not in merged.columns:
        return None
    mask = merged["common_id"].astype(str).isin(allowed_ids).values
    if mask.sum() < 50:
        return None
    func = np.asarray(result["func_energies"].values)[mask]
    ref = np.asarray(result["ref_energies"].values)[mask]
    deviations = func - ref
    mad = float(np.mean(np.abs(deviations)))
    msd = float(np.mean(deviations))
    r_squared, _, _ = calculate_stats(ref, func)
    out = dict(result)
    out["n_structures"] = int(mask.sum())
    out["mad_kjmol"] = mad
    out["msd_kjmol"] = msd
    out["r_squared"] = float(r_squared)
    return out


@contextmanager
def _latex_output_stream(output_path):
    """Path or writable file-like (e.g. second table appended to same .tex file)."""
    if hasattr(output_path, "write"):
        yield output_path
    else:
        with open(output_path, "w", encoding="utf-8") as fp:
            yield fp


def compute_isomerization_energies(
    df, include_d4, reference_method, revdsd_min_isomers=None
):
    """Compute isomerization energies from a dataframe.

    Args:
        df: Dataframe with energy columns
        include_d4: If True, use total_energy_hartree, else use scf_energy_hartree
        reference_method: 'min' or 'avg' for reference energy calculation (ignored if revdsd_min_isomers provided)
        revdsd_min_isomers: Dict mapping (n_carbons, n_hydrogens) -> common_id of minimum energy isomer from revDSD-PBEP86-D4

    Returns:
        Series with isomerization energies indexed by common_id
    """
    energy_col = "total_energy_hartree" if include_d4 else "scf_energy_hartree"
    df = df.copy()
    df["energy"] = df[energy_col]

    if df["energy"].isna().any():
        missing = df[df["energy"].isna()]["common_id"].tolist()
        raise ValueError(f"Missing {energy_col} for structures: {missing[:10]}")

    if df[["n_carbons", "n_hydrogens"]].isna().any().any():
        raise ValueError("Missing n_carbons or n_hydrogens")

    # Use revDSD-PBEP86-D4 minimum isomer if provided
    if revdsd_min_isomers is not None:
        # Build lookup: (basis_set, common_id) -> energy for this functional
        energy_lookup = {}
        for _, row in df.iterrows():
            key = (row["basis_set"], row["common_id"])
            energy_lookup[key] = row["energy"]

        # For each row, find the energy of the minimum isomer (calculated with same basis_set)
        def get_ref_energy(row):
            ch_key = (row["n_carbons"], row["n_hydrogens"])
            min_common_id = revdsd_min_isomers.get(ch_key)
            if min_common_id is None:
                return None
            lookup_key = (row["basis_set"], min_common_id)
            return energy_lookup.get(lookup_key)

        df["ref_energy"] = df.apply(get_ref_energy, axis=1)
        if df["ref_energy"].isna().any():
            missing = df[df["ref_energy"].isna()][
                ["common_id", "n_carbons", "n_hydrogens", "basis_set"]
            ].drop_duplicates()
            raise ValueError(
                f"Could not find reference energy for minimum isomer:\n{missing.head(10)}"
            )
    else:
        # Fallback to old method (per functional/basis group)
        group_cols = ["functional", "basis_set", "n_carbons", "n_hydrogens"]
        agg_method = "mean" if reference_method == "avg" else reference_method
        ref_energy = df.groupby(group_cols)["energy"].agg(agg_method).to_dict()
        df["ref_energy"] = df.apply(
            lambda r: ref_energy.get(tuple(r[c] for c in group_cols)), axis=1
        )

        if df["ref_energy"].isna().any():
            missing = df[df["ref_energy"].isna()][
                ["common_id"] + group_cols
            ].drop_duplicates()
            raise ValueError(f"Could not find reference energy:\n{missing.head(10)}")

    df["isomerization_energy"] = df["energy"] - df["ref_energy"]
    return df.set_index("common_id")["isomerization_energy"]


def find_minimum_energy_isomers(df, include_d4):
    """Find the minimum energy isomer for each C/H group.

    Args:
        df: Dataframe with energy columns and common_id
        include_d4: If True, use total_energy_hartree, else use scf_energy_hartree

    Returns:
        Dictionary mapping (n_carbons, n_hydrogens) -> common_id of minimum energy isomer
    """
    energy_col = "total_energy_hartree" if include_d4 else "scf_energy_hartree"
    df = df.copy()
    df["energy"] = df[energy_col]

    # Group by C/H and find minimum energy isomer
    min_isomers = {}
    for (n_c, n_h), group in df.groupby(["n_carbons", "n_hydrogens"]):
        min_idx = group["energy"].idxmin()
        min_isomer = group.loc[min_idx, "common_id"]
        min_isomers[(n_c, n_h)] = min_isomer

    return min_isomers


def process_functional_comparison(
    df,
    functional,
    reference_functional="revDSD-PBEP86-D4",
    include_d4=True,
    output_dir=None,
    reference_method="min",
    plot_limits=None,
    quiet=False,
):
    """Process comparison for a specific functional."""
    # Filter to COMPAS-3x geometries
    compas3x_df = df[
        (df["isomer_name"].str.contains("compas3x", case=False, na=False))
        & (df["optimizer"] == "GFN2-xTB")
    ].copy()
    compas3x_df["common_id"] = compas3x_df["isomer_name"].apply(extract_common_id)
    compas3x_df = compas3x_df.dropna(subset=["common_id"])

    func_df = compas3x_df[compas3x_df["functional"] == functional].copy()
    ref_df = compas3x_df[compas3x_df["functional"] == reference_functional].copy()

    if len(func_df) == 0 or len(ref_df) == 0:
        return None

    # Find common structures
    common_ids = set(func_df["common_id"].unique()) & set(ref_df["common_id"].unique())
    if len(common_ids) == 0:
        raise ValueError(
            f"No matching structures found for {functional} vs {reference_functional}"
        )

    func_df_common = func_df[func_df["common_id"].isin(common_ids)].copy()
    ref_df_common = ref_df[ref_df["common_id"].isin(common_ids)].copy()

    # Find minimum energy isomer (common_id) for each (C, H) group using revDSD-PBEP86-D4
    revdsd_min_isomers = {}
    for (n_c, n_h), group in ref_df_common.groupby(["n_carbons", "n_hydrogens"]):
        min_idx = group["total_energy_hartree"].idxmin()
        min_common_id = group.loc[min_idx, "common_id"]
        revdsd_min_isomers[(n_c, n_h)] = min_common_id

    # For linear fit, we still need to compute isomerization energies first (use 'min' as base)
    # The linear fit correction will be applied to the isomerization energies afterward
    base_reference_method = (
        "min" if reference_method == "linear_fit" else reference_method
    )

    # Compute isomerization energies using the minimum isomer's energy (calculated with each functional) as reference
    func_energies = compute_isomerization_energies(
        func_df_common, include_d4, base_reference_method, revdsd_min_isomers
    )
    ref_energies = compute_isomerization_energies(
        ref_df_common, True, base_reference_method, revdsd_min_isomers
    )  # Always use total_energy for reference

    # Merge to get common structures
    merged = func_df_common[["common_id"]].merge(
        ref_df_common[["common_id"]], on="common_id", how="inner"
    )
    merged = merged.merge(
        func_energies.to_frame("isomerization_energy_func"),
        left_on="common_id",
        right_index=True,
    )
    merged = merged.merge(
        ref_energies.to_frame("isomerization_energy_ref"),
        left_on="common_id",
        right_index=True,
    )

    if (
        merged[["isomerization_energy_func", "isomerization_energy_ref"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("NaN isomerization energies found")

    # Convert to kJ/mol
    func_energies_kjmol = merged["isomerization_energy_func"] * HARTREE_TO_KJ_PER_MOL
    ref_energies_kjmol = merged["isomerization_energy_ref"] * HARTREE_TO_KJ_PER_MOL

    # Calculate original deviations before linear fit correction
    original_deviations = func_energies_kjmol - ref_energies_kjmol
    original_mad_kjmol = np.mean(np.abs(original_deviations))
    original_msd_kjmol = np.mean(original_deviations)

    # Save original energies for small plots (before linear fit correction)
    original_func_energies_kjmol = func_energies_kjmol.copy()

    # Apply linear fit correction if requested
    gradient = None
    offset = None
    if reference_method == "linear_fit":
        # Fit: ref = gradient * func + offset (fit reference as function of functional, y->x fit)
        slope, intercept, r_value, p_value, std_err = linregress(
            func_energies_kjmol, ref_energies_kjmol
        )
        gradient = slope
        offset = intercept
        # Correct: func_corrected = func * gradient + offset
        func_energies_kjmol = func_energies_kjmol * gradient + offset

    deviations = func_energies_kjmol - ref_energies_kjmol

    mad_kjmol = np.mean(np.abs(deviations))
    msd_kjmol = np.mean(deviations)
    rmsd = np.sqrt(np.mean(deviations**2))
    r_squared, _, mad_percentage = calculate_stats(
        ref_energies_kjmol, func_energies_kjmol
    )

    # Calculate additional statistics
    n_negative_isomer_energies = (func_energies_kjmol < 0).sum()
    n_overestimated = (deviations > 0).sum()
    n_underestimated = (deviations < 0).sum()
    n_overestimates_over_100 = (deviations > 100).sum()
    max_deviation = deviations.max()
    min_deviation = deviations.min()

    # Print results (unless quiet mode)
    if not quiet:
        d4_label = "with D4" if include_d4 else "without D4"
        if reference_method == "linear_fit":
            ref_method_label = "linear fit corrected"
        elif reference_method == "min":
            ref_method_label = "minimum"
        else:
            ref_method_label = "average"
        print(f"\n{'='*60}")
        print(
            f"{functional} ({d4_label}, relative to {ref_method_label}) vs {reference_functional} (COMPAS-3x)"
        )
        print(f"{'='*60}")
        print(f"Number of matched structures: {len(merged)}")
        if gradient is not None and offset is not None:
            print(
                f"  Linear fit: gradient = {gradient:.4f}, offset = {offset:.3f} kJ/mol"
            )
        print(f"  MAD: {mad_kjmol:.3f} kJ/mol")
        print(f"  MSD: {msd_kjmol:.3f} kJ/mol")
        print(f"  RMSD: {rmsd:.3f} kJ/mol")
        if not output_dir:
            print(f"  r²: {r_squared:.4f}")
        print(f"  MAD as percentage: {mad_percentage:.2f}%")
        print(f"  Number of negative isomer energies: {n_negative_isomer_energies}")
        print(f"  Number of overestimated isomer energies: {n_overestimated}")
        print(f"  Number of underestimated isomer energies: {n_underestimated}")
        print(f"  Number of overestimates over 100 kJ/mol: {n_overestimates_over_100}")
        print(f"  Maximum deviation: {max_deviation:.3f} kJ/mol")
        print(f"  Minimum deviation: {min_deviation:.3f} kJ/mol")

        # Top deviations
        print(f"\nTop 10 largest deviations:")
        top_indices = deviations.abs().nlargest(10).index
        for idx in top_indices:
            print(f"  {merged.loc[idx, 'common_id']}: {deviations.loc[idx]:.3f} kJ/mol")
    else:
        d4_label = "with D4" if include_d4 else "without D4"

    # Create scatter plot
    if output_dir:
        func_safe = (
            functional.replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        d4_suffix = "_with_d4" if include_d4 else "_without_d4"
        plot_path = output_dir / f"compas3x_{func_safe}{d4_suffix}_vs_revdsd.png"
        plot_path_small = (
            output_dir / f"compas3x_{func_safe}{d4_suffix}_vs_revdsd_small.png"
        )
        # Format functional names for display
        func_display = format_functional_name(functional)
        ref_display = format_functional_name(reference_functional)
        # Format y-axis label: add "-D4" suffix when D4 is included, otherwise just the functional name
        ylabel_func = f"{func_display}-D4" if include_d4 else func_display
        # Create regular-sized plot
        xlim = plot_limits[0] if plot_limits else None
        ylim = plot_limits[1] if plot_limits else None
        create_scatter_plot(
            ref_energies_kjmol,
            func_energies_kjmol,
            rf"{ref_display} $\Delta E$ (kJ/mol)",
            rf"{ylabel_func} $\Delta E$ (kJ/mol)",
            plot_path,
            mad_kjmol=mad_kjmol,
            msd_kjmol=msd_kjmol,
            xlim=xlim,
            ylim=ylim,
            show_linear_fits=True,
            fit_stats_label=(
                None
                if quiet
                else f"{functional} ({d4_label}) vs {reference_functional}"
            ),
        )
        if not quiet:
            print(f"\nPlot saved to: {plot_path}")
        # Create small version for two-across single column layout
        # Use original (uncorrected) energies for small plots
        if reference_method == "linear_fit":
            small_func_energies = original_func_energies_kjmol
            small_mad = original_mad_kjmol
            small_msd = original_msd_kjmol
        else:
            small_func_energies = func_energies_kjmol
            small_mad = mad_kjmol
            small_msd = msd_kjmol
        create_scatter_plot(
            ref_energies_kjmol,
            small_func_energies,
            rf"{ref_display} $\Delta E$ (kJ/mol)",
            rf"{ylabel_func} $\Delta E$ (kJ/mol)",
            plot_path_small,
            mad_kjmol=small_mad,
            msd_kjmol=small_msd,
            figsize=(1.65, 1.65),
            xlim=xlim,
            ylim=ylim,
            show_linear_fits=False,
        )
        if not quiet:
            print(f"Small plot saved to: {plot_path_small}")

    return {
        "functional": functional,
        "include_d4": include_d4,
        "n_structures": len(merged),
        "mad_kjmol": mad_kjmol,
        "msd_kjmol": msd_kjmol,
        "rmsd": rmsd,
        "r_squared": r_squared,
        "mad_percentage": mad_percentage,
        "original_mad_kjmol": (
            original_mad_kjmol if reference_method == "linear_fit" else mad_kjmol
        ),
        "n_negative_isomer_energies": n_negative_isomer_energies,
        "n_overestimated": n_overestimated,
        "n_underestimated": n_underestimated,
        "n_overestimates_over_100": n_overestimates_over_100,
        "max_deviation": max_deviation,
        "min_deviation": min_deviation,
        "merged": merged,
        "func_energies": func_energies_kjmol,
        "ref_energies": ref_energies_kjmol,
        "deviations": deviations,
        "gradient": gradient,
        "offset": offset,
    }


def process_xtb_comparison(
    df, reference_functional="revDSD-PBEP86-D4", output_dir=None, reference_method="min"
):
    """Process comparison for GFN2-xTB."""
    # Filter to COMPAS-3x geometries
    compas3x_df = df[
        (df["isomer_name"].str.contains("compas3x", case=False, na=False))
        & (df["optimizer"] == "GFN2-xTB")
    ].copy()
    compas3x_df["common_id"] = compas3x_df["isomer_name"].apply(extract_common_id)
    compas3x_df = compas3x_df.dropna(subset=["common_id"])

    # Get revDSD-PBEP86-D4 data
    ref_df = compas3x_df[compas3x_df["functional"] == reference_functional].copy()

    if len(ref_df) == 0:
        return None

    # Find minimum energy isomer for each (C, H) group using revDSD-PBEP86-D4
    revdsd_min_isomers = {}
    for (n_c, n_h), group in ref_df.groupby(["n_carbons", "n_hydrogens"]):
        min_idx = group["total_energy_hartree"].idxmin()
        min_common_id = group.loc[min_idx, "common_id"]
        revdsd_min_isomers[(n_c, n_h)] = min_common_id

    # Load COMPAS-3x data for xTB energies from local cache
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    compas3x_path = (
        project_root / ".compas_cache" / "compas" / "COMPAS-3" / "compas-3x.csv"
    )

    if not compas3x_path.exists():
        print(f"Warning: COMPAS-3x CSV not found at {compas3x_path}")
        return None

    compas3x_data = pd.read_csv(compas3x_path)

    # Extract common_id from molecule column
    if "molecule" not in compas3x_data.columns:
        print("Warning: 'molecule' column not found in COMPAS-3x data")
        return None

    compas3x_data["common_id"] = compas3x_data["molecule"].apply(extract_common_id)

    # Extract n_carbons and n_hydrogens from common_id (format: c16h10_0pent_1)
    def extract_carbons_hydrogens(common_id):
        match = re.match(r"c(\d+)h(\d+)", common_id)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    compas3x_data[["n_carbons", "n_hydrogens"]] = compas3x_data["common_id"].apply(
        lambda x: pd.Series(extract_carbons_hydrogens(x))
    )

    # Get GFN2-xTB absolute energies (convert from eV to hartree)
    if "Etot_eV" not in compas3x_data.columns:
        print("Warning: 'Etot_eV' column not found in COMPAS-3x data")
        return None

    # Convert eV to hartree (1 eV = 0.0367493 hartree)
    compas3x_data["xtb_energy_hartree"] = compas3x_data["Etot_eV"] * 0.0367493

    # Build lookup for xTB energies by common_id
    xtb_energy_lookup = compas3x_data.set_index("common_id")[
        "xtb_energy_hartree"
    ].to_dict()

    # Compute xTB isomerization energies relative to minimum revDSD-PBEP86-D4 isomer
    def compute_xtb_isomerization(row):
        ch_key = (row["n_carbons"], row["n_hydrogens"])
        min_common_id = revdsd_min_isomers.get(ch_key)
        if min_common_id is None:
            return None
        xtb_energy = xtb_energy_lookup.get(row["common_id"])
        xtb_min_energy = xtb_energy_lookup.get(min_common_id)
        if xtb_energy is None or xtb_min_energy is None:
            return None
        return xtb_energy - xtb_min_energy

    compas3x_data["xtb_isomerization_hartree"] = compas3x_data.apply(
        compute_xtb_isomerization, axis=1
    )

    # Compute reference isomerization energies
    ref_energies = compute_isomerization_energies(
        ref_df, True, reference_method, revdsd_min_isomers
    )

    # Merge to get common structures
    merged = ref_df[["common_id"]].merge(
        compas3x_data[["common_id", "xtb_isomerization_hartree"]],
        on="common_id",
        how="inner",
    )
    merged = merged.dropna(subset=["xtb_isomerization_hartree"])
    merged = merged.merge(
        ref_energies.to_frame("isomerization_energy_ref"),
        left_on="common_id",
        right_index=True,
    )

    if len(merged) == 0:
        print(
            "Warning: No matching structures found between GFN2-xTB and revDSD-PBEP86-D4"
        )
        return None

    if (
        merged[["xtb_isomerization_hartree", "isomerization_energy_ref"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("NaN isomerization energies found")

    # Convert to kJ/mol
    xtb_energies_kjmol = merged["xtb_isomerization_hartree"] * HARTREE_TO_KJ_PER_MOL
    ref_energies_kjmol = merged["isomerization_energy_ref"] * HARTREE_TO_KJ_PER_MOL

    # Apply linear fit correction if requested
    gradient = None
    offset = None
    if reference_method == "linear_fit":
        # Fit: ref = gradient * xtb + offset (fit reference as function of xTB, y->x fit)
        slope, intercept, r_value, p_value, std_err = linregress(
            xtb_energies_kjmol, ref_energies_kjmol
        )
        gradient = slope
        offset = intercept
        # Correct: xtb_corrected = xtb * gradient + offset
        xtb_energies_kjmol = xtb_energies_kjmol * gradient + offset

    deviations = xtb_energies_kjmol - ref_energies_kjmol

    mad_kjmol = np.mean(np.abs(deviations))
    msd_kjmol = np.mean(deviations)
    rmsd = np.sqrt(np.mean(deviations**2))
    r_squared, _, mad_percentage = calculate_stats(
        ref_energies_kjmol, xtb_energies_kjmol
    )

    # Calculate additional statistics
    n_negative_isomer_energies = (xtb_energies_kjmol < 0).sum()
    n_overestimated = (deviations > 0).sum()
    n_underestimated = (deviations < 0).sum()
    n_overestimates_over_100 = (deviations > 100).sum()
    max_deviation = deviations.max()
    min_deviation = deviations.min()

    # Print results
    if reference_method == "linear_fit":
        ref_method_label = "linear fit corrected"
    elif reference_method == "min":
        ref_method_label = "minimum"
    else:
        ref_method_label = "average"
    print(f"\n{'='*60}")
    print(
        f"GFN2-xTB (relative to {ref_method_label}) vs {reference_functional} (COMPAS-3x)"
    )
    print(f"{'='*60}")
    print(f"Number of matched structures: {len(merged)}")
    if gradient is not None and offset is not None:
        print(f"  Linear fit: gradient = {gradient:.4f}, offset = {offset:.3f} kJ/mol")
    print(f"  MAD: {mad_kjmol:.3f} kJ/mol")
    print(f"  MSD: {msd_kjmol:.3f} kJ/mol")
    print(f"  RMSD: {rmsd:.3f} kJ/mol")
    print(f"  r²: {r_squared:.4f}")
    print(f"  MAD as percentage: {mad_percentage:.2f}%")
    print(f"  Number of negative isomer energies: {n_negative_isomer_energies}")
    print(f"  Number of overestimated isomer energies: {n_overestimated}")
    print(f"  Number of underestimated isomer energies: {n_underestimated}")
    print(f"  Number of overestimates over 100 kJ/mol: {n_overestimates_over_100}")
    print(f"  Maximum deviation: {max_deviation:.3f} kJ/mol")
    print(f"  Minimum deviation: {min_deviation:.3f} kJ/mol")

    # Top deviations
    print(f"\nTop 10 largest deviations:")
    top_indices = deviations.abs().nlargest(10).index
    for idx in top_indices:
        print(f"  {merged.loc[idx, 'common_id']}: {deviations.loc[idx]:.3f} kJ/mol")

    # Create scatter plot (already done by generate_supporting_info.py, but we could create it here too)
    # The plot is handled separately in generate_supporting_info.py

    return {
        "functional": "GFN2-xTB",
        "include_d4": None,
        "n_structures": len(merged),
        "mad_kjmol": mad_kjmol,
        "msd_kjmol": msd_kjmol,
        "rmsd": rmsd,
        "r_squared": r_squared,
        "mad_percentage": mad_percentage,
        "n_negative_isomer_energies": n_negative_isomer_energies,
        "n_overestimated": n_overestimated,
        "n_underestimated": n_underestimated,
        "n_overestimates_over_100": n_overestimates_over_100,
        "max_deviation": max_deviation,
        "min_deviation": min_deviation,
        "merged": merged,
        "func_energies": xtb_energies_kjmol,
        "ref_energies": ref_energies_kjmol,
        "deviations": deviations,
        "gradient": gradient,
        "offset": offset,
    }


def generate_latex_table(
    results,
    output_path,
    caption_prefix: str = "",
    label: str = "tab:compas3x_benchmarks",
):
    """Generate a LaTeX table with benchmark statistics.

    output_path may be a Path or a writable file-like object.
    """
    # Organize results by functional
    func_data = {}
    xtb_data = None
    for result in results:
        func = result["functional"]
        if func == "GFN2-xTB":
            # xTB doesn't have with/without D4 variants
            xtb_data = {
                "mad": result["mad_kjmol"],
                "msd": result["msd_kjmol"],
                "r_squared": result["r_squared"],
                "original_mad": result.get("original_mad_kjmol", result["mad_kjmol"]),
                "gradient": result.get("gradient"),
                "offset": result.get("offset"),
            }
        else:
            if func not in func_data:
                func_data[func] = {}
            key = "with_d4" if result["include_d4"] else "without_d4"
            func_data[func][key] = {
                "mad": result["mad_kjmol"],
                "msd": result["msd_kjmol"],
                "r_squared": result["r_squared"],
                "original_mad": result.get("original_mad_kjmol", result["mad_kjmol"]),
                "gradient": result.get("gradient"),
                "offset": result.get("offset"),
            }

    # Find best values (including xTB)
    all_mads = [d[k]["mad"] for d in func_data.values() for k in d]
    all_r2s = [d[k]["r_squared"] for d in func_data.values() for k in d]
    if xtb_data is not None:
        all_mads.append(xtb_data["mad"])
        all_r2s.append(xtb_data["r_squared"])
    best_mad = min(all_mads) if all_mads else None
    best_r2 = max(all_r2s) if all_r2s else None

    # Define functional categories and sort
    lda_functionals = ["SVWN5"]
    gga_functionals = ["PBE", "BLYP", "revPBE", "BP86", "BPW91", "B97-D", "HCTH407"]
    mgga_functionals = [
        "TPSS",
        "MN15L",
        "SCAN",
        "rSCAN",
        "r2SCAN",
        "revTPSS",
        "t-HCTH",
        "M06-L",
        "M11-L",
    ]

    def get_mad_for_ordering(func):
        data = func_data[func]
        if func in lda_functionals:
            return data.get("without_d4", {}).get("mad", float("inf"))
        return data.get("with_d4", {}).get("mad") or data.get("without_d4", {}).get(
            "mad", float("inf")
        )

    functional_order = []
    for category in [lda_functionals, gga_functionals, mgga_functionals]:
        functional_order.extend(
            sorted(
                [f for f in category if f in func_data],
                key=get_mad_for_ordering,
                reverse=True,
            )
        )

    # Write LaTeX table
    with _latex_output_stream(output_path) as f:
        f.write("% Requires: \\usepackage{booktabs, multirow, rotating, graphicx}\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        # Check if we have linear fit data (gradient/offset)
        has_linear_fit = any(
            d.get("with_d4", {}).get("gradient") is not None
            or d.get("without_d4", {}).get("gradient") is not None
            for d in func_data.values()
        )

        if has_linear_fit:
            f.write(
                "\\begin{tabular}{@{}c@{\\hspace{0.8em}}l@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{}}\n"
            )
            f.write("\\toprule\n")
            f.write(
                " & \\multirow{3}{*}{Functional} & \\multicolumn{3}{c}{Without D4} & \\multicolumn{3}{c}{With D4} \\\\\n"
            )
            f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n")
            f.write(
                " & & \\shortstack{Slope} & \\shortstack{Offset\\\\(kJ/mol)} & \\shortstack{MAD\\\\ (corrected)\\\\ (kJ/mol)} & \\shortstack{Slope} & \\shortstack{Offset\\\\(kJ/mol)} & \\shortstack{MAD\\\\ (corrected)\\\\ (kJ/mol)} \\\\\n"
            )
        else:
            f.write(
                "\\begin{tabular}{@{}c@{\\hspace{0.8em}}l@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{}}\n"
            )
            f.write("\\toprule\n")
            f.write(
                " & \\multirow{3}{*}{Functional} & \\multicolumn{3}{c}{Without D4} & \\multicolumn{3}{c}{With D4} \\\\\n"
            )
            f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n")
            f.write(
                " & & \\shortstack{MAD\\\\\\footnotesize(kJ/mol)} & \\shortstack{MSD\\\\\\footnotesize(kJ/mol)} & \\parbox[t]{2em}{$r^2$\\vspace{0.5em}} & \\shortstack{MAD\\\\\\footnotesize(kJ/mol)} & \\shortstack{MSD\\\\\\footnotesize(kJ/mol)} & \\parbox[t]{2em}{$r^2$\\vspace{0.5em}} \\\\\n"
            )
        f.write("\\midrule\n")

        # Define formatting functions
        def format_value(val, best_val, is_r2=False):
            if val is None:
                return "---"
            fmt = f"{val:.3f}" if is_r2 else f"{val:.2f}"
            threshold = 0.001 if is_r2 else 0.01
            if best_val is not None and abs(val - best_val) < threshold:
                return fmt
            return fmt

        def format_gradient(grad):
            if grad is None:
                return "---"
            return f"{grad:.3f}"

        def format_offset(off):
            if off is None:
                return "---"
            sign = "$-$" if off < 0 else "$+$"
            return f"{sign}{abs(off):.2f}"

        def format_msd(msd):
            if msd is None:
                return "---"
            sign = "$-$" if msd < 0 else "$+$"
            return f"{sign}{abs(msd):.2f}"

        # Add GFN2-xTB row at the top if present
        if xtb_data is not None:
            xtb_mad = xtb_data.get("mad")
            xtb_msd = xtb_data.get("msd")
            xtb_r2 = xtb_data.get("r_squared")
            xtb_gradient = xtb_data.get("gradient")
            xtb_offset = xtb_data.get("offset")

            if has_linear_fit:
                xtb_grad_str = (
                    format_gradient(xtb_gradient) if xtb_gradient is not None else "---"
                )
                xtb_off_str = (
                    format_offset(xtb_offset) if xtb_offset is not None else "---"
                )
                xtb_corrected_mad_str = format_value(xtb_mad, None)
                f.write(
                    f" & GFN2--xTB & --- & --- & --- & {xtb_grad_str} & {xtb_off_str} & {xtb_corrected_mad_str} \\\\\n"
                )
            else:
                xtb_mad_str = format_value(xtb_mad, best_mad)
                xtb_msd_str = format_msd(xtb_msd)
                xtb_r2_str = format_value(xtb_r2, best_r2, True)
                f.write(
                    f" & GFN2--xTB & --- & --- & --- & {xtb_mad_str} & {xtb_msd_str} & {xtb_r2_str} \\\\\n"
                )
            f.write("\\midrule\n")

        current_category = None
        category_counts = {
            "LDA": sum(1 for f in lda_functionals if f in func_data),
            "GGA": sum(1 for f in gga_functionals if f in func_data),
            "MGGA": sum(1 for f in mgga_functionals if f in func_data),
        }

        for func in functional_order:
            if func not in func_data:
                continue

            if func in lda_functionals:
                category = "LDA"
            elif func in gga_functionals:
                category = "GGA"
            elif func in mgga_functionals:
                category = "MGGA"
            else:
                category = None
            is_first_in_category = category != current_category
            if is_first_in_category:
                if current_category is not None:
                    f.write("\\midrule\n")
                current_category = category
                remaining = category_counts.get(category, 0)
                if remaining == 1:
                    content = f"\\raisebox{{-0.5\\height}}{{\\rotatebox{{90}}{{\\emph{{{category}}}}}}}"
                    f.write(f"\\multirow{{{remaining}}}{{*}}{{{content}}}")
                else:
                    f.write(
                        f"\\multirow{{{remaining}}}{{*}}{{\\rotatebox{{90}}{{\\emph{{{category}}}}}}}"
                    )

            data = func_data[func]
            # Format functional name for LaTeX
            if func == "t-HCTH":
                func_display = r"$\tau$--HCTH"
            else:
                func_display = func.replace("-", "--")

            if has_linear_fit:
                without_d4_gradient = data.get("without_d4", {}).get("gradient")
                without_d4_offset = data.get("without_d4", {}).get("offset")
                without_d4_corrected_mad = (
                    data.get("without_d4", {}).get("mad")
                    if without_d4_gradient is not None
                    else None
                )
                without_d4_grad_str = format_gradient(without_d4_gradient)
                without_d4_off_str = format_offset(without_d4_offset)
                without_d4_corrected_str = format_value(without_d4_corrected_mad, None)

                with_d4_gradient = data.get("with_d4", {}).get("gradient")
                with_d4_offset = data.get("with_d4", {}).get("offset")
                with_d4_corrected_mad = (
                    data.get("with_d4", {}).get("mad")
                    if with_d4_gradient is not None
                    else None
                )
                with_d4_grad_str = format_gradient(with_d4_gradient)
                with_d4_off_str = format_offset(with_d4_offset)
                with_d4_corrected_str = format_value(with_d4_corrected_mad, None)
            else:
                without_d4_str = format_value(
                    data.get("without_d4", {}).get("mad"), best_mad
                )
                without_d4_msd_str = format_msd(data.get("without_d4", {}).get("msd"))
                without_d4_r2_str = format_value(
                    data.get("without_d4", {}).get("r_squared"), best_r2, True
                )
                with_d4_str = format_value(data.get("with_d4", {}).get("mad"), best_mad)
                with_d4_msd_str = format_msd(data.get("with_d4", {}).get("msd"))
                with_d4_r2_str = format_value(
                    data.get("with_d4", {}).get("r_squared"), best_r2, True
                )

            if has_linear_fit:
                if is_first_in_category and category_counts.get(category, 0) == 1:
                    f.write(
                        f" & \\raisebox{{-0.5\\height}}{{{func_display}}} & \\raisebox{{-0.5\\height}}{{{without_d4_grad_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_off_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_corrected_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_grad_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_off_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_corrected_str}}} \\\\[2ex]\n"
                    )
                else:
                    f.write(
                        f" & {func_display} & {without_d4_grad_str} & {without_d4_off_str} & {without_d4_corrected_str} & {with_d4_grad_str} & {with_d4_off_str} & {with_d4_corrected_str} \\\\\n"
                    )
            else:
                if is_first_in_category and category_counts.get(category, 0) == 1:
                    f.write(
                        f" & \\raisebox{{-0.5\\height}}{{{func_display}}} & \\raisebox{{-0.5\\height}}{{{without_d4_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_msd_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_r2_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_msd_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_r2_str}}} \\\\[2ex]\n"
                    )
                else:
                    f.write(
                        f" & {func_display} & {without_d4_str} & {without_d4_msd_str} & {without_d4_r2_str} & {with_d4_str} & {with_d4_msd_str} & {with_d4_r2_str} \\\\\n"
                    )

        f.write("\\bottomrule\n\\end{tabular}\n")
        if has_linear_fit:
            caption = "Mean Absolute Deviation (MAD) and coefficient of determination ($r^2$) of isomerization energies for COMPAS-3x geometries relative to revDSD-PBEP86-D4(noFC)/def2-QZVPP. All DFT calculations performed with (99, 590) grid and def2-TZVP basis set. Linear fit correction has been applied, which sets the mean signed deviation (MSD) to zero. GFN2-xTB results are from semiempirical calculations."
        else:
            caption = "Mean Absolute Deviation (MAD), Mean Signed Deviation (MSD), and coefficient of determination ($r^2$) of isomerization energies for COMPAS-3x geometries relative to revDSD-PBEP86-D4(noFC)/def2-QZVPP. All DFT calculations performed with (99, 590) grid and def2-TZVP basis set. GFN2-xTB results are from semiempirical calculations."
        f.write(f"\\caption{{{caption_prefix}{caption}}}\n")
        f.write(f"\\label{{{label}}}\n\\end{{table}}\n")


def generate_latex_tables_by_maxz(
    results: list[dict],
    exess_df: pd.DataFrame,
    output_path: Path,
    threshold: float = 1.0,
) -> None:
    """Write two MAD/MSD/r² tables split by max_z_displacement threshold."""
    lt_ids, ge_ids = _maxz_common_ids(exess_df, threshold)
    lt_results = [r2 for r in results if (r2 := _subset_result(r, lt_ids)) is not None]
    ge_results = [r2 for r in results if (r2 := _subset_result(r, ge_ids)) is not None]
    thr_s = f"{threshold:.1f}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated: split by max_z_displacement\n")
        f.write(f"% Threshold: {threshold:.2f} Å\n\n")
        f.write("% ---- max_z_displacement < threshold ----\n")
        generate_latex_table(
            lt_results,
            f,
            caption_prefix=rf"(max $z$ displacement $< {thr_s}$~\AA) ",
            label="tab:compas3x_benchmarks_maxz_lt",
        )
        f.write("\n\n% ---- max_z_displacement >= threshold ----\n")
        generate_latex_table(
            ge_results,
            f,
            caption_prefix=rf"(max $z$ displacement $\ge {thr_s}$~\AA) ",
            label="tab:compas3x_benchmarks_maxz_ge",
        )


def generate_latex_table_maxz_combined(
    results: list[dict],
    exess_df: pd.DataFrame,
    output_path: Path,
    threshold: float = 1.0,
) -> None:
    """One table: planar vs non-planar column groups × without/with D4 (MAD/MSD/r²)."""
    lt_ids, ge_ids = _maxz_common_ids(exess_df, threshold)
    func_data: dict = {}
    xtb_lt: dict | None = None
    xtb_ge: dict | None = None

    for result in results:
        func = result["functional"]
        if func == "GFN2-xTB":
            rlt = _subset_result(result, lt_ids)
            rge = _subset_result(result, ge_ids)
            if rlt is not None:
                xtb_lt = {
                    "mad": rlt["mad_kjmol"],
                    "msd": rlt["msd_kjmol"],
                    "r_squared": rlt["r_squared"],
                }
            if rge is not None:
                xtb_ge = {
                    "mad": rge["mad_kjmol"],
                    "msd": rge["msd_kjmol"],
                    "r_squared": rge["r_squared"],
                }
            continue
        rlt = _subset_result(result, lt_ids)
        rge = _subset_result(result, ge_ids)
        if rlt is None or rge is None:
            continue
        key = "with_d4" if result["include_d4"] else "without_d4"
        if func not in func_data:
            func_data[func] = {"lt": {}, "ge": {}}
        func_data[func]["lt"][key] = {
            "mad": rlt["mad_kjmol"],
            "msd": rlt["msd_kjmol"],
            "r_squared": rlt["r_squared"],
        }
        func_data[func]["ge"][key] = {
            "mad": rge["mad_kjmol"],
            "msd": rge["msd_kjmol"],
            "r_squared": rge["r_squared"],
        }

    all_mads: list[float] = []
    all_r2s: list[float] = []
    for d in func_data.values():
        for regime in ("lt", "ge"):
            for k in d[regime]:
                all_mads.append(d[regime][k]["mad"])
                all_r2s.append(d[regime][k]["r_squared"])
    if xtb_lt is not None:
        all_mads.append(xtb_lt["mad"])
        all_r2s.append(xtb_lt["r_squared"])
    if xtb_ge is not None:
        all_mads.append(xtb_ge["mad"])
        all_r2s.append(xtb_ge["r_squared"])
    best_mad = min(all_mads) if all_mads else None
    best_r2 = max(all_r2s) if all_r2s else None

    lda_functionals = ["SVWN5"]
    gga_functionals = ["PBE", "BLYP", "revPBE", "BP86", "BPW91", "B97-D", "HCTH407"]
    mgga_functionals = [
        "TPSS",
        "MN15L",
        "SCAN",
        "rSCAN",
        "r2SCAN",
        "revTPSS",
        "t-HCTH",
        "M06-L",
        "M11-L",
    ]

    def get_mad_for_ordering(fn: str) -> float:
        data = func_data[fn]
        if fn in lda_functionals:
            m = data["lt"].get("without_d4", {}).get("mad")
            return m if m is not None else float("inf")
        return (
            data["lt"].get("with_d4", {}).get("mad")
            or data["lt"].get("without_d4", {}).get("mad")
            or float("inf")
        )

    functional_order: list[str] = []
    for category in (lda_functionals, gga_functionals, mgga_functionals):
        functional_order.extend(
            sorted(
                [f for f in category if f in func_data],
                key=get_mad_for_ordering,
                reverse=True,
            )
        )

    def _stat(fd: dict, regime: str, d4k: str, field: str):
        block = fd.get(regime, {}).get(d4k, {})
        return block.get(field) if block else None

    def format_value(val, best_val, is_r2=False):
        if val is None:
            return "---"
        fmt = f"{val:.3f}" if is_r2 else f"{val:.2f}"
        tol = 0.001 if is_r2 else 0.01
        if best_val is not None and abs(val - best_val) < tol:
            return fmt
        return fmt

    def format_msd(msd):
        if msd is None:
            return "---"
        sign = "$-$" if msd < 0 else "$+$"
        return f"{sign}{abs(msd):.2f}"

    thr_s = f"{threshold:.1f}"
    n_lt = len(lt_ids)
    n_ge = len(ge_ids)

    def _maxz_category_col(category: str, remaining: int) -> str:
        rot = f"\\rotatebox{{90}}{{\\emph{{{category}}}}}"
        if remaining == 1:
            inner = f"\\raisebox{{-0.5\\height}}{{{rot}}}"
            return f"\\multirow{{{remaining}}}{{*}}{{{inner}}}"
        return f"\\multirow{{{remaining}}}{{*}}{{{rot}}}"

    def _maxz_rcell(s: str) -> str:
        return f"\\raisebox{{-0.5\\height}}{{{s}}}"

    _mad_h = "\\shortstack{MAD\\\\\\footnotesize(kJ/mol)}"
    _msd_h = "\\shortstack{MSD\\\\\\footnotesize(kJ/mol)}"
    _r2_h = "\\parbox[t]{2em}{$r^2$\\vspace{0.5em}}"
    _metric_row = (
        f" & & {_mad_h} & {_msd_h} & {_r2_h} & {_mad_h} & {_msd_h} & {_r2_h} & "
        f"{_mad_h} & {_msd_h} & {_r2_h} & {_mad_h} & {_msd_h} & {_r2_h} \\\\\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated: combined max_z_displacement split\n")
        f.write(f"% Threshold: {threshold:.2f} Å\n")
        f.write("% Requires: \\usepackage{booktabs, multirow, rotating, graphicx}\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(
            "\\begin{tabular}{@{}c@{\\hspace{0.8em}}l@{\\hspace{0.25em}}"
            "c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}"
            "c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}"
            "c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}"
            "c@{\\hspace{0.22em}}c@{\\hspace{0.22em}}c@{}}\n"
        )
        f.write("\\toprule\n")
        f.write(
            " & \\multirow{3}{*}{\\raisebox{-0.5\\height}{Functional}} & "
            f"\\multicolumn{{6}}{{c}}{{Planar (max $z$ $< {thr_s}$~\\AA)}} & "
        )
        f.write(
            f"\\multicolumn{{6}}{{c}}{{Non-planar (max $z$ $\\ge {thr_s}$~\\AA)}} \\\\\n"
        )
        f.write("\\cmidrule(lr){3-8} \\cmidrule(lr){9-14}\n")
        f.write(
            " & & \\multicolumn{3}{c}{Without D4} & "
            "\\multicolumn{3}{c}{With D4} & "
            "\\multicolumn{3}{c}{Without D4} & "
            "\\multicolumn{3}{c}{With D4} \\\\\n"
        )
        f.write(
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} "
            "\\cmidrule(lr){9-11} \\cmidrule(lr){12-14}\n"
        )
        f.write(_metric_row)
        f.write("\\midrule\n")

        if xtb_lt is not None or xtb_ge is not None:

            def xtb_cells(xtb: dict | None):
                if xtb is None:
                    return "--- & --- & ---"
                return (
                    f"{format_value(xtb['mad'], best_mad)} & "
                    f"{format_msd(xtb['msd'])} & "
                    f"{format_value(xtb['r_squared'], best_r2, True)}"
                )

            f.write(
                f" & GFN2--xTB & --- & --- & --- & {xtb_cells(xtb_lt)} & "
                f"--- & --- & --- & {xtb_cells(xtb_ge)} \\\\\n"
            )
            f.write("\\midrule\n")

        current_category = None
        category_counts = {
            "LDA": sum(1 for fn in lda_functionals if fn in func_data),
            "GGA": sum(1 for fn in gga_functionals if fn in func_data),
            "MGGA": sum(1 for fn in mgga_functionals if fn in func_data),
        }

        for func in functional_order:
            if func not in func_data:
                continue
            if func in lda_functionals:
                category = "LDA"
            elif func in gga_functionals:
                category = "GGA"
            elif func in mgga_functionals:
                category = "MGGA"
            else:
                category = None
            is_first_in_category = category != current_category
            if is_first_in_category:
                if current_category is not None:
                    f.write("\\midrule\n")
                current_category = category
                remaining = category_counts.get(category, 0)
                f.write(_maxz_category_col(category, remaining))

            func_display = (
                r"$\tau$--HCTH" if func == "t-HCTH" else func.replace("-", "--")
            )
            d = func_data[func]
            parts = []
            for regime in ("lt", "ge"):
                for d4k in ("without_d4", "with_d4"):
                    parts.append(format_value(_stat(d, regime, d4k, "mad"), best_mad))
                    parts.append(format_msd(_stat(d, regime, d4k, "msd")))
                    parts.append(
                        format_value(_stat(d, regime, d4k, "r_squared"), best_r2, True)
                    )
            row_body = " & ".join(parts)

            if is_first_in_category and category_counts.get(category, 0) == 1:
                cells = row_body.split(" & ")
                raised = " & ".join(_maxz_rcell(c) for c in cells)
                f.write(f" & {_maxz_rcell(func_display)} & {raised} \\\\[2ex]\n")
            else:
                f.write(f" & {func_display} & {row_body} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}%\n}\n")
        cap = (
            "Mean Absolute Deviation (MAD), Mean Signed Deviation (MSD), and coefficient "
            "of determination ($r^2$) of isomerization energies for COMPAS-3x geometries "
            "relative to revDSD-PBEP86-D4(noFC)/def2-QZVPP, using the revDSD minimum-energy "
            "isomer as reference without linear correction. "
            f"Statistics are split into planar (max $z$ $< {thr_s}$~\\AA, {n_lt} geometries) "
            f"and non-planar (max $z$ $\\ge {thr_s}$~\\AA, {n_ge} geometries), based on max "
            "$z$ from GFN2-xTB-optimized structures. "
            "All DFT calculations used (99,590) grid and def2-TZVP."
        )
        f.write(f"\\caption{{{cap}}}\n")
        f.write("\\label{tab:compas3x_benchmarks_maxz_combined}\n\\end{table}\n")


def save_results_to_csv(results, output_path):
    """Save all benchmark results to a CSV file.

    Args:
        results: List of result dictionaries from process_functional_comparison or process_xtb_comparison
        output_path: Path to save the CSV file
    """
    rows = []
    for result in results:
        row = {
            "functional": result["functional"],
            "include_d4": (
                "Yes"
                if result["include_d4"]
                else "No" if result["include_d4"] is False else "N/A"
            ),
            "n_structures": result["n_structures"],
            "mad_kjmol": result["mad_kjmol"],
            "msd_kjmol": result["msd_kjmol"],
            "rmsd_kjmol": result["rmsd"],
            "r_squared": result["r_squared"],
            "mad_percentage": result["mad_percentage"],
            "n_negative_isomer_energies": result["n_negative_isomer_energies"],
            "n_overestimated": result["n_overestimated"],
            "n_underestimated": result["n_underestimated"],
            "n_overestimates_over_100": result["n_overestimates_over_100"],
            "max_deviation_kjmol": result["max_deviation"],
            "min_deviation_kjmol": result["min_deviation"],
            "gradient": result.get("gradient"),
            "offset_kjmol": result.get("offset"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark functionals against revDSD-PBEP86-D4(noFC) for COMPAS-3x"
    )
    parser.add_argument(
        "--exess-csv", default="analysis/exess_data.csv", help="EXESS data CSV file"
    )
    parser.add_argument(
        "--output", default="compas3x_benchmarks.txt", help="Output text file"
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Output directory for plots"
    )
    parser.add_argument(
        "--reference-method",
        choices=["min", "avg", "linear_fit"],
        default="min",
        help="Method for calculating isomerization energies: 'min', 'avg', or 'linear_fit' (applies linear correction)",
    )
    parser.add_argument(
        "--write-maxz-split-tables",
        action="store_true",
        help="Also write two LaTeX tables split by max_z_displacement (below / above threshold).",
    )
    parser.add_argument(
        "--write-maxz-combined-table",
        action="store_true",
        help="Also write one LaTeX table: planar vs non-planar × without/with D4.",
    )
    parser.add_argument(
        "--maxz-threshold",
        type=float,
        default=1.0,
        help="Å threshold for max_z_displacement split (default: 1.0).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    csv_path = (
        script_dir.parent / args.exess_csv
        if not Path(args.exess_csv).is_absolute()
        else Path(args.exess_csv)
    )
    output_path = (
        script_dir / args.output
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    output_dir = (
        script_dir / args.output_dir
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: EXESS data file not found at {csv_path}")
        return

    print(f"Loading EXESS data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")

    # Check if required columns exist
    if "functional" not in df.columns:
        print("\nERROR: The CSV file is missing the 'functional' column.")
        print(
            "Please regenerate the CSV file by running: python analysis/extract_exess_data.py"
        )
        return

    functionals_to_test = [
        "SVWN5",
        "PBE",
        "BLYP",
        "revPBE",
        "BP86",
        "BPW91",
        "B97-D",
        "HCTH407",
        "TPSS",
        "MN15L",
        "SCAN",
        "rSCAN",
        "r2SCAN",
        "revTPSS",
        "t-HCTH",
        "M06-L",
        "M11-L",
    ]
    all_results = []

    # MGGA functionals without D4 support
    mgga_no_d4 = ["M11-L", "MN15L", "t-HCTH"]

    # First pass: collect all results without creating plots to calculate global limits
    print("Collecting data for all functionals...")
    for functional in functionals_to_test:
        if functional == "SVWN5" or functional == "HCTH407" or functional in mgga_no_d4:
            # These functionals don't have D4 support, so only test without D4
            result = process_functional_comparison(
                df,
                functional,
                include_d4=False,
                output_dir=None,
                reference_method=args.reference_method,
            )
            if result:
                all_results.append(result)
        else:
            # For other functionals, test both with and without D4
            for include_d4 in [True, False]:
                result = process_functional_comparison(
                    df,
                    functional,
                    include_d4=include_d4,
                    output_dir=None,
                    reference_method=args.reference_method,
                )
                if result:
                    all_results.append(result)

    # Process GFN2-xTB comparison
    xtb_result = process_xtb_comparison(
        df, output_dir=None, reference_method=args.reference_method
    )
    if xtb_result:
        all_results.append(xtb_result)

    # Calculate global limits from all collected data
    if len(all_results) > 0:
        all_ref_energies = []
        all_func_energies = []
        for result in all_results:
            all_ref_energies.extend(result["ref_energies"].values)
            all_func_energies.extend(result["func_energies"].values)

        if all_ref_energies and all_func_energies:
            global_min = min(min(all_ref_energies), min(all_func_energies))
            global_max = max(max(all_ref_energies), max(all_func_energies))
            # Add small padding
            padding = (global_max - global_min) * 0.05
            global_min -= padding
            global_max += padding
            # Ensure square aspect ratio
            max_range = max(global_max - global_min, 1.0)
            center = (global_min + global_max) / 2
            plot_limits = (
                (center - max_range / 2, center + max_range / 2),
                (center - max_range / 2, center + max_range / 2),
            )
            print(f"\nGlobal plot limits: x={plot_limits[0]}, y={plot_limits[1]}")
        else:
            plot_limits = None
    else:
        plot_limits = None

    # Second pass: create plots with global limits
    if output_dir and plot_limits:
        print("\nCreating plots with global scale...")
        for result in all_results:
            functional = result["functional"]
            include_d4 = result.get("include_d4")

            if functional == "GFN2-xTB":
                # xTB doesn't create plots here, handled separately
                continue

            # Re-run just to create plots with limits (quiet mode to avoid duplicate output)
            process_functional_comparison(
                df,
                functional,
                include_d4=include_d4,
                output_dir=output_dir,
                reference_method=args.reference_method,
                plot_limits=plot_limits,
                quiet=True,
            )

    if len(all_results) == 0:
        print("\nNo results to save.")
        return

    # Save results to file
    with open(output_path, "w") as f:
        f.write("COMPAS-3x Benchmark: Functionals vs revDSD-PBEP86-D4(noFC)\n")
        f.write("=" * 60 + "\n\n")
        for result in all_results:
            if result["functional"] == "GFN2-xTB":
                f.write(f"\n{result['functional']}\n")
            else:
                d4_label = "with D4" if result["include_d4"] else "without D4"
                f.write(f"\n{result['functional']} ({d4_label})\n")
            f.write("-" * 60 + "\n")
            f.write(f"Number of matched structures: {result['n_structures']}\n\n")
            f.write("Statistics:\n")
            f.write(f"  MAD: {result['mad_kjmol']:.3f} kJ/mol\n")
            f.write(f"  MSD: {result['msd_kjmol']:.3f} kJ/mol\n")
            f.write(f"  RMSD: {result['rmsd']:.3f} kJ/mol\n")
            f.write(f"  r²: {result['r_squared']:.4f}\n")
            f.write(f"  MAD as percentage: {result['mad_percentage']:.2f}%\n\n")
            top_indices = result["deviations"].abs().nlargest(10).index
            f.write("Top 10 largest deviations:\n")
            for idx in top_indices:
                f.write(
                    f"  {result['merged'].loc[idx, 'common_id']}: {result['deviations'].loc[idx]:.3f} kJ/mol\n"
                )
            f.write("\n")

    # Generate LaTeX table
    latex_path = output_path.with_suffix(".tex")
    generate_latex_table(all_results, latex_path)

    if args.write_maxz_split_tables:
        if args.reference_method != "min":
            print(
                "NOTE: --write-maxz-split-tables is intended for --reference-method min."
            )
        stem = output_path.with_suffix("").name
        split_path = output_path.with_suffix("").with_name(f"{stem}_maxz_split.tex")
        try:
            generate_latex_tables_by_maxz(
                all_results, df, split_path, threshold=float(args.maxz_threshold)
            )
            print(f"Max-z split tables saved to {split_path}")
        except Exception as e:
            print(f"WARNING: Failed to generate max-z split tables: {e}")

    if args.write_maxz_combined_table:
        if args.reference_method != "min":
            print(
                "NOTE: --write-maxz-combined-table is intended for --reference-method min."
            )
        combined_path = output_path.with_name(output_path.stem + "_maxz_combined.tex")
        try:
            generate_latex_table_maxz_combined(
                all_results, df, combined_path, threshold=float(args.maxz_threshold)
            )
            print(f"Max-z combined table saved to {combined_path}")
        except Exception as e:
            print(f"WARNING: Failed to generate max-z combined table: {e}")

    # Save results to CSV
    csv_path = output_path.with_suffix(".csv")
    save_results_to_csv(all_results, csv_path)

    print(f"\nResults saved to {output_path}")
    print(f"LaTeX table saved to {latex_path}")
    print(f"CSV results saved to {csv_path}")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
