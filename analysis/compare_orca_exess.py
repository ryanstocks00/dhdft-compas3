#!/usr/bin/env python3

"""Compare ORCA and EXESS results for C24H14 subset."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

HARTREE_TO_KJ_PER_MOL = 2625.5


def extract_common_id(name):
    parts = name.split("_", 1)
    if len(parts) > 1:
        id = parts[1]
        return id[3:] if id.startswith("hc_") else id
    return name


def compute_mad_orca_vs_reference(orca_df, reference_basis="qz_ri"):
    """Compare ORCA basis set combinations for C24H14 molecules only."""
    c24h14 = orca_df[(orca_df["num_carbons"] == 24) & (orca_df["num_hydrogens"] == 14)]
    print(f"Filtering ORCA data: {len(orca_df)} total rows -> {len(c24h14)} C24H14 rows")
    ref = c24h14[c24h14["basis_combo_id"] == reference_basis].set_index("isomer")
    
    results = []
    for _, row in c24h14[["basis_combo_id", "primary_basis", "scf_aux_basis", "ri_aux_basis"]].drop_duplicates().iterrows():
        basis_id = row["basis_combo_id"]
        data = c24h14[(c24h14["basis_combo_id"] == basis_id) & (c24h14["isomer"].isin(ref.index))]
        
        if basis_id == reference_basis:
            mad_abs = mad_rel = 0.0
        else:
            abs_dev = data["total_energy_hartree"].values - ref.loc[data["isomer"], "total_energy_hartree"].values
            rel_dev = data["relative_energy_hartree"].values - ref.loc[data["isomer"], "relative_energy_hartree"].values
            mad_abs = np.mean(np.abs(abs_dev)) * HARTREE_TO_KJ_PER_MOL
            mad_rel = np.mean(np.abs(rel_dev)) * HARTREE_TO_KJ_PER_MOL

        results.append({
            "primary_basis": row["primary_basis"],
            "scf_aux_basis": row["scf_aux_basis"] if pd.notna(row["scf_aux_basis"]) else "NA",
            "ri_aux_basis": row["ri_aux_basis"] if pd.notna(row["ri_aux_basis"]) else "NA",
            "basis_combo_id": basis_id,
            "mad_abs_kjmol": mad_abs,
            "mad_rel_kjmol": mad_rel,
        })
    return pd.DataFrame(results)


def compute_mad_exess_vs_orca_reference(exess_df, orca_df, reference_basis="qz_riri"):
    """Compare EXESS results to ORCA reference for C24H14 molecules only.
    
    EXESS data is filtered to:
    - C24H14 molecules only
    - def2-QZVPP basis set
    - revDSD-PBEP86-D4 functional
    """
    # Filter to C24H14 molecules only
    exess = exess_df[(exess_df["n_carbons"] == 24) & (exess_df["n_hydrogens"] == 14)].copy()
    orca = orca_df[(orca_df["num_carbons"] == 24) & (orca_df["num_hydrogens"] == 14)]
    print(f"Filtering to C24H14 molecules: EXESS {len(exess_df)} -> {len(exess)} rows, ORCA {len(orca_df)} -> {len(orca)} rows")
    
    # Filter ORCA to reference basis set
    orca_before_basis = len(orca)
    ref = orca[orca["basis_combo_id"] == reference_basis].copy()
    print(f"Filtering ORCA to reference basis '{reference_basis}': {orca_before_basis} -> {len(ref)} rows")
    
    # Debug: Check how many unique molecules we have in ORCA C24H14 data
    if "isomer" in orca.columns:
        orca_unique_isomers = orca["isomer"].nunique()
        print(f"ORCA C24H14 unique isomers (all basis sets): {orca_unique_isomers}")
        if "isomer" in ref.columns:
            ref_unique_isomers = ref["isomer"].nunique()
            print(f"ORCA reference unique isomers (basis '{reference_basis}'): {ref_unique_isomers}")
            if ref_unique_isomers < 26:
                print(f"WARNING: Expected 26 C24H14 molecules in ORCA reference, but found only {ref_unique_isomers}")
                print(f"Available isomers in ORCA reference: {sorted(ref['isomer'].unique())}")
    
    # Check what basis sets are available in ORCA data
    if len(ref) == 0:
        print(f"Error: No ORCA data found with basis_combo_id == '{reference_basis}'")
        print(f"Available basis_combo_id values in ORCA C24H14 data: {orca['basis_combo_id'].unique() if 'basis_combo_id' in orca.columns else 'N/A'}")
        return None

    # Filter EXESS to def2-QZVPP basis set
    exess_before = len(exess)
    if "basis_set" not in exess.columns:
        print(f"Error: 'basis_set' column not found in EXESS data. Available columns: {exess.columns.tolist()}")
        return None
    
    exess = exess[exess["basis_set"] == "def2-QZVPP"].copy()
    print(f"Filtered EXESS data by basis set: {exess_before} -> {len(exess)} rows (basis_set == 'def2-QZVPP')")
    if len(exess) == 0:
        print(f"Error: No EXESS data found with basis_set == 'def2-QZVPP'")
        print(f"Available basis sets in EXESS data: {exess_df['basis_set'].unique() if 'basis_set' in exess_df.columns else 'N/A'}")
        return None
    
    # Filter EXESS to revDSD-PBEP86-D4 functional
    exess_before_func = len(exess)
    if "functional" not in exess.columns:
        print(f"Error: 'functional' column not found in EXESS data. Available columns: {exess.columns.tolist()}")
        return None
    
    # Handle different possible names for the functional
    target_functional = "revDSD-PBEP86-D4"
    exess = exess[exess["functional"] == target_functional].copy()
    print(f"Filtered EXESS data by functional: {exess_before_func} -> {len(exess)} rows (functional == '{target_functional}')")
    if len(exess) == 0:
        print(f"Error: No EXESS data found with functional == '{target_functional}'")
        print(f"Available functionals in EXESS data: {exess_df['functional'].unique() if 'functional' in exess_df.columns else 'N/A'}")
        return None

    exess["common_id"] = exess["isomer_name"].apply(extract_common_id)
    exess["prefix"] = exess["isomer_name"].str.split("_").str[0]
    
    # Filter to only COMPAS-3 data (exclude PAH335)
    exess_before_compas = len(exess)
    exess = exess[exess["prefix"].isin(["compas3x", "compas3D"])].copy()
    print(f"Filtered to COMPAS-3 only (excluding PAH335): {exess_before_compas} -> {len(exess)} rows")
    if len(exess) == 0:
        print(f"Error: No COMPAS-3 data found. Available prefixes: {exess_df['isomer_name'].str.split('_').str[0].unique() if 'isomer_name' in exess_df.columns else 'N/A'}")
        return None
    
    # Debug: Check for duplicates and multiple entries per molecule
    print(f"\nAfter all filters: {len(exess)} rows")
    if "optimizer" in exess.columns:
        print(f"  Unique optimizers: {exess['optimizer'].unique()}")
        print(f"  Rows per optimizer: {exess['optimizer'].value_counts().to_dict()}")
    unique_common_ids = exess["common_id"].nunique()
    print(f"  Unique common_id values: {unique_common_ids}")
    if unique_common_ids < len(exess):
        print(f"  WARNING: {len(exess) - unique_common_ids} duplicate common_id entries found")
        # Show duplicates
        duplicates = exess[exess.duplicated(subset=["common_id"], keep=False)]
        if len(duplicates) > 0:
            print(f"  Duplicate entries (showing first 10):")
            print(duplicates[["common_id", "isomer_name", "optimizer"]].head(10).to_string())
    
    # Check for duplicates: same common_id but different prefixes (COMPAS-3x vs COMPAS-3D)
    # We want to keep both COMPAS-3x and COMPAS-3D entries (26 total: 13 from each)
    prefix_counts = exess["prefix"].value_counts()
    print(f"  Rows per prefix: {prefix_counts.to_dict()}")
    
    # Deduplicate only if we have multiple entries with the same (common_id, prefix) combination
    # This handles cases where the same molecule appears multiple times in the same dataset
    duplicates_by_prefix = exess.duplicated(subset=["common_id", "prefix"], keep=False)
    if duplicates_by_prefix.sum() > 0:
        print(f"  Found {duplicates_by_prefix.sum()} duplicate entries with same (common_id, prefix)")
        # If optimizer column exists, prefer xTB optimizer for COMPAS-3x
        if "optimizer" in exess.columns:
            # Create a priority column: 0 for xTB/GFN2-xTB, 1 for others
            exess["_priority"] = exess["optimizer"].apply(lambda x: 0 if x in ["GFN2-xTB", "xTB"] else 1)
            # Sort by priority (xTB first), then by common_id and prefix
            exess = exess.sort_values(by=["common_id", "prefix", "_priority"])
            exess = exess.drop(columns=["_priority"])
        exess = exess.drop_duplicates(subset=["common_id", "prefix"], keep="first")
        print(f"  After deduplicating (common_id, prefix) pairs: {len(exess)} rows")
    
    # Expected: 26 rows (13 COMPAS-3x + 13 COMPAS-3D)
    if len(exess) != 26:
        print(f"  WARNING: Expected 26 C24H14 molecules (13 COMPAS-3x + 13 COMPAS-3D), but found {len(exess)}")
        print(f"  Unique common_ids: {sorted(exess['common_id'].unique())}")
        print(f"  Prefix distribution: {exess['prefix'].value_counts().to_dict()}")
    
    # Filter ORCA reference to only COMPAS-3 data (exclude PAH335)
    ref["common_id"] = ref["isomer"].apply(extract_common_id)
    ref["prefix"] = ref["isomer"].str.split("_").str[0]
    
    # Debug: Show sample ORCA isomer names to verify prefix extraction
    print(f"\nSample ORCA isomer names and extracted prefixes:")
    sample_isomers = ref[["isomer", "prefix", "common_id", "total_energy_hartree"]].head(10)
    print(sample_isomers.to_string())
    
    # Check if there are any isomers that don't start with compas3x or compas3D
    non_compas = ref[~ref["prefix"].isin(["compas3x", "compas3D"])]
    if len(non_compas) > 0:
        print(f"\nWARNING: Found {len(non_compas)} ORCA entries with unexpected prefixes:")
        print(non_compas[["isomer", "prefix"]].head(10).to_string())
    ref_before_compas = len(ref)
    
    # Debug: show prefix distribution before filtering
    if ref_before_compas > 0:
        print(f"ORCA reference prefix distribution before COMPAS-3 filter: {ref['prefix'].value_counts().to_dict()}")
        # Check if we have duplicate common_ids with different prefixes
        dup_common_ids = ref[ref.duplicated(subset=["common_id"], keep=False)]
        if len(dup_common_ids) > 0:
            print(f"ORCA reference has {len(dup_common_ids)} entries with duplicate common_ids (different prefixes):")
            sample = dup_common_ids[["isomer", "common_id", "prefix", "total_energy_hartree"]].head(6)
            print(sample.to_string())
        else:
            print(f"WARNING: ORCA reference has NO duplicate common_ids - each common_id appears only once!")
            print(f"This means compas3x and compas3D share the same ORCA calculations.")
            # Show sample to verify
            sample = ref[["isomer", "common_id", "prefix", "total_energy_hartree"]].head(10)
            print("Sample ORCA reference data:")
            print(sample.to_string())
    
    ref = ref[ref["prefix"].isin(["compas3x", "compas3D"])].copy()
    print(f"Filtered ORCA reference to COMPAS-3 only: {ref_before_compas} -> {len(ref)} rows")
    
    if len(ref) == 0:
        print(f"Error: No ORCA reference data found for COMPAS-3 (compas3x or compas3D)")
        print(f"Available prefixes in ORCA reference: {ref['prefix'].unique() if len(ref) > 0 else 'N/A'}")
        return None
    
    # Show prefix distribution after filtering
    print(f"ORCA reference prefix distribution after COMPAS-3 filter: {ref['prefix'].value_counts().to_dict()}")
    print(f"ORCA reference unique common_ids: {ref['common_id'].nunique()}")
    
    # Check if common_ids are duplicated across prefixes
    common_id_counts = ref["common_id"].value_counts()
    duplicated_common_ids = common_id_counts[common_id_counts > 1]
    if len(duplicated_common_ids) > 0:
        print(f"ORCA reference has {len(duplicated_common_ids)} common_ids that appear in both compas3x and compas3D")
        # Show an example of a duplicated common_id to verify they have different energies
        example_common_id = duplicated_common_ids.index[0]
        example_rows = ref[ref["common_id"] == example_common_id][["isomer", "prefix", "common_id", "total_energy_hartree"]]
        print(f"Example duplicated common_id '{example_common_id}':")
        print(example_rows.to_string())
    else:
        print(f"WARNING: ORCA reference has NO common_ids that appear in both prefixes!")
        print(f"This means ORCA data only has one entry per common_id, not separate entries for compas3x and compas3D")
        # Show what prefixes we actually have
        print(f"ORCA reference prefix breakdown:")
        for prefix_val in ref["prefix"].unique():
            prefix_data = ref[ref["prefix"] == prefix_val]
            print(f"  {prefix_val}: {len(prefix_data)} entries, common_ids: {sorted(prefix_data['common_id'].unique())[:5]}")

    merged_list = []
    for opt in exess["optimizer"].unique():
        exess_opt = exess[exess["optimizer"] == opt].copy()
        # Map optimizer to prefix: xTB/GFN2-xTB -> compas3x, DFT/CAM-B3LYP-D3BJ -> compas3D
        if opt in ["xTB", "GFN2-xTB"]:
            prefix = "compas3x"
        elif opt in ["DFT", "CAM-B3LYP-D3BJ"]:
            prefix = "compas3D"
        else:
            print(f"WARNING: Unknown optimizer '{opt}', defaulting to compas3D")
            prefix = "compas3D"
        
        # Ensure EXESS prefix matches what we expect
        exess_opt["prefix"] = prefix
        ref_opt = ref[ref["prefix"] == prefix].copy()
        
        # Debug: check if we have matching data
        print(f"\nMerging {prefix} data:")
        print(f"  EXESS entries: {len(exess_opt)}")
        print(f"  ORCA reference entries (filtered by prefix '{prefix}'): {len(ref_opt)}")
        
        # If ORCA doesn't have data for this prefix, check if it has data for the other prefix
        # This would indicate ORCA only has one set of calculations (likely compas3x)
        if len(ref_opt) == 0:
            other_prefix = "compas3D" if prefix == "compas3x" else "compas3x"
            ref_other = ref[ref["prefix"] == other_prefix].copy()
            if len(ref_other) > 0:
                print(f"  WARNING: No ORCA data for prefix '{prefix}', but found {len(ref_other)} entries for '{other_prefix}'")
                print(f"  This suggests ORCA only calculated one set of geometries (likely {other_prefix})")
                print(f"  Using ORCA data from '{other_prefix}' for '{prefix}' comparison")
                ref_opt = ref_other.copy()
        
        if len(ref_opt) > 0:
            print(f"  ORCA common_ids: {sorted(ref_opt['common_id'].unique())}")
            # Show a sample of ORCA energies to verify they're different
            sample = ref_opt[["common_id", "prefix", "total_energy_hartree"]].head(3)
            print(f"  Sample ORCA energies for {prefix}:")
            print(sample.to_string())
        if len(exess_opt) > 0:
            print(f"  EXESS common_ids: {sorted(exess_opt['common_id'].unique())}")
        
        # Merge on common_id only (prefix is already filtered, so we don't need it in the merge key)
        # But we need to ensure we're using the right prefix's data
        # Include prefix in ref_opt_merge to preserve which prefix the ORCA data came from
        ref_opt_merge = ref_opt[["common_id", "prefix", "total_energy_hartree", "scf_energy_hartree", "dft_exc_hartree", "mp2_corr_hartree"]].copy()
        # Rename prefix column to avoid conflict during merge
        ref_opt_merge = ref_opt_merge.rename(columns={"prefix": "orca_prefix"})
        
        merged = exess_opt.merge(
            ref_opt_merge,
            on="common_id", suffixes=("_exess", "_orca")
        )
        
        # Verify that the ORCA prefix matches what we expect
        if len(merged) > 0:
            mismatched_prefix = merged[merged["orca_prefix"] != prefix]
            if len(mismatched_prefix) > 0:
                print(f"  ERROR: Found {len(mismatched_prefix)} entries where ORCA prefix doesn't match expected '{prefix}'")
                print(mismatched_prefix[["common_id", "prefix", "orca_prefix"]].head().to_string())
            # Update the prefix column to use the ORCA prefix to ensure consistency
            merged["prefix"] = merged["orca_prefix"]
            merged = merged.drop(columns=["orca_prefix"])
        
        # Verify the merge worked correctly - check that energies are different
        if len(merged) > 0:
            print(f"  Merged entries: {len(merged)}")
            # Show sample merged data to verify ORCA energies are correct
            sample_merged = merged[["common_id", "prefix", "total_energy_hartree_exess", "total_energy_hartree_orca"]].head(3)
            print(f"  Sample merged data for {prefix}:")
            print(sample_merged.to_string())
        else:
            print(f"  WARNING: No entries merged for {prefix}!")
            print(f"    EXESS common_ids: {sorted(exess_opt['common_id'].unique())}")
            print(f"    ORCA common_ids: {sorted(ref_opt['common_id'].unique())}")
        
        merged_list.append(merged)

    merged = pd.concat(merged_list, ignore_index=True)
    
    if len(merged) == 0:
        print("Error: No matching structures found between EXESS and ORCA data")
        return None
    
    # Debug: Verify that compas3x and compas3D have different ORCA energies
    if "prefix" in merged.columns and "total_energy_hartree_orca" in merged.columns:
        print(f"\nVerifying ORCA energies are different for compas3x vs compas3D:")
        print(f"Total merged rows: {len(merged)}")
        print(f"Prefix distribution in merged data: {merged['prefix'].value_counts().to_dict()}")
        
        for common_id in merged["common_id"].unique()[:3]:  # Check first 3 common_ids
            subset = merged[merged["common_id"] == common_id]
            print(f"\n  Checking {common_id}: {len(subset)} rows")
            print(f"    Prefixes in subset: {subset['prefix'].unique()}")
            if len(subset) == 2:  # Should have both compas3x and compas3D
                compas3x_row = subset[subset["prefix"] == "compas3x"]
                compas3d_row = subset[subset["prefix"] == "compas3D"]
                if len(compas3x_row) > 0 and len(compas3d_row) > 0:
                    orca_x = compas3x_row.iloc[0]["total_energy_hartree_orca"]
                    orca_d = compas3d_row.iloc[0]["total_energy_hartree_orca"]
                    print(f"  {common_id}: compas3x ORCA={orca_x:.10f}, compas3D ORCA={orca_d:.10f}, diff={abs(orca_x-orca_d):.2e}")
                    if abs(orca_x - orca_d) < 1e-10:
                        print(f"    WARNING: ORCA energies are identical for {common_id}!")
                        # Show the full rows to debug
                        print(f"    compas3x row:")
                        print(compas3x_row[["common_id", "prefix", "total_energy_hartree_orca"]].to_string())
                        print(f"    compas3D row:")
                        print(compas3d_row[["common_id", "prefix", "total_energy_hartree_orca"]].to_string())
                else:
                    print(f"    WARNING: Missing one of the prefixes for {common_id}")
                    print(f"      compas3x rows: {len(compas3x_row)}, compas3D rows: {len(compas3d_row)}")
            else:
                print(f"    WARNING: Expected 2 rows (one for each prefix) but found {len(subset)}")
                print(f"      Rows: {subset[['common_id', 'prefix', 'total_energy_hartree_orca']].to_string()}")
    
    # Absolute deviations
    abs_dev = merged["total_energy_hartree_exess"].values - merged["total_energy_hartree_orca"].values
    
    # Relative energies: compute relative to minimum for each method separately
    # This is the standard way to compute isomerization energies
    exess_min = merged["total_energy_hartree_exess"].min()
    orca_min = merged["total_energy_hartree_orca"].min()
    
    exess_rel = merged["total_energy_hartree_exess"].values - exess_min
    orca_rel = merged["total_energy_hartree_orca"].values - orca_min
    
    # Relative deviation: difference in isomerization energies
    rel_dev = exess_rel - orca_rel
    scf_dev = merged["scf_energy_hartree_exess"].values - merged["scf_energy_hartree_orca"].values
    xc_dev = merged["xc_energy_hartree"].values - merged["dft_exc_hartree"].values
    mp2_dev = (merged["pt2_os_correction_hartree"] + merged["pt2_ss_correction_hartree"]).values - merged["mp2_corr_hartree"].values

    def mad(x): return np.mean(np.abs(x)) * HARTREE_TO_KJ_PER_MOL
    
    # Calculate EXESS MP2 total (OS + SS)
    exess_mp2_total = (merged["pt2_os_correction_hartree"].fillna(0) + 
                       merged["pt2_ss_correction_hartree"].fillna(0))
    
    # Add deviation columns to merged dataframe for CSV export
    merged["abs_deviation_hartree"] = abs_dev
    merged["abs_deviation_kjmol"] = abs_dev * HARTREE_TO_KJ_PER_MOL
    merged["rel_deviation_hartree"] = rel_dev
    merged["rel_deviation_kjmol"] = rel_dev * HARTREE_TO_KJ_PER_MOL
    merged["scf_deviation_hartree"] = scf_dev
    merged["scf_deviation_kjmol"] = scf_dev * HARTREE_TO_KJ_PER_MOL
    merged["xc_deviation_hartree"] = xc_dev
    merged["xc_deviation_kjmol"] = xc_dev * HARTREE_TO_KJ_PER_MOL
    merged["mp2_deviation_hartree"] = mp2_dev
    merged["mp2_deviation_kjmol"] = mp2_dev * HARTREE_TO_KJ_PER_MOL
    
    # Add relative energies
    merged["exess_rel_energy_hartree"] = exess_rel
    merged["orca_rel_energy_hartree"] = orca_rel
    
    # Add EXESS MP2 total for clarity
    merged["exess_mp2_total_hartree"] = exess_mp2_total
    
    return {
        "mad_abs_kjmol": mad(abs_dev),
        "mad_rel_kjmol": mad(rel_dev),
        "mad_scf_kjmol": mad(scf_dev),
        "mad_xc_kjmol": mad(xc_dev),
        "mad_mp2_kjmol": mad(mp2_dev),
        "merged_data": merged,
    }


def format_basis(basis):
    if pd.isna(basis) or basis == "NA":
        return "NA"
    basis_map = {
        "def2-SVP": "def2-SVP",
        "def2-TZVP": "def2-TZVP",
        "def2-QZVPP": "def2-QZVPP",
        "def2/JK": "def2-JKFIT",
        "def2-SVP/C": "def2-SVP-RIFIT",
        "def2-TZVP/C": "def2-TZVP-RIFIT",
        "def2-QZVPP/C": "def2-QZVPP-RIFIT",
    }
    return basis_map.get(basis, basis)


def generate_latex_table(orca_results):
    basis_order = {"def2-SVP": 1, "def2-TZVP": 2, "def2-QZVPP": 3}
    orca_results["basis_sort"] = orca_results["primary_basis"].map(basis_order)

    def get_scf_sort_key(row):
        scf = row["scf_aux_basis"]
        if scf == "NA":
            return 3
        if "JKFIT" in scf or scf == "def2/JK":
            return 1
        if "RIFIT" in scf or "/C" in scf:
            return 2
        return 4

    def get_ri_sort_key(row):
        ri = row["ri_aux_basis"]
        if ri == "NA":
            return 2
        print(row)
        if row["scf_aux_basis"] in ["def2-JKFIT", "def2/JK"]:
            return 1 if ("JKFIT" in ri or ri == "def2/JK") else 2
        if "RIFIT" in ri or "/C" in ri:
            return 1
        if row["scf_aux_basis"] == "NA":
            return 1 if ("RIFIT" in ri or "/C" in ri) else 2
        return 3

    orca_results["scf_sort"] = orca_results.apply(get_scf_sort_key, axis=1)
    orca_results["ri_sort"] = orca_results.apply(get_ri_sort_key, axis=1)
    orca_results = orca_results[orca_results["basis_combo_id"] != "qz_ri"].copy()
    orca_results = orca_results.sort_values(["basis_sort", "scf_sort", "ri_sort"]).reset_index(drop=True)
    orca_results = orca_results.drop(columns=["basis_sort", "scf_sort", "ri_sort"])

    lines = [
        "\\begin{table*}[]",
        "\\begin{tabular}{ccc",
        "                S[table-format=4.2]",
        "                S[table-format=2.3]}",
        "\\toprule",
        "\\textbf{\\begin{tabular}[c]{@{}c@{}}Primary\\\\ Basis\\end{tabular}} &",
        "\\textbf{\\begin{tabular}[c]{@{}c@{}}SCF\\\\ Auxiliary Basis\\end{tabular}} &",
        "\\textbf{\\begin{tabular}[c]{@{}c@{}}PT2\\\\ Auxiliary Basis\\end{tabular}} &",
        "\\textbf{\\begin{tabular}[c]{@{}c@{}}MAD Absolute\\\\ Energy (kJ/mol)\\end{tabular}} &",
        "\\textbf{\\begin{tabular}[c]{@{}c@{}}MAD Relative\\\\ Energy (kJ/mol)\\end{tabular}} \\\\ \\midrule",
    ]

    current_primary = current_scf = None
    for i, (_, row) in enumerate(orca_results.iterrows()):
        primary = format_basis(row["primary_basis"])
        scf = format_basis(row["scf_aux_basis"])
        ri = format_basis(row["ri_aux_basis"])
        mad_abs = row["mad_abs_kjmol"]
        mad_rel = row["mad_rel_kjmol"]
        is_qz_riri = row["basis_combo_id"] == "qz_riri"

        parts = []
        if primary != current_primary:
            n = len(orca_results[orca_results["primary_basis"] == row["primary_basis"]])
            parts.append(f"\\multirow{{{n}}}{{*}}{{{primary}}}" if n > 1 else primary)
            current_primary = primary
        else:
            parts.append("")

        if scf != current_scf or primary != current_primary:
            n = len(orca_results[
                (orca_results["primary_basis"] == row["primary_basis"])
                & (orca_results["scf_aux_basis"] == row["scf_aux_basis"])
            ])
            parts.append(f"\\multirow{{{n}}}{{*}}{{{scf}}}" if n > 1 else scf)
            current_scf = scf
        else:
            parts.append("")

        parts.append(ri)

        if row["primary_basis"] == "def2-SVP":
            fmt_abs = f"\\bfseries {mad_abs:.0f}" if is_qz_riri else f"{mad_abs:.0f}"
            fmt_rel = f"\\bfseries {mad_rel:.1f}" if is_qz_riri else f"{mad_rel:.1f}"
        elif row["primary_basis"] == "def2-TZVP":
            fmt_abs = f"\\bfseries {mad_abs:.0f}" if is_qz_riri else f"{mad_abs:.0f}"
            fmt_rel = f"\\bfseries {mad_rel:.1f}" if is_qz_riri else f"{mad_rel:.1f}"
        else:
            fmt_abs = f"\\bfseries {mad_abs:.2f}" if is_qz_riri else f"{mad_abs:.1f}"
            fmt_rel = f"\\bfseries {mad_rel:.3f}" if is_qz_riri else (f"{mad_rel:.3f}" if mad_rel < 0.1 else f"{mad_rel:.2f}")
        parts.extend([fmt_abs, fmt_rel])
        lines.append(" & ".join(parts) + " \\\\")

        if i < len(orca_results) - 1:
            next_row = orca_results.iloc[i + 1]
            if next_row["primary_basis"] != row["primary_basis"]:
                lines.append("\\cline{1-2}")
            elif next_row["scf_aux_basis"] != row["scf_aux_basis"]:
                lines.append("\\cline{2-2}")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Comparison of absolute and relative energies across basis set configurations for all 13 \\ce{C24H14} isomers in both the COMPAS-3x and COMPAS-3D datasets (26 geometries total). Energy deviations are reported relative to a def2-QZVPP calculation without RI-HF and with the def2-QZVPP-RIFIT basis for the PT2 step (direct PT2 without RI was prohibitively expensive for these systems). Relative energies were calculated with respect to the isomer that had the minimum energy using revDSD-PBEP86-D4 across all basis set combinations. All calculations were performed with the revDSD-PBEP86-D4(noFC) functional using ORCA v6.0.1.}\\label{tab:orca_mads}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orca-csv", default=Path(__file__).parent / "orca_data.csv")
    parser.add_argument("--exess-csv", default=Path(__file__).parent / "exess_data.csv")
    args = parser.parse_args()

    orca_df = pd.read_csv(args.orca_csv)
    exess_df = pd.read_csv(args.exess_csv)

    orca_results = compute_mad_orca_vs_reference(orca_df)
    exess_result = compute_mad_exess_vs_orca_reference(exess_df, orca_df)
    
    if exess_result is None:
        print("\nFailed to compute EXESS vs ORCA comparison. Check basis set filtering.")
        return
    
    print(f"EXESS MAD Absolute: {exess_result['mad_abs_kjmol']:.2f} kJ/mol")
    print(f"EXESS MAD Relative: {exess_result['mad_rel_kjmol']:.3f} kJ/mol")
    print(f"EXESS MAD SCF: {exess_result['mad_scf_kjmol']:.3f} kJ/mol")
    print(f"EXESS MAD XC: {exess_result['mad_xc_kjmol']:.3f} kJ/mol")
    print(f"EXESS MAD MP2: {exess_result['mad_mp2_kjmol']:.3f} kJ/mol")
    
    # Save detailed CSV with all molecule data
    if 'merged_data' in exess_result:
        merged = exess_result['merged_data']
        # Select and reorder columns for the CSV
        csv_columns = [
            'common_id', 'prefix', 'isomer_name',
            # Total energies
            'total_energy_hartree_exess', 'total_energy_hartree_orca',
            # SCF energies
            'scf_energy_hartree_exess', 'scf_energy_hartree_orca',
            # XC energies
            'xc_energy_hartree', 'dft_exc_hartree',
            # MP2 components
            'pt2_os_correction_hartree', 'pt2_ss_correction_hartree',
            'exess_mp2_total_hartree', 'mp2_corr_hartree',
            # Relative energies
            'exess_rel_energy_hartree', 'orca_rel_energy_hartree',
            # Absolute deviations
            'abs_deviation_hartree', 'abs_deviation_kjmol',
            # Relative deviations
            'rel_deviation_hartree', 'rel_deviation_kjmol',
            # Component deviations
            'scf_deviation_hartree', 'scf_deviation_kjmol',
            'xc_deviation_hartree', 'xc_deviation_kjmol',
            'mp2_deviation_hartree', 'mp2_deviation_kjmol',
        ]
        # Only include columns that exist
        available_columns = [col for col in csv_columns if col in merged.columns]
        detailed_df = merged[available_columns].copy()
        
        # Sort by prefix and common_id
        detailed_df = detailed_df.sort_values(by=['prefix', 'common_id'])
        
        csv_path = Path(__file__).parent / "orca_exess_comparison_detailed.csv"
        detailed_df.to_csv(csv_path, index=False)
        print(f"\nDetailed comparison CSV saved to: {csv_path}")
        print(f"  Contains {len(detailed_df)} molecules with all energy components and deviations")
    
    print("\n" + generate_latex_table(orca_results))


if __name__ == "__main__":
    main()
