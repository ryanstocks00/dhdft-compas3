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
    c24h14 = orca_df[(orca_df["num_carbons"] == 24) & (orca_df["num_hydrogens"] == 14)]
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
    exess = exess_df[(exess_df["n_carbons"] == 24) & (exess_df["n_hydrogens"] == 14)].copy()
    orca = orca_df[(orca_df["num_carbons"] == 24) & (orca_df["num_hydrogens"] == 14)]
    ref = orca[orca["basis_combo_id"] == reference_basis].copy()

    exess["common_id"] = exess["isomer_name"].apply(extract_common_id)
    exess["prefix"] = exess["isomer_name"].str.split("_").str[0]
    ref["common_id"] = ref["isomer"].apply(extract_common_id)
    ref["prefix"] = ref["isomer"].str.split("_").str[0]

    merged_list = []
    for opt in exess["optimizer"].unique():
        exess_opt = exess[exess["optimizer"] == opt]
        prefix = "compas3x" if opt == "xTB" else "compas3D"
        ref_opt = ref[ref["prefix"] == prefix]
        merged = exess_opt.merge(
            ref_opt[["common_id", "total_energy_hartree", "scf_energy_hartree", "dft_exc_hartree", "mp2_corr_hartree"]],
            on="common_id", suffixes=("_exess", "_orca")
        )
        merged_list.append(merged)

    merged = pd.concat(merged_list, ignore_index=True)
    
    exess_avg = merged["total_energy_hartree_exess"].mean()
    orca_avg = merged["total_energy_hartree_orca"].mean()
    exess_rel = merged["total_energy_hartree_exess"].values - exess_avg
    orca_rel = merged["total_energy_hartree_orca"].values - orca_avg

    abs_dev = merged["total_energy_hartree_exess"].values - merged["total_energy_hartree_orca"].values
    rel_dev = exess_rel - orca_rel
    scf_dev = merged["scf_energy_hartree_exess"].values - merged["scf_energy_hartree_orca"].values
    xc_dev = merged["xc_energy_hartree"].values - merged["dft_exc_hartree"].values
    mp2_dev = (merged["pt2_os_correction_hartree"] + merged["pt2_ss_correction_hartree"]).values - merged["mp2_corr_hartree"].values

    def mad(x): return np.mean(np.abs(x)) * HARTREE_TO_KJ_PER_MOL
    
    return {
        "mad_abs_kjmol": mad(abs_dev),
        "mad_rel_kjmol": mad(rel_dev),
        "mad_scf_kjmol": mad(scf_dev),
        "mad_xc_kjmol": mad(xc_dev),
        "mad_mp2_kjmol": mad(mp2_dev),
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
        "\\caption{Comparison of absolute and relative energies across basis set configurations for all 13 \\ce{C24H14} isomers in both the COMPAS-3x and COMPAS-3D datasets (26 geometries total). Energy deviations are reported relative to a def2-QZVPP calculation without RI-HF and with the def2-QZVPP-RIFIT basis for the PT2 step (direct PT2 without RI was prohibitively expensive for these systems). Relative energies were calculated with respect to the average across all 26 geometries for the corresponding basis set combination. All calculations were performed with the revDSD-PBEP86-D4(noFC) functional using ORCA v6.0.1.}\\label{tab:orca_mads}",
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

    print(f"EXESS MAD Absolute: {exess_result['mad_abs_kjmol']:.2f} kJ/mol")
    print(f"EXESS MAD Relative: {exess_result['mad_rel_kjmol']:.3f} kJ/mol")
    print(f"EXESS MAD SCF: {exess_result['mad_scf_kjmol']:.3f} kJ/mol")
    print(f"EXESS MAD XC: {exess_result['mad_xc_kjmol']:.3f} kJ/mol")
    print(f"EXESS MAD MP2: {exess_result['mad_mp2_kjmol']:.3f} kJ/mol")
    print("\n" + generate_latex_table(orca_results))


if __name__ == "__main__":
    main()
