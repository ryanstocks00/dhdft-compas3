#!/usr/bin/env python3
"""Generate supporting information document for DFT functional comparison paper."""

import shutil
import subprocess
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Add analysis directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from plotting_utils import (
    create_scatter_plot,
    extract_common_id,
    calculate_stats,
    HARTREE_TO_KJ_PER_MOL,
)

# Functional categories
LDA_FUNCTIONALS = ["SVWN5"]
GGA_FUNCTIONALS = ["PBE", "BLYP", "revPBE", "BP86", "BPW91", "B97-D", "HCTH407"]
MGGA_FUNCTIONALS = [
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

# Functionals without D4 support
NO_D4_FUNCTIONALS = ["SVWN5", "HCTH407", "M11-L", "MN15L", "t-HCTH"]


def format_functional_name(func):
    """Convert functional name to LaTeX-safe format."""
    return func.replace("-", "_").replace("(", "").replace(")", "").replace("/", "_")


def generate_xtb_comparison_plot(exess_csv, plots_dir):
    """Generate GFN2-xTB vs revDSD-PBEP86-D4 comparison plot."""
    print("Generating GFN2-xTB comparison plot...")

    # Load EXESS data
    exess_df = pd.read_csv(exess_csv)
    exess_df["common_id"] = exess_df["isomer_name"].apply(extract_common_id)

    # Filter to COMPAS-3x geometries with GFN2-xTB optimizer
    compas3x_df = exess_df[
        (exess_df["isomer_name"].str.contains("compas3x", case=False, na=False))
        & (exess_df["optimizer"] == "GFN2-xTB")
    ].copy()

    # Get revDSD-PBEP86-D4 data
    revdsd_df = compas3x_df[compas3x_df["functional"] == "revDSD-PBEP86-D4"].copy()

    if len(revdsd_df) == 0:
        print("Warning: No revDSD-PBEP86-D4 data found for COMPAS-3x")
        return False

    # Find minimum energy isomer for each (C, H) group using revDSD-PBEP86-D4
    revdsd_min_isomers = {}
    for (n_c, n_h), group in revdsd_df.groupby(["n_carbons", "n_hydrogens"]):
        min_idx = group["total_energy_hartree"].idxmin()
        min_common_id = group.loc[min_idx, "common_id"]
        revdsd_min_isomers[(n_c, n_h)] = min_common_id

    # Load COMPAS-3x data for xTB energies from local cache
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        compas3x_path = (
            project_root / ".compas_cache" / "compas" / "COMPAS-3" / "compas-3x.csv"
        )

        if not compas3x_path.exists():
            print(f"Warning: COMPAS-3x CSV not found at {compas3x_path}")
            return False

        compas3x_data = pd.read_csv(compas3x_path)

        # Extract common_id from molecule column
        if "molecule" not in compas3x_data.columns:
            print("Warning: 'molecule' column not found in COMPAS-3x data")
            return False

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

        # Get GFN2-xTB absolute energies (convert from eV to hartree, then to kJ/mol for comparison)
        if "Etot_eV" not in compas3x_data.columns:
            print("Warning: 'Etot_eV' column not found in COMPAS-3x data")
            return False

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

        # Merge with revDSD-PBEP86-D4 data
        merged = revdsd_df[
            ["common_id", "n_carbons", "n_hydrogens", "isomerization_energy_hartree"]
        ].merge(
            compas3x_data[["common_id", "xtb_isomerization_hartree"]],
            on="common_id",
            how="inner",
        )

        # Filter out rows where xTB isomerization energy couldn't be computed
        merged = merged.dropna(subset=["xtb_isomerization_hartree"])

        if len(merged) == 0:
            print(
                "Warning: No matching structures found between GFN2-xTB and revDSD-PBEP86-D4"
            )
            return False

        # Convert to kJ/mol
        revdsd_energies_kjmol = (
            merged["isomerization_energy_hartree"] * HARTREE_TO_KJ_PER_MOL
        )
        xtb_energies_kjmol = merged["xtb_isomerization_hartree"] * HARTREE_TO_KJ_PER_MOL

        # Calculate statistics
        deviations = xtb_energies_kjmol - revdsd_energies_kjmol
        mad_kjmol = np.mean(np.abs(deviations))
        r_squared, _, mad_percentage = calculate_stats(
            revdsd_energies_kjmol, xtb_energies_kjmol
        )

        print(
            f"  GFN2-xTB vs revDSD-PBEP86-D4: {len(merged)} structures, MAD = {mad_kjmol:.3f} kJ/mol, r² = {r_squared:.4f}"
        )

        # Calculate plot limits from all functional comparison data to match other plots
        # Collect all reference (revDSD-PBEP86-D4) and functional isomerization energies
        all_ref_energies = []
        all_func_energies = []

        # Get all revDSD-PBEP86-D4 isomerization energies
        revdsd_data = compas3x_df[compas3x_df["functional"] == "revDSD-PBEP86-D4"]
        if (
            len(revdsd_data) > 0
            and "isomerization_energy_hartree" in revdsd_data.columns
        ):
            all_ref_energies.extend(
                (
                    revdsd_data["isomerization_energy_hartree"] * HARTREE_TO_KJ_PER_MOL
                ).values
            )

        # Get all functional isomerization energies (excluding revDSD-PBEP86-D4)
        for func in compas3x_df["functional"].unique():
            if func != "revDSD-PBEP86-D4":
                func_data = compas3x_df[compas3x_df["functional"] == func]
                if (
                    len(func_data) > 0
                    and "isomerization_energy_hartree" in func_data.columns
                ):
                    all_func_energies.extend(
                        (
                            func_data["isomerization_energy_hartree"]
                            * HARTREE_TO_KJ_PER_MOL
                        ).values
                    )

        # Also include the current plot's data
        all_ref_energies.extend(revdsd_energies_kjmol.values)
        all_func_energies.extend(xtb_energies_kjmol.values)

        # Calculate global limits
        if len(all_ref_energies) > 0 and len(all_func_energies) > 0:
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
        else:
            plot_limits = None

        # Create plot with same limits as other plots
        plot_path = plots_dir / "compas3x_xtb_vs_revdsd.png"
        xlim = plot_limits[0] if plot_limits else None
        ylim = plot_limits[1] if plot_limits else None
        create_scatter_plot(
            revdsd_energies_kjmol,
            xtb_energies_kjmol,
            r"revDSD-PBEP86-D4(noFC) $\Delta E$ (kJ/mol)",
            r"GFN2-xTB $\Delta E$ (kJ/mol)",
            plot_path,
            mad_kjmol=mad_kjmol,
            xlim=xlim,
            ylim=ylim,
        )

        return True

    except Exception as e:
        print(f"Warning: Failed to generate GFN2-xTB comparison plot: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_latex_file(plots_dir, table_file, output_file):
    """Generate the supporting_information.tex file."""

    # Check which plot files exist
    plot_files = {f.stem: f for f in plots_dir.glob("compas3x_*_vs_revdsd.png")}
    plot_files_err_z = {
        f.stem: f for f in plots_dir.glob("compas3x_*_error_vs_max_z.png")
    }
    xtb_plot_exists = (plots_dir / "compas3x_xtb_vs_revdsd.png").exists()
    xtb_err_z_exists = (plots_dir / "compas3x_GFN2_xTB_error_vs_max_z.png").exists()

    def get_plot_path(func_name, with_d4):
        """Get the plot file path for a functional."""
        suffix = "_with_d4" if with_d4 else "_without_d4"
        plot_name = (
            f"compas3x_{format_functional_name(func_name)}{suffix}_vs_revdsd.png"
        )
        return f"plots/{plot_name}"

    def has_plot(func_name, with_d4):
        """Check if a plot exists for a functional."""
        suffix = "_with_d4" if with_d4 else "_without_d4"
        plot_stem = f"compas3x_{format_functional_name(func_name)}{suffix}_vs_revdsd"
        return plot_stem in plot_files

    def get_err_z_plot_path(func_name, with_d4):
        suffix = "_with_d4" if with_d4 else "_without_d4"
        return (
            f"plots/compas3x_{format_functional_name(func_name)}"
            f"{suffix}_error_vs_max_z.png"
        )

    def has_err_z_plot(func_name, with_d4):
        suffix = "_with_d4" if with_d4 else "_without_d4"
        stem = (
            f"compas3x_{format_functional_name(func_name)}{suffix}_error_vs_max_z"
        )
        return stem in plot_files_err_z

    def generate_error_vs_max_z_section(func_name, func_display):
        """Second figure: signed error vs max z (without/with D4 subfigures)."""
        if not (
            has_err_z_plot(func_name, False)
            or (
                func_name not in NO_D4_FUNCTIONALS and has_err_z_plot(func_name, True)
            )
        ):
            return ""
        has_d4 = func_name not in NO_D4_FUNCTIONALS
        latex = "\\begin{figure}[H]\n\\centering\n"
        latex += "\\begin{subfigure}{0.45\\textwidth}\n\\centering\n"
        if has_err_z_plot(func_name, False):
            latex += (
                f"\\includegraphics[width=\\textwidth]{{{get_err_z_plot_path(func_name, False)}}}\n"
            )
            latex += "\\caption{Without D4}\n"
        else:
            latex += "% Plot not available\n"
            latex += "\\caption{Without D4 - Not available}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_err_z_no_d4}}\n"
        latex += "\\end{subfigure}\n\\hfill\n"
        latex += "\\begin{subfigure}{0.45\\textwidth}\n\\centering\n"
        if has_d4 and has_err_z_plot(func_name, True):
            latex += (
                f"\\includegraphics[width=\\textwidth]{{{get_err_z_plot_path(func_name, True)}}}\n"
            )
            latex += "\\caption{With D4}\n"
        else:
            latex += (
                f"% {func_display} does not support D4\n"
                if not has_d4
                else "% Plot not available\n"
            )
            latex += "\\caption{With D4 - Not available}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_err_z_with_d4}}\n"
        latex += "\\end{subfigure}\n"
        latex += (
            f"\\caption{{{func_display}: signed isomerization-energy error vs "
            f"maximum $z$ displacement.}}\n"
        )
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_err_z}}\n"
        latex += "\\end{figure}\n\n"
        return latex

    def generate_functional_section(func_name, func_display):
        """Generate LaTeX code for a functional's figure section."""
        has_d4 = func_name not in NO_D4_FUNCTIONALS

        latex = f"\\FloatBarrier\n\\needspace{{7\\baselineskip}}\n\\subsubsection{{{func_display}}}\n\n"
        latex += "\\begin{figure}[H]\n\\centering\n"

        # Left subfigure (without D4)
        latex += "\\begin{subfigure}{0.45\\textwidth}\n\\centering\n"
        if has_plot(func_name, False):
            latex += f"\\includegraphics[width=\\textwidth]{{{get_plot_path(func_name, False)}}}\n"
            latex += "\\caption{Without D4}\n"
        else:
            latex += "% Plot not available\n"
            latex += "\\caption{Without D4 - Not available}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_no_d4}}\n"
        latex += "\\end{subfigure}\n\\hfill\n"

        # Right subfigure (with D4)
        latex += "\\begin{subfigure}{0.45\\textwidth}\n\\centering\n"
        if has_d4 and has_plot(func_name, True):
            latex += f"\\includegraphics[width=\\textwidth]{{{get_plot_path(func_name, True)}}}\n"
            latex += "\\caption{With D4}\n"
        else:
            latex += (
                f"% {func_display} does not support D4\n"
                if not has_d4
                else "% Plot not available\n"
            )
            latex += "\\caption{With D4 - Not available}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_with_d4}}\n"
        latex += "\\end{subfigure}\n"

        latex += f"\\caption{{{func_display}/def2-TZVP comparison to revDSD-PBEP86-D4(noFC)/def2-QZVPP isomerization energies for COMPAS-3x geometries.}}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}}}\n"
        latex += "\\end{figure}\n\n"

        return latex

    # Get table filename (just the name, not the path)
    table_filename = table_file.name if isinstance(table_file, Path) else table_file

    # Generate LaTeX content
    latex_content = (
        """\\documentclass[12pt]{achemso}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{placeins}
\\usepackage{subcaption}
\\usepackage{needspace}
\\usepackage{rotating}
\\usepackage{microtype}
\\usepackage{hyperref}
\\usepackage{amsmath}
\\usepackage{float}
\\usepackage{enumitem}
\\usepackage{lmodern}
\\setlist[itemize]{itemsep=0pt, parsep=0pt, topsep=2pt, partopsep=0pt}
\\usepackage{textcomp}
\\usepackage[T1]{fontenc}
\\renewcommand{\\ttdefault}{lmtt}
\\newcommand{\\colname}[1]{{\\ttfamily\\bfseries #1}}
\\renewcommand{\\affilfont}{\\scriptsize}
\\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={Supporting Information: Double-Hybrid, but not Double-Cost: GPU Accelerated DHDFT for the COMPAS-3 Dataset},
    pdfauthor={Ryan Stocks, Elise Palethorpe, Amir Karton, Giuseppe M. J. Barca},
}

\\title{Supporting Information:\\\\
Double-Hybrid, but not Double-Cost: GPU Accelerated DHDFT for the COMPAS-3 Dataset}

\\author{Ryan Stocks}
\\affiliation{School of Computing, Australian National University, Canberra, ACT 2601, Australia}
\\email{ryan.stocks@anu.edu.au}

\\author{Elise Palethorpe}
\\affiliation{School of Computing, Australian National University, Canberra, ACT 2601, Australia}

\\author{Amir Karton}
\\affiliation{School of Science and Technology, University of New England, Armidale, New South Wales, Australia}

\\author{Giuseppe M. J. Barca}
\\affiliation{School of Computing, Australian National University, Canberra, ACT 2601, Australia}
\\alsoaffiliation{Monash Institute of Pharmaceutical Sciences, Melbourne, Parkville VIC 3052, Australia}
\\alsoaffiliation{QDX Technologies, Dickson, ACT 2602, Australia}
\\email{giuseppe.barca@anu.edu.au}

\\date{}

\\begin{document}

\\maketitle

\\section{Introduction}

This supporting information provides parameters for linear-fit corrections, comparison plots for benchmarked functionals against the reference method revDSD-PBEP86-D4(noFC)/def2-QZVPP, and detailed descriptions of the produced dataset of DFT energies evaluated on the COMPAS-3 database of polybenzenoid hydrocarbon isomers. The generated dataset contains revDSD-PBEP86-D4(noFC)/def2-QZVPP calculations on the full COMPAS-3 dataset (both COMPAS-3x and COMPAS-3D geometries), along with benchmark calculations for LDA, GGA, and meta-GGA functionals on the COMPAS-3x subset. The dataset includes a breakdown of energy components including SCF, PT2 opposite-spin and same-spin corrections, D4 dispersion, exchange-correlation, and nuclear repulsion energies, molecular orbital properties (HOMO, LUMO, HOMO-LUMO gap) from the SCF calculation, and computational performance metrics including timing and floating-point throughput.

All calculations were performed using the Extreme Scale Electronic Structure System (EXESS) software package with a robustly pruned (99,590) integration grid. The revDSD-PBEP86-D4(noFC) calculations were performed using the def2-QZVPP basis set and all LDA, GGA, and meta-GGA calculations used def2-TZVP. A tight convergence threshold of $10^{-10}$ was used for all calculations which were each performed on a single 4 $\\times$ A100 node of the Perlmutter supercomputer in batches of 20 geometries per submitted calculation. The RI approximation was used for both the SCF and PT2 components of all calculations.

\\section{Linear Fit Corrections}

Here we present linear fit corrections and the mean absolute deviation following the correction for the tested DFT functionals against the reference method revDSD-PBEP86-D4(noFC)/def2-QZVPP. The statistics are based on isomerization energies for all COMPAS-3x geometries. 

Isomerization energies were evaluated with respect to the isomer with the minimum energy using the revDSD-PBEP86-D4(noFC) functional. For an isomer $A$ and functional $F$, the isomerization energy is calculated as:
\\begin{equation}
\\Delta E ^F_{A,\\text{isomerization}} = E^F_A - E^F_{M(A)},
\\end{equation}
where $M(A)$ is the minimum energy isomer evaluated using the double hybrid reference method. As such, isomerization energies can be negative for the benchmarked functionals because of error resulting in a different minimum-energy isomer.

\\FloatBarrier
\\input{"""
        + table_filename
        + """}
\\FloatBarrier

The table shows linear fit correction parameters (slope and offset), and MAD after linear fit correction. The linear fit correction takes the form:
\\begin{equation}
E^{\\text{corrected}}_A = E^F_A \\times \\text{slope} + \\text{offset},
\\end{equation}
where the slope and offset are determined by linear regression of the reference revDSD-PBEP86-D4(noFC) isomerization energies relative to the tested functionals isomerization energies. The linear fit correction significantly improves the performance of many functionals, with revPBE-D4 achieving the best performance with a corrected MAD of just 1.4~kJ/mol.

\\FloatBarrier
\\subsection{Functional benchmarks by planar/non-planar}

Table~\\ref{tab:compas3x_benchmarks_maxz_combined} lists MAD, MSD, and $r^2$ with no linear correction for COMPAS-3x structures grouped into planar (max $z$ $< 1$~\\AA) and non-planar (max $z$ $\\ge 1$~\\AA) subsets (GFN2-xTB-optimized geometries).

\\FloatBarrier
\\input{compas3x_benchmarks_maxz_combined.tex}
\\FloatBarrier

\\section{Comparison Plots}

This section contains scatter plots comparing various DFT functionals to revDSD-PBEP86-D4(noFC)/def2-QZVPP for COMPAS-3x geometries. 

For each functional, two subfigures are provided: the left subfigure shows comparisons without the D4 dispersion correction, and the right subfigure shows comparisons with the D4 correction (when supported by the functional). Each plot displays the mean absolute deviation (MAD) in kJ/mol, and the red dashed diagonal line represents perfect agreement between the methods.

Immediately below each functional's energy comparison figure, a second figure shows the \\emph{signed} isomerization-energy error relative to revDSD-PBEP86-D4(noFC) as a function of maximum out-of-plane displacement (see dataset column \\colname{max\\_z\\_displacement}), again without/with D4 subfigures.

\\subsection{LDA Functionals}

"""
    )

    # Add LDA functionals
    for func in LDA_FUNCTIONALS:
        latex_content += generate_functional_section(func, func)
        latex_content += generate_error_vs_max_z_section(func, func)

    # Add GGA functionals
    latex_content += "\\FloatBarrier\n\\subsection{GGA Functionals}\n\n"
    for func in GGA_FUNCTIONALS:
        latex_content += generate_functional_section(func, func)
        latex_content += generate_error_vs_max_z_section(func, func)

    # Add MGGA functionals
    latex_content += "\\FloatBarrier\n\\subsection{MGGA Functionals}\n\n"
    for func in MGGA_FUNCTIONALS:
        latex_content += generate_functional_section(func, func)
        latex_content += generate_error_vs_max_z_section(func, func)

    # Add GFN2-xTB comparison
    if xtb_plot_exists:
        latex_content += """\\FloatBarrier
\\subsection{Semiempirical Methods}

\\FloatBarrier
\\needspace{7\\baselineskip}
\\subsubsection{GFN2-xTB}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.45\\textwidth]{plots/compas3x_xtb_vs_revdsd.png}
\\caption{GFN2-xTB comparison to revDSD-PBEP86-D4(noFC)/def2-QZVPP isomerization energies for COMPAS-3x geometries.}
\\label{fig:xtb}
\\end{figure}

"""
        if xtb_err_z_exists:
            latex_content += """\\begin{figure}[H]
\\centering
\\includegraphics[width=0.45\\textwidth]{plots/compas3x_GFN2_xTB_error_vs_max_z.png}
\\caption{GFN2-xTB: signed isomerization-energy error vs maximum $z$ displacement.}
\\label{fig:xtb_err_z}
\\end{figure}

"""

    latex_content += """\\section{Dataset Description}

This section provides a description of the computational dataset generated through this study. The complete dataset is available in the supporting information as a CSV file named \\texttt{compas3\\_dhdft\\_benchmark\\_raw\\_data.csv}. Each row represents a single calculation for a specific molecular isomer geometry and functional combination. The dataset includes:

\\begin{itemize}
    \\item All revDSD-PBEP86-D4(noFC)/def2-QZVPP calculations on the full COMPAS-3 dataset (both COMPAS-3x and COMPAS-3D geometries)
    \\item Benchmark calculations for LDA, GGA, and MGGA functionals on the COMPAS-3x subset using def2-TZVP basis set
    \\item Energy components including SCF, PT2 opposite-spin and same-spin corrections, D4 dispersion, exchange-correlation, and nuclear repulsion energies
    \\item Molecular orbital properties (HOMO, LUMO, HOMO-LUMO gap) from the SCF calculation
    \\item Computational performance metrics including timing and floating-point throughput
\\end{itemize}

\\subsubsection{Column Descriptions}

The following sections describe all columns in the dataset, organized by category:

\\textbf{System Identification:}

\\begin{itemize}
    \\item \\colname{isomer\\_name}: Full name of the molecular isomer, including prefix indicating the geometry source (e.g., \\texttt{compas3x\\_hc\\_c24h14\\_0pent\\_1} for COMPAS-3x, \\texttt{compas3D\\_hc\\_c24h14\\_0pent\\_1} for COMPAS-3D).
    \\item \\colname{optimizer}: Geometry optimization method used: \\texttt{GFN2-xTB} for COMPAS-3x geometries or \\texttt{CAM-B3LYP-D3BJ} for COMPAS-3D geometries.
    \\item \\colname{functional}: Density functional used in the calculation (e.g., \\texttt{revDSD-PBEP86-D4}, \\texttt{PBE}, \\texttt{BLYP}, \\texttt{TPSS}).
    \\item \\colname{basis\\_set}: Basis set used (e.g., \\texttt{def2-QZVPP}, \\texttt{def2-TZVP}).
    \\item \\colname{id}: Unique identifier for the isomer within its (C, H) group.
\\end{itemize}

\\textbf{Energy Components:}

\\begin{itemize}
    \\item \\colname{total\\_energy\\_hartree}: Total electronic energy in hartree. For double-hybrid functionals (e.g., revDSD-PBEP86-D4), this is calculated as: SCF energy + PT2 opposite-spin correction + PT2 same-spin correction + D4 dispersion correction. For LDA, GGA, and MGGA functionals, the PT2 corrections are zero (\\texttt{None}), so the total energy is: SCF energy + D4 dispersion correction (when applicable). Note that the D4 correction must be subtracted to obtain the raw functionals energy.
    \\item \\colname{isomerization\\_energy\\_hartree}: Energy relative to the minimum energy isomer (with revDSD-PBEP86-D4(noFC)) for the same (C, H) composition, calculated with the same functional and basis set. This represents the relative stability of different structural isomers.
    \\item \\colname{scf\\_energy\\_hartree}: Converged self-consistent field (SCF) energy in hartree, prior to PT2 step for double hybrids.
    \\item \\colname{pt2\\_os\\_correction\\_hartree}: PT2 opposite-spin (OS) correlation correction in hartree, accounting for electron correlation between electrons of opposite spin. This value is \\texttt{None} for LDA, GGA, and MGGA functionals, as PT2 corrections are only calculated for double-hybrid functionals (e.g., revDSD-PBEP86-D4). Note that it has already been scaled by $c_{os}$.
    \\item \\colname{pt2\\_ss\\_correction\\_hartree}: PT2 same-spin (SS) correlation correction in hartree, accounting for electron correlation between electrons of the same spin. This value is \\texttt{None} for LDA, GGA, and MGGA functionals, as PT2 corrections are only calculated for double-hybrid functionals (e.g., revDSD-PBEP86-D4). Note that it has already been scaled by $c_{ss}$.
    \\item \\colname{d4\\_energy\\_hartree}: D4 dispersion correction energy in hartree. This value is \\texttt{None} for functionals that do not support the D4 correction (e.g., SVWN5, HCTH407, M11-L, MN15L, t-HCTH).
    \\item \\colname{xc\\_energy\\_hartree}: Exchange-correlation energy from the DFT functional evaluated on the converged SCF density in hartree.
    \\item \\colname{nuc\\_repulsion\\_energy\\_hartree}: Nuclear repulsion energy in hartree, representing the classical electrostatic repulsion between atomic nuclei.
    \\item \\colname{elec\\_energy\\_hartree}: Total electronic energy in hartree, calculated as SCF energy $-$ Nuclear Repulsion Energy $-$ XC energy.
\\end{itemize}

\\textbf{Molecular Orbital Properties:}

\\begin{itemize}
    \\item \\colname{homo\\_hartree}: Highest occupied molecular orbital (HOMO) energy in hartree (no PT2 correction).
    \\item \\colname{lumo\\_hartree}: Lowest unoccupied molecular orbital (LUMO) energy in hartree (no PT2 correction).
    \\item \\colname{hlg\\_hartree}: HOMO-LUMO gap in hartree, calculated as LUMO energy - HOMO energy. This represents the energy gap between the highest occupied and lowest unoccupied molecular orbitals.
\\end{itemize}

\\textbf{Molecular Composition:}

\\begin{itemize}
    \\item \\colname{n\\_atoms}: Total number of atoms in the molecule (\\# Carbons $+$ \\# Hydrogens).
    \\item \\colname{n\\_carbons}: Number of carbon atoms.
    \\item \\colname{n\\_hydrogens}: Number of hydrogen atoms.
\\end{itemize}

\\textbf{Computational Details:}

\\begin{itemize}
    \\item \\colname{n\\_primary\\_basis\\_functions}: Number of primary basis functions used in the calculation.
    \\item \\colname{n\\_scf\\_iterations}: Number of self-consistent field (SCF) iterations required for convergence.
\\end{itemize}

\\textbf{Timing Information:}

\\begin{itemize}
    \\item \\colname{total\\_time\\_s}: Total calculation time in seconds.
    \\item \\colname{scf\\_time\\_s}: Time spent in self-consistent field (SCF) iterations in seconds.
    \\item \\colname{pt2\\_time\\_s}: Time spent in PT2 correlation calculation in seconds.
    \\item \\colname{b\\_formation\\_time\\_s}: Time spent forming $B_{\\mu\\nu}^P$ matrices for the resolution of identity (RI) approximation in seconds.
    \\item \\colname{diag\\_time\\_s}: Time spent in Fock matrix diagonalization during SCF iterations in seconds.
    \\item \\colname{ri\\_fock\\_time\\_s}: Time spent in RI-Fock matrix construction (Coulomb + Exchange) in seconds.
    \\item \\colname{xc\\_time\\_s}: Time spent in exchange-correlation energy integration in seconds.
    \\item \\colname{basis\\_transforms\\_time\\_s}: Time spent in basis set transformations (Cartesian to Spherical) in seconds.
\\end{itemize}

\\textbf{Computational Performance:}

Only floating point operations from dense linear algebra operations are counted towards the throughput metrics so these are lower bounds on the actual performance.

\\begin{itemize}
    \\item \\colname{total\\_tflop/s}: Total computational throughput in teraflops per second (TFLOP/s) for the overall calculation.
    \\item \\colname{scf\\_tflop/s}: SCF computational throughput in teraflops per second.
    \\item \\colname{pt2\\_tflop/s}: PT2 computational throughput in teraflops per second.
    \\item \\colname{b\\_formation\\_tflop/s}: $B_{\\mu\\nu}^P$ matrix formation computational throughput in teraflops per second.
    \\item \\colname{ri\\_fock\\_tflop/s}: RI-Fock matrix construction computational throughput in teraflops per second.
    \\item \\colname{xc\\_tflop/s}: Exchange-correlation integration computational throughput in teraflops per second.
\\end{itemize}

\\end{document}
"""

    # Write the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"Generated LaTeX file: {output_file}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    exess_csv = project_root / "analysis" / "exess_data.csv"
    plots_dir = script_dir / "plots"
    table_file = script_dir / "compas3x_benchmarks.tex"
    latex_file = script_dir / "supporting_information.tex"

    # Check if EXESS CSV exists
    if not exess_csv.exists():
        print(f"ERROR: EXESS data file not found at {exess_csv}")
        sys.exit(1)

    # Check if benchmark script exists
    benchmark_script = script_dir / "benchmark_compas3x.py"
    if not benchmark_script.exists():
        print(f"ERROR: Benchmark script not found at {benchmark_script}")
        sys.exit(1)

    print("=" * 70)
    print("Generating Supporting Information Document")
    print("=" * 70)
    print(f"\nThis script will automatically run the benchmark script to generate")
    print(f"all required plots and tables before compiling the LaTeX document.\n")

    # Step 1: Generate plots with min reference method (no linear correction)
    print(
        "Step 1: Running benchmark script to generate plots (minimum reference method)..."
    )
    print("-" * 70)
    # Use relative paths for output to ensure consistent path resolution
    output_txt = "compas3x_benchmarks.txt"
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--exess-csv",
        str(exess_csv),
        "--output-dir",
        str(plots_dir),
        "--output",
        output_txt,
        "--reference-method",
        "min",
        "--write-maxz-combined-table",
    ]

    result = subprocess.run(cmd, cwd=str(script_dir))
    if result.returncode != 0:
        print("\nERROR: Benchmark script failed to generate plots")
        print(f"Command: {' '.join(cmd)}")
        sys.exit(1)

    print("✓ Plots generated successfully\n")

    maxz_tex = script_dir / "compas3x_benchmarks_maxz_combined.tex"
    if not maxz_tex.exists():
        print(
            f"WARNING: {maxz_tex.name} missing after Step 1; "
            "LaTeX will fail on \\input{compas3x_benchmarks_maxz_combined.tex} unless you regenerate it.\n"
        )

    # Save plots directory before second run (which will overwrite plots)
    plots_backup = script_dir / "plots_backup"
    if plots_backup.exists():
        shutil.rmtree(plots_backup)
    shutil.copytree(plots_dir, plots_backup)

    # Step 2: Generate table with linear_fit to get gradient/offset columns
    print(
        "Step 2: Running benchmark script to generate statistics table (linear fit method)..."
    )
    print("-" * 70)
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--exess-csv",
        str(exess_csv),
        "--output-dir",
        str(plots_dir),
        "--output",
        output_txt,
        "--reference-method",
        "linear_fit",
    ]

    result = subprocess.run(cmd, cwd=str(script_dir))
    if result.returncode != 0:
        print("\nERROR: Benchmark script failed to generate statistics table")
        print(f"Command: {' '.join(cmd)}")
        sys.exit(1)

    # Verify that the .tex file was generated
    if not table_file.exists():
        print(f"\nERROR: LaTeX table file not found at {table_file}")
        print(
            "The benchmark script should have generated this file. Check for errors above."
        )
        sys.exit(1)

    print("✓ Statistics table generated successfully\n")

    # Restore plots from min run (overwrite the linear_fit plots)
    print("Step 3: Restoring plots with minimum reference method...")
    for pattern in (
        "compas3x_*_vs_revdsd.png",
        "compas3x_*_vs_revdsd_small.png",
        "compas3x_*_error_vs_max_z.png",
        "compas3x_*_error_vs_max_z_small.png",
    ):
        for plot_file in plots_backup.glob(pattern):
            shutil.copy2(plot_file, plots_dir / plot_file.name)

    # Clean up backup
    shutil.rmtree(plots_backup)
    print("✓ Plots restored\n")

    # Step 4: Generate GFN2-xTB comparison plot
    print("Step 4: Generating GFN2-xTB comparison plot...")
    print("-" * 70)
    if generate_xtb_comparison_plot(exess_csv, plots_dir):
        print("✓ GFN2-xTB comparison plot generated\n")
    else:
        print("⚠ Warning: GFN2-xTB comparison plot could not be generated\n")

    z_err_script = script_dir / "plot_error_vs_z_displacement.py"
    if z_err_script.exists():
        print("Step 4b: Signed error vs max $z$ displacement plots...")
        print("-" * 70)
        zcmd = [
            sys.executable,
            str(z_err_script),
            "--exess-csv",
            str(exess_csv),
            "--output-dir",
            str(plots_dir),
            "--reference-method",
            "min",
        ]
        zres = subprocess.run(zcmd, cwd=str(script_dir))
        if zres.returncode == 0:
            print("✓ Error-vs-max-z plots finished\n")
        else:
            print("⚠ Warning: plot_error_vs_z_displacement.py failed\n")
    else:
        print(
            "⚠ Skipping error-vs-z plots (plot_error_vs_z_displacement.py not found)\n"
        )

    # Step 5: Generate LaTeX file
    print("Step 5: Generating supporting_information.tex...")
    print("-" * 70)
    generate_latex_file(plots_dir, table_file, latex_file)
    print("✓ LaTeX file generated\n")

    # Step 6: Compile LaTeX document
    print("Step 6: Compiling LaTeX document...")
    print("-" * 70)
    latex_dir = latex_file.parent
    latex_filename = latex_file.name

    # Run pdflatex twice to resolve references
    for run_num in [1, 2]:
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(latex_dir),
            str(latex_filename),
        ]
        subprocess.run(cmd, cwd=str(latex_dir))

    # Clean up intermediate LaTeX files (but keep .tex files)
    intermediate_extensions = [
        ".aux",
        ".log",
        ".out",
        ".toc",
        ".lof",
        ".lot",
        ".fls",
        ".fdb_latexmk",
        ".synctex.gz",
    ]
    for ext in intermediate_extensions:
        intermediate_file = latex_file.with_suffix(ext)
        if intermediate_file.exists():
            intermediate_file.unlink()

    pdf_file = latex_file.with_suffix(".pdf")
    if pdf_file.exists():
        print("✓ LaTeX compilation successful")
        print("=" * 70)
        print(f"\nSUCCESS: Supporting information PDF generated at {pdf_file}")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"\nWARNING: PDF file not found at {pdf_file}")
        print("Check LaTeX compilation output above for errors.")
        print("=" * 70)


if __name__ == "__main__":
    main()
