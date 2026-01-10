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
    create_scatter_plot, extract_common_id, calculate_stats, HARTREE_TO_KJ_PER_MOL
)

# Functional categories
LDA_FUNCTIONALS = ['SVWN5']
GGA_FUNCTIONALS = ['PBE', 'BLYP', 'revPBE', 'BP86', 'BPW91', 'B97-D', 'HCTH407']
MGGA_FUNCTIONALS = ['TPSS', 'MN15L', 'SCAN', 'rSCAN', 'r2SCAN', 'revTPSS', 't-HCTH', 'M06-L', 'M11-L']

# Functionals without D4 support
NO_D4_FUNCTIONALS = ['SVWN5', 'HCTH407', 'M11-L', 'MN15L', 't-HCTH']


def format_functional_name(func):
    """Convert functional name to LaTeX-safe format."""
    return func.replace('-', '_').replace('(', '').replace(')', '').replace('/', '_')


def generate_xtb_comparison_plot(exess_csv, plots_dir):
    """Generate GFN2-xTB vs revDSD-PBEP86-D4 comparison plot."""
    print("Generating GFN2-xTB comparison plot...")
    
    # Load EXESS data
    exess_df = pd.read_csv(exess_csv)
    exess_df['common_id'] = exess_df['isomer_name'].apply(extract_common_id)
    
    # Filter to COMPAS-3x geometries with xTB optimizer
    compas3x_df = exess_df[
        (exess_df['isomer_name'].str.contains('compas3x', case=False, na=False)) &
        (exess_df['optimizer'] == 'xTB')
    ].copy()
    
    # Get revDSD-PBEP86-D4 data
    revdsd_df = compas3x_df[compas3x_df['functional'] == 'revDSD-PBEP86-D4'].copy()
    
    if len(revdsd_df) == 0:
        print("Warning: No revDSD-PBEP86-D4 data found for COMPAS-3x")
        return False
    
    # Find minimum energy isomer for each (C, H) group using revDSD-PBEP86-D4
    revdsd_min_isomers = {}
    for (n_c, n_h), group in revdsd_df.groupby(['n_carbons', 'n_hydrogens']):
        min_idx = group['total_energy_hartree'].idxmin()
        min_common_id = group.loc[min_idx, 'common_id']
        revdsd_min_isomers[(n_c, n_h)] = min_common_id
    
    # Load COMPAS-3x data for xTB energies from local cache
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        compas3x_path = project_root / '.compas_cache' / 'compas' / 'COMPAS-3' / 'compas-3x.csv'
        
        if not compas3x_path.exists():
            print(f"Warning: COMPAS-3x CSV not found at {compas3x_path}")
            return False
        
        compas3x_data = pd.read_csv(compas3x_path)
        
        # Extract common_id from molecule column
        if 'molecule' not in compas3x_data.columns:
            print("Warning: 'molecule' column not found in COMPAS-3x data")
            return False
        
        compas3x_data['common_id'] = compas3x_data['molecule'].apply(extract_common_id)
        
        # Extract n_carbons and n_hydrogens from common_id (format: c16h10_0pent_1)
        def extract_carbons_hydrogens(common_id):
            match = re.match(r'c(\d+)h(\d+)', common_id)
            if match:
                return int(match.group(1)), int(match.group(2))
            return None, None
        
        compas3x_data[['n_carbons', 'n_hydrogens']] = compas3x_data['common_id'].apply(
            lambda x: pd.Series(extract_carbons_hydrogens(x))
        )
        
        # Get GFN2-xTB absolute energies (convert from eV to hartree, then to kJ/mol for comparison)
        if 'Etot_eV' not in compas3x_data.columns:
            print("Warning: 'Etot_eV' column not found in COMPAS-3x data")
            return False
        
        # Convert eV to hartree (1 eV = 0.0367493 hartree)
        compas3x_data['xtb_energy_hartree'] = compas3x_data['Etot_eV'] * 0.0367493
        
        # Build lookup for xTB energies by common_id
        xtb_energy_lookup = compas3x_data.set_index('common_id')['xtb_energy_hartree'].to_dict()
        
        # Compute xTB isomerization energies relative to minimum revDSD-PBEP86-D4 isomer
        def compute_xtb_isomerization(row):
            ch_key = (row['n_carbons'], row['n_hydrogens'])
            min_common_id = revdsd_min_isomers.get(ch_key)
            if min_common_id is None:
                return None
            xtb_energy = xtb_energy_lookup.get(row['common_id'])
            xtb_min_energy = xtb_energy_lookup.get(min_common_id)
            if xtb_energy is None or xtb_min_energy is None:
                return None
            return xtb_energy - xtb_min_energy
        
        compas3x_data['xtb_isomerization_hartree'] = compas3x_data.apply(compute_xtb_isomerization, axis=1)
        
        # Merge with revDSD-PBEP86-D4 data
        merged = revdsd_df[['common_id', 'n_carbons', 'n_hydrogens', 'isomerization_energy_hartree']].merge(
            compas3x_data[['common_id', 'xtb_isomerization_hartree']],
            on='common_id',
            how='inner'
        )
        
        # Filter out rows where xTB isomerization energy couldn't be computed
        merged = merged.dropna(subset=['xtb_isomerization_hartree'])
        
        if len(merged) == 0:
            print("Warning: No matching structures found between GFN2-xTB and revDSD-PBEP86-D4")
            return False
        
        # Convert to kJ/mol
        revdsd_energies_kjmol = merged['isomerization_energy_hartree'] * HARTREE_TO_KJ_PER_MOL
        xtb_energies_kjmol = merged['xtb_isomerization_hartree'] * HARTREE_TO_KJ_PER_MOL
        
        # Calculate statistics
        deviations = xtb_energies_kjmol - revdsd_energies_kjmol
        mad_kjmol = np.mean(np.abs(deviations))
        r_squared, _, mad_percentage = calculate_stats(revdsd_energies_kjmol, xtb_energies_kjmol)
        
        print(f"  GFN2-xTB vs revDSD-PBEP86-D4: {len(merged)} structures, MAD = {mad_kjmol:.3f} kJ/mol, r² = {r_squared:.4f}")
        
        # Create plot
        plot_path = plots_dir / 'compas3x_xtb_vs_revdsd.png'
        create_scatter_plot(
            revdsd_energies_kjmol, xtb_energies_kjmol,
            r'revDSD-PBEP86-D4(noFC) $\Delta E$ (kJ/mol)',
            r'GFN2-xTB $\Delta E$ (kJ/mol)',
            plot_path, mad_kjmol=mad_kjmol
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
    xtb_plot_exists = (plots_dir / 'compas3x_xtb_vs_revdsd.png').exists()
    
    def get_plot_path(func_name, with_d4):
        """Get the plot file path for a functional."""
        suffix = "_with_d4" if with_d4 else "_without_d4"
        plot_name = f"compas3x_{format_functional_name(func_name)}{suffix}_vs_revdsd.png"
        return f"plots/{plot_name}"
    
    def has_plot(func_name, with_d4):
        """Check if a plot exists for a functional."""
        suffix = "_with_d4" if with_d4 else "_without_d4"
        plot_stem = f"compas3x_{format_functional_name(func_name)}{suffix}_vs_revdsd"
        return plot_stem in plot_files
    
    def generate_functional_section(func_name, func_display):
        """Generate LaTeX code for a functional's figure section."""
        has_d4 = func_name not in NO_D4_FUNCTIONALS
        
        latex = f"\\FloatBarrier\n\\needspace{{15\\baselineskip}}\n\\subsubsection{{{func_display}}}\n\n"
        latex += "\\begin{figure}[htbp]\n\\centering\n"
        
        # Left subfigure (without D4)
        latex += "\\begin{subfigure}{0.48\\textwidth}\n\\centering\n"
        if has_plot(func_name, False):
            latex += f"\\includegraphics[width=\\textwidth]{{{get_plot_path(func_name, False)}}}\n"
            latex += "\\caption{Without D4}\n"
        else:
            latex += "% Plot not available\n"
            latex += "\\caption{Without D4 - Not available}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_no_d4}}\n"
        latex += "\\end{subfigure}\n\\hfill\n"
        
        # Right subfigure (with D4)
        latex += "\\begin{subfigure}{0.48\\textwidth}\n\\centering\n"
        if has_d4 and has_plot(func_name, True):
            latex += f"\\includegraphics[width=\\textwidth]{{{get_plot_path(func_name, True)}}}\n"
            latex += "\\caption{With D4}\n"
        else:
            latex += f"% {func_display} does not support D4\n" if not has_d4 else "% Plot not available\n"
            latex += "\\caption{With D4 - Not available}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}_with_d4}}\n"
        latex += "\\end{subfigure}\n"
        
        latex += f"\\caption{{{func_display} comparison to revDSD-PBEP86-D4}}\n"
        latex += f"\\label{{fig:{format_functional_name(func_name).lower()}}}\n"
        latex += "\\end{figure}\n\n"
        
        return latex
    
    # Get table filename (just the name, not the path)
    table_filename = table_file.name if isinstance(table_file, Path) else table_file
    
    # Generate LaTeX content
    latex_content = """\\documentclass[12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{geometry}
\\usepackage{placeins}
\\usepackage{subcaption}
\\usepackage{needspace}
\\usepackage{rotating}
\\geometry{a4paper, margin=1in}

\\title{Supporting Information:\\\\
Comparison of DFT Functionals to revDSD-PBEP86-D4}
\\author{}
\\date{}

\\begin{document}

\\maketitle

\\section{Dataset Summary}

This section provides a comprehensive description of the computational datasets used in this study. All data files are available in the supporting information.

\\subsection{EXESS Dataset (\\texttt{exess\\_data.csv})}

The EXESS dataset contains results from density-functional theory calculations performed using the EXESS (Extended Embedded Subsystem) method. This dataset includes calculations for COMPAS-3x (GFN2-xTB optimized geometries), COMPAS-3D (CAM-B3LYP-D3BJ optimized geometries), and PAH335 (G4(MP2) optimized geometries) molecular systems across multiple functionals and basis sets.

\\subsubsection{Column Descriptions}

\\begin{description}
\\item[\\texttt{isomer\\_name}] Full name of the molecular isomer, including prefix (e.g., \\texttt{compas3x\\_hc\\_c24h14\\_0pent\\_1}, \\texttt{compas3D\\_hc\\_c24h14\\_0pent\\_1}, \\texttt{PAH335\\_C24H14\\_pah1}).

\\item[\\texttt{optimizer}] Geometry optimization method used: \\texttt{GFN2-xTB} for COMPAS-3x geometries or \\texttt{CAM-B3LYP-D3BJ} for COMPAS-3D geometries.

\\item[\\texttt{functional}] Density functional used in the calculation (e.g., \\texttt{revDSD-PBEP86-D4}, \\texttt{PBE}, \\texttt{BLYP}, \\texttt{TPSS}).

\\item[\\texttt{basis\\_set}] Basis set used (e.g., \\texttt{def2-QZVPP}, \\texttt{def2-TZVP}).

\\item[\\texttt{batch\\_name}] Name of the calculation batch (e.g., \\texttt{isomers\\_0-19}).

\\item[\\texttt{topology\\_index}] Index of the topology within the batch.

\\item[\\texttt{id}] Unique identifier for the isomer within its (C, H) group.

\\item[\\texttt{filename}] Name of the JSON output file from EXESS.

\\item[\\texttt{total\\_energy\\_hartree}] Total electronic energy in hartree, including SCF, MP2 corrections, and D4 dispersion correction.

\\item[\\texttt{isomerization\\_energy\\_hartree}] Energy relative to the minimum energy isomer for the same (C, H) composition, calculated with the same functional and basis set.

\\item[\\texttt{scf\\_energy\\_hartree}] Self-consistent field (SCF) energy in hartree.

\\item[\\texttt{pt2\\_os\\_correction\\_hartree}] MP2 opposite-spin (OS) correlation correction in hartree.

\\item[\\texttt{pt2\\_ss\\_correction\\_hartree}] MP2 same-spin (SS) correlation correction in hartree.

\\item[\\texttt{d4\\_energy\\_hartree}] D4 dispersion correction energy in hartree (may be \\texttt{None} for functionals without D4 support).

\\item[\\texttt{xc\\_energy\\_hartree}] Exchange-correlation energy from the DFT functional in hartree.

\\item[\\texttt{nuc\\_repulsion\\_energy\\_hartree}] Nuclear repulsion energy in hartree.

\\item[\\texttt{elec\\_energy\\_hartree}] Total electronic energy (SCF + XC) in hartree.

\\item[\\texttt{homo\\_hartree}] Highest occupied molecular orbital (HOMO) energy in hartree.

\\item[\\texttt{lumo\\_hartree}] Lowest unoccupied molecular orbital (LUMO) energy in hartree.

\\item[\\texttt{hlg\\_hartree}] HOMO-LUMO gap in hartree (LUMO - HOMO).

\\item[\\texttt{n\\_primary\\_basis\\_functions}] Number of primary basis functions used in the calculation.

\\item[\\texttt{n\\_atoms}] Total number of atoms in the molecule.

\\item[\\texttt{n\\_carbons}] Number of carbon atoms.

\\item[\\texttt{n\\_hydrogens}] Number of hydrogen atoms.

\\item[\\texttt{n\\_scf\\_iterations}] Number of SCF iterations required for convergence.

\\item[\\texttt{total\\_time\\_s}] Total calculation time in seconds.

\\item[\\texttt{scf\\_time\\_s}] Time spent in SCF iterations in seconds.

\\item[\\texttt{mp2\\_time\\_s}] Time spent in MP2 calculation in seconds.

\\item[\\texttt{b\\_formation\\_time\\_s}] Time spent forming B matrices in seconds.

\\item[\\texttt{diag\\_time\\_s}] Time spent in matrix diagonalization in seconds.

\\item[\\texttt{ri\\_fock\\_time\\_s}] Time spent in RI-Fock matrix construction in seconds.

\\item[\\texttt{xc\\_time\\_s}] Time spent in exchange-correlation integration in seconds.

\\item[\\texttt{basis\\_transforms\\_time\\_s}] Time spent in basis set transformations in seconds.

\\item[\\texttt{total\\_tflop/s}] Total computational throughput in teraflops per second.

\\item[\\texttt{scf\\_tflop/s}] SCF computational throughput in teraflops per second.

\\item[\\texttt{mp2\\_tflop/s}] MP2 computational throughput in teraflops per second.

\\item[\\texttt{b\\_formation\\_tflop/s}] B matrix formation computational throughput in teraflops per second.

\\item[\\texttt{ri\\_fock\\_tflop/s}] RI-Fock computational throughput in teraflops per second.

\\item[\\texttt{xc\\_tflop/s}] Exchange-correlation computational throughput in teraflops per second.
\\end{description}

\\subsection{ORCA Dataset (\\texttt{orca\\_data.csv})}

The ORCA dataset contains results from calculations performed using ORCA v6.0.1 with the revDSD-PBEP86-D4(noFC) functional. This dataset includes calculations for COMPAS-3x and COMPAS-3D molecular systems across multiple basis set combinations.

\\subsubsection{Column Descriptions}

\\begin{description}
\\item[\\texttt{isomer}] Full name of the molecular isomer (e.g., \\texttt{compas3x\\_hc\\_c24h14\\_0pent\\_1}, \\texttt{compas3D\\_hc\\_c24h14\\_0pent\\_1}).

\\item[\\texttt{num\\_carbons}] Number of carbon atoms.

\\item[\\texttt{num\\_hydrogens}] Number of hydrogen atoms.

\\item[\\texttt{basis\\_combo\\_id}] Identifier for the basis set combination (e.g., \\texttt{qz\\_riri}, \\texttt{tz\\_rijk}).

\\item[\\texttt{primary\\_basis}] Primary basis set (e.g., \\texttt{def2-QZVPP}, \\texttt{def2-TZVP}, \\texttt{def2-SVP}).

\\item[\\texttt{scf\\_aux\\_basis}] Auxiliary basis set for SCF calculations (e.g., \\texttt{def2/JK}, \\texttt{def2-QZVPP/C}, or empty if not used).

\\item[\\texttt{ri\\_aux\\_basis}] Auxiliary basis set for RI-MP2 calculations (e.g., \\texttt{def2-QZVPP/C}, \\texttt{def2/JK}, or empty if not used).

\\item[\\texttt{filename}] Name of the ORCA output file.

\\item[\\texttt{multiplicity}] Spin multiplicity of the system.

\\item[\\texttt{charge}] Total charge of the system.

\\item[\\texttt{num\\_atoms}] Total number of atoms in the molecule.

\\item[\\texttt{num\\_basis\\_functions}] Number of basis functions used in the calculation.

\\item[\\texttt{total\\_energy\\_hartree}] Total electronic energy in hartree, including all corrections.

\\item[\\texttt{isomerization\\_energy\\_hartree}] Energy relative to the minimum energy isomer for the same (C, H) composition, calculated with the same basis set.

\\item[\\texttt{relative\\_energy\\_hartree}] Same as \\texttt{isomerization\\_energy\\_hartree} (relative to minimum isomer).

\\item[\\texttt{scf\\_energy\\_hartree}] Self-consistent field (SCF) energy in hartree.

\\item[\\texttt{dft\\_eexchange\\_hartree}] DFT exchange energy component in hartree.

\\item[\\texttt{dft\\_ecorr\\_hartree}] DFT correlation energy component in hartree.

\\item[\\texttt{dft\\_exc\\_hartree}] Total DFT exchange-correlation energy in hartree.

\\item[\\texttt{dft\\_finalen\\_hartree}] Final DFT energy (SCF + XC) in hartree.

\\item[\\texttt{mp2\\_ref\\_hartree}] MP2 reference energy (SCF energy) in hartree.

\\item[\\texttt{mp2\\_corr\\_hartree}] MP2 correlation correction in hartree.

\\item[\\texttt{mp2\\_total\\_hartree}] Total MP2 energy (reference + correlation) in hartree.

\\item[\\texttt{vdw\\_correction\\_hartree}] van der Waals (D4) dispersion correction in hartree.

\\item[\\texttt{scf\\_iterations}] Number of SCF iterations required for convergence.

\\item[\\texttt{startup\\_time\\_s}] Calculation startup time in seconds.

\\item[\\texttt{scf\\_time\\_s}] Time spent in SCF iterations in seconds.

\\item[\\texttt{property\\_time\\_s}] Time spent in property calculations in seconds.

\\item[\\texttt{rij\\_k\\_time\\_s}] Time spent in RI-JK calculations in seconds.

\\item[\\texttt{xc\\_integration\\_time\\_s}] Time spent in exchange-correlation integration in seconds.

\\item[\\texttt{mp2\\_time\\_s}] Time spent in MP2 calculation in seconds.

\\item[\\texttt{total\\_time\\_s}] Total calculation time in seconds.
\\end{description}

\\subsection{ORCA-EXESS Comparison Dataset (\\texttt{orca\\_exess\\_comparison\\_detailed.csv})}

This dataset contains a detailed comparison between ORCA and EXESS calculations for C24H14 isomers (26 total: 13 COMPAS-3x and 13 COMPAS-3D), including energy components and deviations. All calculations use the revDSD-PBEP86-D4 functional and def2-QZVPP basis set.

\\subsubsection{Column Descriptions}

\\begin{description}
\\item[\\texttt{common\\_id}] Common identifier extracted from isomer name (e.g., \\texttt{c24h14\\_0pent\\_1}).

\\item[\\texttt{prefix}] Dataset prefix: \\texttt{compas3x} for GFN2-xTB optimized geometries or \\texttt{compas3D} for CAM-B3LYP-D3BJ optimized geometries.

\\item[\\texttt{isomer\\_name}] Full name of the molecular isomer.

\\item[\\texttt{total\\_energy\\_hartree\\_exess}] Total energy from EXESS calculation in hartree.

\\item[\\texttt{total\\_energy\\_hartree\\_orca}] Total energy from ORCA calculation in hartree.

\\item[\\texttt{scf\\_energy\\_hartree\\_exess}] SCF energy from EXESS in hartree.

\\item[\\texttt{scf\\_energy\\_hartree\\_orca}] SCF energy from ORCA in hartree.

\\item[\\texttt{xc\\_energy\\_hartree}] Exchange-correlation energy from EXESS in hartree.

\\item[\\texttt{dft\\_exc\\_hartree}] Exchange-correlation energy from ORCA in hartree.

\\item[\\texttt{pt2\\_os\\_correction\\_hartree}] MP2 opposite-spin correction from EXESS in hartree.

\\item[\\texttt{pt2\\_ss\\_correction\\_hartree}] MP2 same-spin correction from EXESS in hartree.

\\item[\\texttt{exess\\_mp2\\_total\\_hartree}] Total MP2 correction from EXESS (OS + SS) in hartree.

\\item[\\texttt{mp2\\_corr\\_hartree}] MP2 correlation correction from ORCA in hartree.

\\item[\\texttt{exess\\_rel\\_energy\\_hartree}] Relative energy from EXESS (relative to minimum EXESS energy) in hartree.

\\item[\\texttt{orca\\_rel\\_energy\\_hartree}] Relative energy from ORCA (relative to minimum ORCA energy) in hartree.

\\item[\\texttt{abs\\_deviation\\_hartree}] Absolute deviation between EXESS and ORCA total energies in hartree.

\\item[\\texttt{abs\\_deviation\\_kjmol}] Absolute deviation between EXESS and ORCA total energies in kJ/mol.

\\item[\\texttt{rel\\_deviation\\_hartree}] Deviation in relative energies (isomerization energies) in hartree.

\\item[\\texttt{rel\\_deviation\\_kjmol}] Deviation in relative energies (isomerization energies) in kJ/mol.

\\item[\\texttt{scf\\_deviation\\_hartree}] Deviation in SCF energies in hartree.

\\item[\\texttt{scf\\_deviation\\_kjmol}] Deviation in SCF energies in kJ/mol.

\\item[\\texttt{xc\\_deviation\\_hartree}] Deviation in exchange-correlation energies in hartree.

\\item[\\texttt{xc\\_deviation\\_kjmol}] Deviation in exchange-correlation energies in kJ/mol.

\\item[\\texttt{mp2\\_deviation\\_hartree}] Deviation in MP2 correlation corrections in hartree.

\\item[\\texttt{mp2\\_deviation\\_kjmol}] Deviation in MP2 correlation corrections in kJ/mol.
\\end{description}

\\section{Summary Statistics}

\\input{""" + table_filename + """}

\\section{Comparison Plots}

This supporting information contains scatter plots comparing various DFT functionals to revDSD-PBEP86-D4(noFC)/def2-QZVPP for COMPAS-3x geometries. All calculations were performed with the (99,590) grid and def2-TZVP basis set. The left column shows comparisons without the D4 correction, and the right column shows comparisons with the D4 correction.

\\subsection{LDA Functionals}

"""
    
    # Add LDA functionals
    for func in LDA_FUNCTIONALS:
        latex_content += generate_functional_section(func, func)
    
    # Add GGA functionals
    latex_content += "\\FloatBarrier\n\\subsection{GGA Functionals}\n\n"
    for func in GGA_FUNCTIONALS:
        latex_content += generate_functional_section(func, func)
    
    # Add MGGA functionals
    latex_content += "\\FloatBarrier\n\\subsection{MGGA Functionals}\n\n"
    for func in MGGA_FUNCTIONALS:
        latex_content += generate_functional_section(func, func)
    
    # Add GFN2-xTB comparison
    if xtb_plot_exists:
        latex_content += """\\FloatBarrier
\\subsection{Semiempirical Methods}

\\FloatBarrier
\\needspace{15\\baselineskip}
\\subsubsection{GFN2-xTB}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.6\\textwidth]{plots/compas3x_xtb_vs_revdsd.png}
\\caption{GFN2-xTB comparison to revDSD-PBEP86-D4(noFC)/def2-QZVPP for COMPAS-3x geometries.}
\\label{fig:xtb}
\\end{figure}

"""
    
    latex_content += "\\end{document}\n"
    
    # Write the file
    with open(output_file, 'w', encoding='utf-8') as f:
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
    
    # Step 1: Generate plots with min reference method (no linear correction)
    print("Generating benchmark plots (using minimum reference method)...")
    benchmark_script = script_dir / "benchmark_compas3x.py"
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--exess-csv", str(exess_csv),
        "--output-dir", str(plots_dir),
        "--output", str(table_file.with_suffix('.txt')),
        "--reference-method", "min"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Failed to generate plots")
        sys.exit(1)
    
    # Save plots directory before second run (which will overwrite plots)
    plots_backup = script_dir / "plots_backup"
    if plots_backup.exists():
        shutil.rmtree(plots_backup)
    shutil.copytree(plots_dir, plots_backup)
    
    # Step 2: Generate table with linear_fit to get gradient/offset columns
    print("Generating statistics table with gradient and offset...")
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--exess-csv", str(exess_csv),
        "--output-dir", str(plots_dir),
        "--output", str(table_file.with_suffix('.txt')),
        "--reference-method", "linear_fit"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Failed to generate table")
        sys.exit(1)
    
    # Restore plots from min run (overwrite the linear_fit plots)
    print("Restoring plots with minimum reference method...")
    for plot_file in plots_backup.glob("compas3x_*_vs_revdsd.png"):
        shutil.copy2(plot_file, plots_dir / plot_file.name)
    
    # Clean up backup
    shutil.rmtree(plots_backup)
    
    # Step 3: Generate GFN2-xTB comparison plot
    generate_xtb_comparison_plot(exess_csv, plots_dir)
    
    # Step 4: Generate LaTeX file
    print("\nGenerating supporting_information.tex...")
    generate_latex_file(plots_dir, table_file, latex_file)
    
    # Step 5: Compile LaTeX document
    print("\nCompiling LaTeX document...")
    latex_dir = latex_file.parent
    latex_filename = latex_file.name
    
    # Run pdflatex twice to resolve references
    for run_num in [1, 2]:
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory", str(latex_dir),
            str(latex_filename)
        ]
        subprocess.run(cmd, cwd=str(latex_dir))
    
    # Clean up intermediate LaTeX files (but keep .tex files)
    intermediate_extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.fls', '.fdb_latexmk', '.synctex.gz']
    for ext in intermediate_extensions:
        intermediate_file = latex_file.with_suffix(ext)
        if intermediate_file.exists():
            intermediate_file.unlink()
    
    pdf_file = latex_file.with_suffix('.pdf')
    if pdf_file.exists():
        print(f"\nSUCCESS: Supporting information PDF generated at {pdf_file}")
    else:
        print(f"\nWARNING: PDF file not found. Check LaTeX compilation output.")


if __name__ == "__main__":
    main()
