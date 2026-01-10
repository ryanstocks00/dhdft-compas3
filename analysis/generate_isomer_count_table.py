#!/usr/bin/env python3

"""
Script to generate a LaTeX table showing the number of isomers for each molecule
in the COMPAS-3x and COMPAS-3D datasets.
"""

import argparse
import pandas as pd
from pathlib import Path
import re


def extract_molecule_id(isomer_name):
    """
    Extract molecule identifier from isomer name.
    
    Examples:
    - compas3x_hc_c16h10_0pent_1 -> c16h10_0pent
    - compas3D_hc_c20h12_0pent_2 -> c20h12_0pent
    """
    # Remove prefix (compas3x_ or compas3D_)
    if isomer_name.startswith('compas3x_'):
        name = isomer_name[9:]  # Remove 'compas3x_'
    elif isomer_name.startswith('compas3D_'):
        name = isomer_name[9:]  # Remove 'compas3D_'
    else:
        return None
    
    # Remove 'hc_' prefix if present
    if name.startswith('hc_'):
        name = name[3:]
    
    # Remove the last underscore and number (isomer number)
    # e.g., c16h10_0pent_1 -> c16h10_0pent
    match = re.match(r'^(.+)_\d+$', name)
    if match:
        return match.group(1)
    return name


def count_isomers_by_molecule(df, dataset_prefix):
    """
    Count isomers for each molecule in a dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with isomer_name column
    dataset_prefix : str
        Prefix to filter by ('compas3x' or 'compas3D')
    
    Returns:
    --------
    pd.Series
        Series with molecule_id as index and isomer count as values
    """
    # Filter to the specified dataset
    filtered = df[df['isomer_name'].str.startswith(dataset_prefix + '_')].copy()
    
    # Extract molecule IDs
    filtered['molecule_id'] = filtered['isomer_name'].apply(extract_molecule_id)
    
    # Count isomers per molecule
    counts = filtered.groupby('molecule_id').size()
    
    return counts


def extract_molecule_sort_key(mol_id):
    """
    Extract sort key (n_carbons, n_hydrogens) from molecule ID.
    Returns tuple (n_c, n_h) for sorting.
    """
    match = re.match(r'^c(\d+)h(\d+)', mol_id)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def format_molecule_name(mol_id):
    """
    Format molecule ID for LaTeX display.
    e.g., c16h10_0pent -> C_{16}H_{10} (0 pentagons)
    """
    # Extract carbon and hydrogen counts
    match = re.match(r'^c(\d+)h(\d+)(?:_(\d+)pent)?', mol_id)
    if match:
        n_c = match.group(1)
        n_h = match.group(2)
        n_pent = match.group(3) if match.group(3) else '0'
        
        # Format as C_{n}H_{m} with pentagon count
        if n_pent == '0':
            return f"\\ce{{C_{{{n_c}}}H_{{{n_h}}}}}"
        else:
            return f"\\ce{{C_{{{n_c}}}H_{{{n_h}}}}} ({n_pent} pent)"
    
    # Fallback: just return the ID with underscores escaped
    return mol_id.replace('_', '\\_')


def generate_latex_table(compas3x_counts, compas3d_counts):
    """
    Generate a LaTeX table comparing isomer counts between COMPAS-3x and COMPAS-3D.
    """
    # Get all unique molecules from both datasets
    all_molecules = list(set(compas3x_counts.index) | set(compas3d_counts.index))
    
    # Sort by number of carbons, then hydrogens
    all_molecules.sort(key=extract_molecule_sort_key)
    
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{lS[table-format=5.0]S[table-format=5.0]}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Molecule} & {\\textbf{COMPAS-3x}} & {\\textbf{COMPAS-3D}} \\\\")
    latex_lines.append("\\midrule")
    
    total_3x = 0
    total_3d = 0
    
    for mol_id in all_molecules:
        mol_name = format_molecule_name(mol_id)
        count_3x = compas3x_counts.get(mol_id, 0)
        count_3d = compas3d_counts.get(mol_id, 0)
        total_3x += count_3x
        total_3d += count_3d
        
        latex_lines.append(f"{mol_name} & {count_3x} & {count_3d} \\\\")
    
    latex_lines.append("\\midrule")
    latex_lines.append(f"\\textbf{{Total}} & \\textbf{{{total_3x}}} & \\textbf{{{total_3d}}} \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Number of isomers for each molecule in the COMPAS-3x and COMPAS-3D datasets.}")
    latex_lines.append("\\label{tab:isomer_counts}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table of isomer counts for COMPAS-3x and COMPAS-3D datasets"
    )
    parser.add_argument(
        "--input",
        default="COMPAS3_EXESS_data.csv",
        help="Input CSV file (default: COMPAS3_EXESS_data.csv)"
    )
    parser.add_argument(
        "--output",
        default="isomer_counts_table.tex",
        help="Output LaTeX file (default: isomer_counts_table.tex)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    input_path = script_dir.parent / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    print("\nCounting isomers for COMPAS-3x...")
    compas3x_counts = count_isomers_by_molecule(df, 'compas3x')
    print(f"Found {len(compas3x_counts)} unique molecules in COMPAS-3x")
    print(f"Total isomers in COMPAS-3x: {compas3x_counts.sum()}")
    
    print("\nCounting isomers for COMPAS-3D...")
    compas3d_counts = count_isomers_by_molecule(df, 'compas3D')
    print(f"Found {len(compas3d_counts)} unique molecules in COMPAS-3D")
    print(f"Total isomers in COMPAS-3D: {compas3d_counts.sum()}")
    
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(compas3x_counts, compas3d_counts)
    
    output_path.write_text(latex_table)
    print(f"\nSaved LaTeX table to {output_path}")
    print("\nTable preview:")
    print(latex_table)


if __name__ == "__main__":
    main()

