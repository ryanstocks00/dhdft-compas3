#!/usr/bin/env python3
"""Compare SVWN5 and revDSD-PBEP86-D4(noFC) results for COMPAS-3x geometries."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_utils import create_scatter_plot, HARTREE_TO_KJ_PER_MOL, calculate_stats, extract_common_id


def main():
    parser = argparse.ArgumentParser(description="Compare SVWN5 vs revDSD-PBEP86-D4(noFC) for COMPAS-3x")
    parser.add_argument("--exess-csv", default="analysis/exess_data.csv", help="EXESS data CSV file")
    parser.add_argument("--output", default="svwn5_vs_revdsd_compas3x_comparison.txt", help="Output text file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / args.exess_csv if not Path(args.exess_csv).is_absolute() else Path(args.exess_csv)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: EXESS data file not found at {csv_path}")
        return
    
    print(f"Loading EXESS data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Filter to COMPAS-3x geometries
    compas3x_df = df[df['isomer_name'].str.contains('compas3x', case=False, na=False)].copy()
    print(f"Found {len(compas3x_df)} COMPAS-3x rows")
    
    if len(compas3x_df) == 0:
        print("Error: No COMPAS-3x data found in CSV")
        return
    
    # Extract common IDs
    compas3x_df['common_id'] = compas3x_df['isomer_name'].apply(extract_common_id)
    compas3x_df = compas3x_df.dropna(subset=['common_id'])
    
    # Filter to xTB optimizer (COMPAS-3x uses xTB geometries)
    compas3x_df = compas3x_df[compas3x_df['optimizer'] == 'xTB'].copy()
    print(f"Found {len(compas3x_df)} COMPAS-3x rows with xTB optimizer")
    
    # Separate SVWN5 and revDSD-PBEP86-D4 data
    svwn5_df = compas3x_df[compas3x_df['functional'] == 'SVWN5'].copy()
    revdsd_df = compas3x_df[compas3x_df['functional'] == 'revDSD-PBEP86-D4'].copy()
    
    print(f"\nSVWN5 rows: {len(svwn5_df)}")
    print(f"revDSD-PBEP86-D4 rows: {len(revdsd_df)}")
    
    if len(svwn5_df) == 0:
        print("Error: No SVWN5 data found for COMPAS-3x. Please run extract_exess_data.py first.")
        return
    
    if len(revdsd_df) == 0:
        print("Error: No revDSD-PBEP86-D4 data found for COMPAS-3x")
        return
    
    # Check basis sets
    svwn5_basis = svwn5_df['basis_set'].unique()
    revdsd_basis = revdsd_df['basis_set'].unique()
    print(f"\nSVWN5 basis sets: {svwn5_basis}")
    print(f"revDSD-PBEP86-D4 basis sets: {revdsd_basis}")
    
    # Merge on common_id
    merged = svwn5_df[['common_id', 'isomerization_energy_hartree', 'basis_set']].merge(
        revdsd_df[['common_id', 'isomerization_energy_hartree', 'basis_set']],
        on='common_id',
        how='inner',
        suffixes=('_svwn5', '_revdsd')
    )
    
    if len(merged) == 0:
        print("Error: No matching structures found between SVWN5 and revDSD-PBEP86-D4")
        return
    
    print(f"\nMatched {len(merged)} structures")
    
    # Convert energies to kJ/mol
    svwn5_energies_kjmol = merged['isomerization_energy_hartree_svwn5'] * HARTREE_TO_KJ_PER_MOL
    revdsd_energies_kjmol = merged['isomerization_energy_hartree_revdsd'] * HARTREE_TO_KJ_PER_MOL
    
    # Calculate statistics
    deviations = svwn5_energies_kjmol - revdsd_energies_kjmol
    deviations_abs = np.abs(deviations)
    mad_kjmol = np.mean(deviations_abs)
    rmsd = np.sqrt(np.mean(deviations ** 2))
    r_squared, _, mad_percentage = calculate_stats(revdsd_energies_kjmol, svwn5_energies_kjmol)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"SVWN5 vs revDSD-PBEP86-D4(noFC) Comparison (COMPAS-3x)")
    print(f"{'='*60}")
    print(f"\nStatistics:")
    print(f"  Number of matched structures: {len(merged)}")
    print(f"  MAD (Mean Absolute Deviation): {mad_kjmol:.3f} kJ/mol")
    print(f"  RMSD (Root Mean Square Deviation): {rmsd:.3f} kJ/mol")
    print(f"  R² (Coefficient of Determination): {r_squared:.4f}")
    print(f"  MAD as percentage: {mad_percentage:.2f}%")
    
    # Find largest deviations
    print(f"\nTop 10 largest deviations:")
    top_indices = deviations_abs.nlargest(10).index
    for idx in top_indices:
        print(f"  {merged.loc[idx, 'common_id']}:")
        print(f"    SVWN5: {svwn5_energies_kjmol.loc[idx]:.3f} kJ/mol")
        print(f"    revDSD-PBEP86-D4(noFC): {revdsd_energies_kjmol.loc[idx]:.3f} kJ/mol")
        print(f"    Deviation: {deviations.loc[idx]:.3f} kJ/mol (|{deviations_abs.loc[idx]:.3f}| kJ/mol)")
    
    # Create scatter plot
    plot_path = output_dir / 'svwn5_vs_revdsd_compas3x.png'
    create_scatter_plot(
        revdsd_energies_kjmol,
        svwn5_energies_kjmol,
        r'revDSD-PBEP86-D4(noFC) $\Delta E$ (kJ/mol)',
        r'SVWN5 $\Delta E$ (kJ/mol)',
        plot_path,
        mad_kjmol=mad_kjmol
    )
    print(f"\nPlot saved to: {plot_path}")
    
    # Save results to file
    with open(output_path, 'w') as f:
        f.write("SVWN5 vs revDSD-PBEP86-D4(noFC) Comparison (COMPAS-3x)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of matched structures: {len(merged)}\n\n")
        f.write("Statistics:\n")
        f.write(f"  MAD (Mean Absolute Deviation): {mad_kjmol:.3f} kJ/mol\n")
        f.write(f"  RMSD (Root Mean Square Deviation): {rmsd:.3f} kJ/mol\n")
        f.write(f"  R² (Coefficient of Determination): {r_squared:.4f}\n")
        f.write(f"  MAD as percentage: {mad_percentage:.2f}%\n\n")
        
        # Find top deviations
        top_indices = deviations_abs.nlargest(10).index
        f.write("Top 10 largest deviations:\n")
        for idx in top_indices:
            f.write(f"  {merged.loc[idx, 'common_id']}:\n")
            f.write(f"    SVWN5: {svwn5_energies_kjmol.loc[idx]:.3f} kJ/mol\n")
            f.write(f"    revDSD-PBEP86-D4(noFC): {revdsd_energies_kjmol.loc[idx]:.3f} kJ/mol\n")
            f.write(f"    Deviation: {deviations.loc[idx]:.3f} kJ/mol\n")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()




