#!/usr/bin/env python3
"""Compare EXESS revDSD-PBEP86-D4 energies with G4(MP2) energies from PAH335 dataset."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_utils import create_scatter_plot, HARTREE_TO_KJ_PER_MOL, calculate_stats


def extract_common_id_from_pah335(structure_name):
    """Extract common identifier from PAH335 structure name."""
    if pd.isna(structure_name) or not isinstance(structure_name, str):
        return None
    return structure_name.replace('.log', '')


def extract_pah_id_from_isomer(name):
    """Extract PAH ID from isomer name, removing PAH335_ prefix if present."""
    if pd.isna(name) or not isinstance(name, str):
        return None
    return name[7:] if name.startswith('PAH335_') else name


def main():
    parser = argparse.ArgumentParser(description="Compare EXESS revDSD-PBEP86-D4 energies with G4(MP2) energies")
    parser.add_argument("--exess-csv", default="analysis/exess_data.csv", help="EXESS data CSV file")
    parser.add_argument("--g4mp2-csv", default="PAH335_g4mp2_isomerization_energies.csv", help="G4(MP2) isomerization energies CSV file")
    parser.add_argument("--output", default="exess_g4mp2_comparison.txt", help="Output text file")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    exess_path = script_dir.parent / args.exess_csv if not Path(args.exess_csv).is_absolute() else Path(args.exess_csv)
    g4mp2_path = script_dir / args.g4mp2_csv if not Path(args.g4mp2_csv).is_absolute() else Path(args.g4mp2_csv)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    print(f"Loading G4(MP2) data from {g4mp2_path}...")
    g4mp2_df = pd.read_csv(g4mp2_path)
    print(f"Loaded {len(g4mp2_df)} rows")
    
    if not exess_path.exists():
        print(f"Error: EXESS data file not found at {exess_path}")
        return
    
    print(f"\nLoading EXESS data from {exess_path}...")
    exess_df = pd.read_csv(exess_path)
    print(f"Loaded {len(exess_df)} rows from CSV")
    
    # Filter to PAH335 data
    exess_df_to_use = exess_df[
        (exess_df['optimizer'] == 'G4(MP2)') | 
        (exess_df['isomer_name'].str.contains('PAH335', case=False, na=False))
    ].copy()
    print(f"Found {len(exess_df_to_use)} PAH335 structures in CSV")
    
    if len(exess_df_to_use) == 0:
        print("Warning: No PAH335 EXESS data found in CSV.")
        return
    
    # Extract common IDs
    g4mp2_df['common_id'] = g4mp2_df['PAH335_structure'].apply(extract_common_id_from_pah335)
    exess_df_to_use['common_id'] = exess_df_to_use['isomer_name'].apply(extract_pah_id_from_isomer)
    
    # Drop rows with None common_id
    g4mp2_df = g4mp2_df.dropna(subset=['common_id'])
    exess_df_to_use = exess_df_to_use.dropna(subset=['common_id'])
    
    print(f"\nG4(MP2) unique structures: {g4mp2_df['common_id'].nunique()}")
    print(f"EXESS unique structures: {exess_df_to_use['common_id'].nunique()}")
    
    # Merge on common_id
    merged = exess_df_to_use.merge(
        g4mp2_df[['common_id', 'g4mp2_isomerization_energy_kj/mol']],
        on='common_id', how='inner', suffixes=('_exess', '_g4mp2')
    )
    
    if len(merged) == 0:
        print("\nWarning: No matching structures found between EXESS and G4(MP2) data")
        print("\nSample G4(MP2) common_ids:", g4mp2_df['common_id'].head(10).tolist())
        if 'common_id' in exess_df_to_use.columns:
            print("\nSample EXESS common_ids:", exess_df_to_use['common_id'].head(10).tolist())
        return
    
    print(f"\nMatched {len(merged)} structures")
    
    # Convert energies to kJ/mol
    exess_energies_kjmol = merged['isomerization_energy_hartree'] * HARTREE_TO_KJ_PER_MOL
    g4mp2_energies_kjmol = merged['g4mp2_isomerization_energy_kj/mol']
    
    # Calculate statistics
    deviations = exess_energies_kjmol - g4mp2_energies_kjmol
    deviations_abs = np.abs(deviations)
    mad = np.mean(deviations_abs)
    rmsd = np.sqrt(np.mean(deviations ** 2))
    r_squared, _, mad_percentage = calculate_stats(exess_energies_kjmol, g4mp2_energies_kjmol)
    
    # Print results
    print("\n" + "="*60)
    print("EXESS revDSD-PBEP86-D4 vs G4(MP2) Comparison")
    print("="*60)
    print(f"\nNumber of matched structures: {len(merged)}")
    print(f"\nStatistics:")
    print(f"  MAD (Mean Absolute Deviation): {mad:.3f} kJ/mol")
    print(f"  RMSD (Root Mean Square Deviation): {rmsd:.3f} kJ/mol")
    print(f"  R² (Coefficient of Determination): {r_squared:.4f}")
    print(f"  MAD as percentage: {mad_percentage:.2f}%")
    
    # Find largest deviations
    print(f"\nTop 10 largest deviations:")
    top_indices = deviations_abs.nlargest(10).index
    for idx in top_indices:
        print(f"  {merged.loc[idx, 'common_id']}:")
        print(f"    EXESS: {exess_energies_kjmol.loc[idx]:.3f} kJ/mol")
        print(f"    G4(MP2): {g4mp2_energies_kjmol.loc[idx]:.3f} kJ/mol")
        print(f"    Deviation: {deviations.loc[idx]:.3f} kJ/mol (|{deviations_abs.loc[idx]:.3f}| kJ/mol)")
    
    # Create scatter plot
    create_scatter_plot(g4mp2_energies_kjmol, exess_energies_kjmol,
                       r'G4(MP2) $\Delta E$ (kJ/mol)', r'EXESS revDSD-PBEP86-D4 $\Delta E$ (kJ/mol)',
                       output_path.parent / 'exess_vs_g4mp2.png')
    
    # Save results to file
    with open(output_path, 'w') as f:
        f.write("EXESS revDSD-PBEP86-D4 vs G4(MP2) Comparison\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of matched structures: {len(merged)}\n\n")
        f.write("Statistics:\n")
        f.write(f"  MAD (Mean Absolute Deviation): {mad:.3f} kJ/mol\n")
        f.write(f"  RMSD (Root Mean Square Deviation): {rmsd:.3f} kJ/mol\n")
        f.write(f"  R² (Coefficient of Determination): {r_squared:.4f}\n")
        f.write(f"  MAD as percentage: {mad_percentage:.2f}%\n\n")
        f.write("Top 10 largest deviations:\n")
        for idx in top_indices:
            f.write(f"  {merged.loc[idx, 'common_id']}:\n")
            f.write(f"    EXESS: {exess_energies_kjmol.loc[idx]:.3f} kJ/mol\n")
            f.write(f"    G4(MP2): {g4mp2_energies_kjmol.loc[idx]:.3f} kJ/mol\n")
            f.write(f"    Deviation: {deviations.loc[idx]:.3f} kJ/mol (|{deviations_abs.loc[idx]:.3f}| kJ/mol)\n")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
