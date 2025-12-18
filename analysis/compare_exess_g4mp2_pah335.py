#!/usr/bin/env python3
"""Compare EXESS energies (revDSD-PBEP86-D4 and PBE0) with G4(MP2) energies from PAH335 dataset."""

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


def process_functional_comparison(exess_df, g4mp2_df, functional, basis_sets=None, plots_dir=None, output_path=None):
    """Process comparison for a specific functional.
    
    Args:
        exess_df: Full EXESS dataframe
        g4mp2_df: G4(MP2) dataframe with common_id column
        functional: Functional name ('revDSD-PBEP86-D4' or 'PBE0')
        basis_sets: List of basis sets to process, or None for all
        plots_dir: Directory to save plots
        output_path: Path for text output file
    
    Returns:
        List of result dictionaries
    """
    # Filter to functional and PAH335 data
    exess_filtered = exess_df[
        (exess_df['functional'] == functional) &
        (exess_df['optimizer'] == 'G4(MP2)') &
        (exess_df['isomer_name'].str.contains('PAH335', case=False, na=False))
    ].copy()
    
    if len(exess_filtered) == 0:
        print(f"Warning: No {functional} PAH335 EXESS data found.")
        return []
    
    # Extract common IDs
    exess_filtered['common_id'] = exess_filtered['isomer_name'].apply(extract_pah_id_from_isomer)
    exess_filtered = exess_filtered.dropna(subset=['common_id'])
    
    # Determine which basis sets to process
    available_basis = exess_filtered['basis_set'].unique()
    if basis_sets is None:
        basis_sets = available_basis
    else:
        basis_sets = [bs for bs in basis_sets if bs in available_basis]
    
    if len(basis_sets) == 0:
        print(f"Warning: No matching basis sets found for {functional}")
        return []
    
    all_results = []
    
    for basis_set in basis_sets:
        print(f"\n{'='*60}")
        print(f"Processing {functional} / {basis_set}")
        print(f"{'='*60}")
        
        # Filter to current basis set
        exess_basis = exess_filtered[exess_filtered['basis_set'] == basis_set].copy()
        
        # Merge on common_id
        merged = exess_basis.merge(
            g4mp2_df[['common_id', 'g4mp2_isomerization_energy_kj/mol']],
            on='common_id', how='inner', suffixes=('_exess', '_g4mp2')
        )
        
        if len(merged) == 0:
            print(f"Warning: No matching structures found for {functional} / {basis_set}")
            continue
        
        print(f"Matched {len(merged)} structures")
        
        # Convert energies to kJ/mol
        exess_energies_kjmol = merged['isomerization_energy_hartree'] * HARTREE_TO_KJ_PER_MOL
        g4mp2_energies_kjmol = merged['g4mp2_isomerization_energy_kj/mol']
        
        # Calculate statistics
        deviations = exess_energies_kjmol - g4mp2_energies_kjmol
        deviations_abs = np.abs(deviations)
        mad_kjmol = np.mean(deviations_abs)
        rmsd = np.sqrt(np.mean(deviations ** 2))
        r_squared, _, mad_percentage = calculate_stats(exess_energies_kjmol, g4mp2_energies_kjmol)
        
        # Print results
        print(f"\nStatistics:")
        print(f"  MAD (Mean Absolute Deviation): {mad_kjmol:.3f} kJ/mol")
        print(f"  RMSD (Root Mean Square Deviation): {rmsd:.3f} kJ/mol")
        print(f"  R² (Coefficient of Determination): {r_squared:.4f}")
        print(f"  MAD as percentage: {mad_percentage:.2f}%")
        
        # Find largest deviations
        print(f"\nTop 10 largest deviations:")
        top_indices = deviations_abs.nlargest(10).index
        for idx in top_indices:
            print(f"  {merged.loc[idx, 'common_id']}:")
            print(f"    {functional}: {exess_energies_kjmol.loc[idx]:.3f} kJ/mol")
            print(f"    G4(MP2): {g4mp2_energies_kjmol.loc[idx]:.3f} kJ/mol")
            print(f"    Deviation: {deviations.loc[idx]:.3f} kJ/mol (|{deviations_abs.loc[idx]:.3f}| kJ/mol)")
        
        # Create scatter plot
        if plots_dir:
            basis_suffix = basis_set.replace('def2-', '').lower()
            func_suffix = functional.lower().replace('-', '_').replace('(', '').replace(')', '')
            plot_path = plots_dir / f'{func_suffix}_{basis_suffix}_vs_g4mp2.png'
            
            # Format functional name for plot
            if functional == 'revDSD-PBEP86-D4':
                func_label = 'revDSD-PBEP86-D4'
            else:
                func_label = functional
            
            create_scatter_plot(g4mp2_energies_kjmol, exess_energies_kjmol,
                               r'G4(MP2) $\Delta E$ (kJ/mol)', 
                               f'{func_label}/{basis_set} $\Delta E$ (kJ/mol)',
                               plot_path, mad_kjmol=mad_kjmol)
        
        # Store results
        all_results.append({
            'functional': functional,
            'basis_set': basis_set,
            'n_structures': len(merged),
            'mad_kjmol': mad_kjmol,
            'rmsd': rmsd,
            'r_squared': r_squared,
            'mad_percentage': mad_percentage,
            'merged': merged,
            'exess_energies': exess_energies_kjmol,
            'g4mp2_energies': g4mp2_energies_kjmol,
            'deviations': deviations
        })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare EXESS energies with G4(MP2) energies")
    parser.add_argument("--exess-csv", default="analysis/exess_data.csv", help="EXESS data CSV file")
    parser.add_argument("--g4mp2-csv", default="PAH335_g4mp2_isomerization_energies.csv", help="G4(MP2) isomerization energies CSV file")
    parser.add_argument("--output", default="exess_g4mp2_comparison.txt", help="Output text file")
    parser.add_argument("--functional", choices=["revDSD-PBEP86-D4", "PBE0", "both"], default="both",
                       help="Which functional(s) to compare (default: both)")
    parser.add_argument("--basis", nargs="+", default=None,
                       help="Which basis set(s) to compare (default: all available)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    exess_path = script_dir.parent / args.exess_csv if not Path(args.exess_csv).is_absolute() else Path(args.exess_csv)
    g4mp2_path = script_dir / args.g4mp2_csv if not Path(args.g4mp2_csv).is_absolute() else Path(args.g4mp2_csv)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    plots_dir = output_path.parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading G4(MP2) data from {g4mp2_path}...")
    g4mp2_df = pd.read_csv(g4mp2_path)
    print(f"Loaded {len(g4mp2_df)} rows")
    
    if not exess_path.exists():
        print(f"Error: EXESS data file not found at {exess_path}")
        return
    
    print(f"\nLoading EXESS data from {exess_path}...")
    exess_df = pd.read_csv(exess_path)
    print(f"Loaded {len(exess_df)} rows from CSV")
    
    # Extract common IDs for G4(MP2)
    g4mp2_df['common_id'] = g4mp2_df['PAH335_structure'].apply(extract_common_id_from_pah335)
    g4mp2_df = g4mp2_df.dropna(subset=['common_id'])
    print(f"G4(MP2) unique structures: {g4mp2_df['common_id'].nunique()}")
    
    # Determine which functionals to process
    functionals = []
    if args.functional == "both":
        functionals = ["revDSD-PBEP86-D4", "PBE0"]
    else:
        functionals = [args.functional]
    
    all_results = []
    
    # Process each functional
    for functional in functionals:
        results = process_functional_comparison(
            exess_df, g4mp2_df, functional, 
            basis_sets=args.basis, 
            plots_dir=plots_dir,
            output_path=output_path
        )
        all_results.extend(results)
    
    if len(all_results) == 0:
        print("\nNo results to save.")
        return
    
    # Save results to file
    with open(output_path, 'w') as f:
        f.write("EXESS vs G4(MP2) Comparison (PAH335 Dataset)\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            functional = result['functional']
            basis_set = result['basis_set']
            f.write(f"\n{functional} / {basis_set}\n")
            f.write("-"*60 + "\n")
            f.write(f"Number of matched structures: {result['n_structures']}\n\n")
            f.write("Statistics:\n")
            f.write(f"  MAD (Mean Absolute Deviation): {result['mad_kjmol']:.3f} kJ/mol\n")
            f.write(f"  RMSD (Root Mean Square Deviation): {result['rmsd']:.3f} kJ/mol\n")
            f.write(f"  R² (Coefficient of Determination): {result['r_squared']:.4f}\n")
            f.write(f"  MAD as percentage: {result['mad_percentage']:.2f}%\n\n")
            
            # Find top deviations
            top_indices = result['deviations'].abs().nlargest(10).index
            f.write("Top 10 largest deviations:\n")
            for idx in top_indices:
                f.write(f"  {result['merged'].loc[idx, 'common_id']}:\n")
                f.write(f"    {functional}: {result['exess_energies'].loc[idx]:.3f} kJ/mol\n")
                f.write(f"    G4(MP2): {result['g4mp2_energies'].loc[idx]:.3f} kJ/mol\n")
                f.write(f"    Deviation: {result['deviations'].loc[idx]:.3f} kJ/mol\n")
            f.write("\n")
    
    print(f"\nResults saved to {output_path}")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()






