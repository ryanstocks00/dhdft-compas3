#!/usr/bin/env python3
"""Benchmark SVWN5 and GGAs against revDSD-PBEP86-D4(noFC) for COMPAS-3x geometries."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from plotting_utils import create_scatter_plot, HARTREE_TO_KJ_PER_MOL, calculate_stats, extract_common_id, format_functional_name


def compute_isomerization_energies(df, include_d4, reference_method):
    """Compute isomerization energies from a dataframe.
    
    Args:
        df: Dataframe with energy columns
        include_d4: If True, use total_energy_hartree, else use scf_energy_hartree
        reference_method: 'min' or 'avg' for reference energy calculation
    
    Returns:
        Series with isomerization energies indexed by common_id
    """
    energy_col = 'total_energy_hartree' if include_d4 else 'scf_energy_hartree'
    df = df.copy()
    df['energy'] = df[energy_col]
    
    if df['energy'].isna().any():
        missing = df[df['energy'].isna()]['common_id'].tolist()
        raise ValueError(f"Missing {energy_col} for structures: {missing[:10]}")
    
    if df[['n_carbons', 'n_hydrogens']].isna().any().any():
        raise ValueError("Missing n_carbons or n_hydrogens")
    
    group_cols = ['functional', 'basis_set', 'n_carbons', 'n_hydrogens']
    agg_method = 'mean' if reference_method == 'avg' else reference_method
    ref_energy = df.groupby(group_cols)['energy'].agg(agg_method).to_dict()
    df['ref_energy'] = df.apply(lambda r: ref_energy.get(tuple(r[c] for c in group_cols)), axis=1)
    
    if df['ref_energy'].isna().any():
        missing = df[df['ref_energy'].isna()][['common_id'] + group_cols].drop_duplicates()
        raise ValueError(f"Could not find reference energy:\n{missing.head(10)}")
    
    df['isomerization_energy'] = df['energy'] - df['ref_energy']
    return df.set_index('common_id')['isomerization_energy']


def find_minimum_energy_isomers(df, include_d4):
    """Find the minimum energy isomer for each C/H group.
    
    Args:
        df: Dataframe with energy columns and common_id
        include_d4: If True, use total_energy_hartree, else use scf_energy_hartree
    
    Returns:
        Dictionary mapping (n_carbons, n_hydrogens) -> common_id of minimum energy isomer
    """
    energy_col = 'total_energy_hartree' if include_d4 else 'scf_energy_hartree'
    df = df.copy()
    df['energy'] = df[energy_col]
    
    # Group by C/H and find minimum energy isomer
    min_isomers = {}
    for (n_c, n_h), group in df.groupby(['n_carbons', 'n_hydrogens']):
        min_idx = group['energy'].idxmin()
        min_isomer = group.loc[min_idx, 'common_id']
        min_isomers[(n_c, n_h)] = min_isomer
    
    return min_isomers


def process_functional_comparison(df, functional, reference_functional='revDSD-PBEP86-D4',
                                  include_d4=True, output_dir=None, reference_method='min'):
    """Process comparison for a specific functional."""
    # Filter to COMPAS-3x geometries
    compas3x_df = df[
        (df['isomer_name'].str.contains('compas3x', case=False, na=False)) &
        (df['optimizer'] == 'xTB')
    ].copy()
    compas3x_df['common_id'] = compas3x_df['isomer_name'].apply(extract_common_id)
    compas3x_df = compas3x_df.dropna(subset=['common_id'])
    
    func_df = compas3x_df[compas3x_df['functional'] == functional].copy()
    ref_df = compas3x_df[compas3x_df['functional'] == reference_functional].copy()
    
    if len(func_df) == 0 or len(ref_df) == 0:
        return None
    
    # Find common structures
    common_ids = set(func_df['common_id'].unique()) & set(ref_df['common_id'].unique())
    if len(common_ids) == 0:
        raise ValueError(f"No matching structures found for {functional} vs {reference_functional}")
    
    func_df_common = func_df[func_df['common_id'].isin(common_ids)].copy()
    ref_df_common = ref_df[ref_df['common_id'].isin(common_ids)].copy()
    
    # For linear fit, we still need to compute isomerization energies first (use 'min' as base)
    # The linear fit correction will be applied to the isomerization energies afterward
    base_reference_method = 'min' if reference_method == 'linear_fit' else reference_method
    
    # Check if minimum energy isomers differ (only for 'min' method)
    if base_reference_method == 'min':
        func_min_isomers = find_minimum_energy_isomers(func_df_common, include_d4)
        ref_min_isomers = find_minimum_energy_isomers(ref_df_common, True)  # Always use total_energy for reference
        
        # Find C/H groups where minimum energy isomers differ
        differing_groups = []
        for ch_key in func_min_isomers:
            if ch_key in ref_min_isomers:
                if func_min_isomers[ch_key] != ref_min_isomers[ch_key]:
                    differing_groups.append({
                        'n_c': ch_key[0],
                        'n_h': ch_key[1],
                        'func_min': func_min_isomers[ch_key],
                        'ref_min': ref_min_isomers[ch_key]
                    })
        
        if differing_groups:
            print(f"\n⚠️  WARNING: Minimum energy isomers differ between {functional} and {reference_functional}:")
            for group in differing_groups[:10]:  # Show first 10
                print(f"  C{group['n_c']}H{group['n_h']}: {functional} min = {group['func_min']}, "
                      f"{reference_functional} min = {group['ref_min']}")
            if len(differing_groups) > 10:
                print(f"  ... and {len(differing_groups) - 10} more groups")
    
    # Compute isomerization energies
    func_energies = compute_isomerization_energies(func_df_common, include_d4, base_reference_method)
    ref_energies = compute_isomerization_energies(ref_df_common, True, base_reference_method)  # Always use total_energy for reference
    
    # Merge to get common structures
    merged = func_df_common[['common_id']].merge(
        ref_df_common[['common_id']], on='common_id', how='inner'
    )
    merged = merged.merge(func_energies.to_frame('isomerization_energy_func'), left_on='common_id', right_index=True)
    merged = merged.merge(ref_energies.to_frame('isomerization_energy_ref'), left_on='common_id', right_index=True)
    
    if merged[['isomerization_energy_func', 'isomerization_energy_ref']].isna().any().any():
        raise ValueError("NaN isomerization energies found")
    
    # Convert to kJ/mol
    func_energies_kjmol = merged['isomerization_energy_func'] * HARTREE_TO_KJ_PER_MOL
    ref_energies_kjmol = merged['isomerization_energy_ref'] * HARTREE_TO_KJ_PER_MOL
    
    # Apply linear fit correction if requested
    gradient = None
    offset = None
    if reference_method == 'linear_fit':
        # Fit: func = gradient * ref + offset
        slope, intercept, r_value, p_value, std_err = linregress(ref_energies_kjmol, func_energies_kjmol)
        gradient = slope
        offset = intercept
        # Correct: func_corrected = (func - offset) / gradient
        func_energies_kjmol = (func_energies_kjmol - offset) / gradient
    
    deviations = func_energies_kjmol - ref_energies_kjmol
    
    mad_kjmol = np.mean(np.abs(deviations))
    rmsd = np.sqrt(np.mean(deviations ** 2))
    r_squared, _, mad_percentage = calculate_stats(ref_energies_kjmol, func_energies_kjmol)
    
    # Print results
    d4_label = "with D4" if include_d4 else "without D4"
    if reference_method == 'linear_fit':
        ref_method_label = "linear fit corrected"
    elif reference_method == 'min':
        ref_method_label = "minimum"
    else:
        ref_method_label = "average"
    print(f"\n{'='*60}")
    print(f"{functional} ({d4_label}, relative to {ref_method_label}) vs {reference_functional} (COMPAS-3x)")
    print(f"{'='*60}")
    print(f"Number of matched structures: {len(merged)}")
    if gradient is not None and offset is not None:
        print(f"  Linear fit: gradient = {gradient:.4f}, offset = {offset:.3f} kJ/mol")
    print(f"  MAD: {mad_kjmol:.3f} kJ/mol")
    print(f"  RMSD: {rmsd:.3f} kJ/mol")
    print(f"  R²: {r_squared:.4f}")
    print(f"  MAD as percentage: {mad_percentage:.2f}%")
    
    # Top deviations
    print(f"\nTop 10 largest deviations:")
    top_indices = deviations.abs().nlargest(10).index
    for idx in top_indices:
        print(f"  {merged.loc[idx, 'common_id']}: {deviations.loc[idx]:.3f} kJ/mol")
    
    # Create scatter plot
    if output_dir:
        func_safe = functional.replace('-', '_').replace('(', '').replace(')', '').replace('/', '_')
        d4_suffix = "_with_d4" if include_d4 else "_without_d4"
        plot_path = output_dir / f'compas3x_{func_safe}{d4_suffix}_vs_revdsd.png'
        # Format functional names for display
        func_display = format_functional_name(functional)
        ref_display = format_functional_name(reference_functional)
        create_scatter_plot(
            ref_energies_kjmol, func_energies_kjmol,
            f'{ref_display} $\Delta E$ (kJ/mol)',
            f'{func_display} ({d4_label}) $\Delta E$ (kJ/mol)',
            plot_path, mad_kjmol=mad_kjmol
        )
        print(f"\nPlot saved to: {plot_path}")
    
    return {
        'functional': functional, 'include_d4': include_d4, 'n_structures': len(merged),
        'mad_kjmol': mad_kjmol, 'rmsd': rmsd, 'r_squared': r_squared, 'mad_percentage': mad_percentage,
        'merged': merged, 'func_energies': func_energies_kjmol, 'ref_energies': ref_energies_kjmol,
        'deviations': deviations, 'gradient': gradient, 'offset': offset
    }


def generate_latex_table(results, output_path):
    """Generate a LaTeX table with MAD values for each functional."""
    # Organize results by functional
    func_data = {}
    for result in results:
        func = result['functional']
        if func not in func_data:
            func_data[func] = {}
        key = 'with_d4' if result['include_d4'] else 'without_d4'
        func_data[func][key] = {
            'mad': result['mad_kjmol'], 'r_squared': result['r_squared'],
            'gradient': result.get('gradient'), 'offset': result.get('offset')
        }
    
    # Find best values
    all_mads = [d[k]['mad'] for d in func_data.values() for k in d]
    all_r2s = [d[k]['r_squared'] for d in func_data.values() for k in d]
    best_mad = min(all_mads) if all_mads else None
    best_r2 = max(all_r2s) if all_r2s else None
    
    # Define functional categories and sort
    lda_functionals = ['SVWN5']
    gga_functionals = ['PBE', 'BLYP', 'revPBE', 'BP86', 'BPW91', 'B97-D', 'HCTH407']
    mgga_functionals = ['TPSS', 'MN15L', 'SCAN', 'rSCAN', 'r2SCAN', 'revTPSS', 't-HCTH', 'M06-L', 'M11-L']
    
    def get_mad_for_ordering(func):
        data = func_data[func]
        if func in lda_functionals:
            return data.get('without_d4', {}).get('mad', float('inf'))
        return data.get('with_d4', {}).get('mad') or data.get('without_d4', {}).get('mad', float('inf'))
    
    functional_order = []
    for category in [lda_functionals, gga_functionals, mgga_functionals]:
        functional_order.extend(sorted([f for f in category if f in func_data],
                                      key=get_mad_for_ordering, reverse=True))
    
    # Write LaTeX table
    with open(output_path, 'w') as f:
        f.write("% Requires: \\usepackage{booktabs, multirow, rotating, graphicx}\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        # Check if we have linear fit data (gradient/offset)
        has_linear_fit = any(
            d.get('with_d4', {}).get('gradient') is not None or 
            d.get('without_d4', {}).get('gradient') is not None
            for d in func_data.values()
        )
        
        if has_linear_fit:
            f.write("\\begin{tabular}{@{}c@{\\hspace{0.8em}}l@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{}}\n")
            f.write("\\toprule\n")
            f.write(" & \\multirow{3}{*}{\\textbf{Functional}} & \\multicolumn{3}{c}{\\textbf{Without D4}} & \\multicolumn{3}{c}{\\textbf{With D4}} \\\\\n")
            f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n")
            f.write(" & & \\shortstack{\\textbf{MAD}\\\\\\textbf{(kJ/mol)}} & \\textbf{R²} & \\shortstack{\\textbf{Gradient/}\\\\\\textbf{Offset}} & \\shortstack{\\textbf{MAD}\\\\\\textbf{(kJ/mol)}} & \\textbf{R²} & \\shortstack{\\textbf{Gradient/}\\\\\\textbf{Offset}} \\\\\n")
        else:
            f.write("\\begin{tabular}{@{}c@{\\hspace{0.8em}}l@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{\\hspace{0.3em}}c@{}}\n")
            f.write("\\toprule\n")
            f.write(" & \\multirow{3}{*}{\\textbf{Functional}} & \\multicolumn{2}{c}{\\textbf{Without D4}} & \\multicolumn{2}{c}{\\textbf{With D4}} \\\\\n")
            f.write("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n")
            f.write(" & & \\shortstack{\\textbf{MAD}\\\\\\textbf{(kJ/mol)}} & \\textbf{R²} & \\shortstack{\\textbf{MAD}\\\\\\textbf{(kJ/mol)}} & \\textbf{R²} \\\\\n")
        f.write("\\midrule\n")
        
        current_category = None
        category_counts = {'LDA': sum(1 for f in lda_functionals if f in func_data),
                          'GGA': sum(1 for f in gga_functionals if f in func_data),
                          'MGGA': sum(1 for f in mgga_functionals if f in func_data)}
          
        for func in functional_order:
            if func not in func_data:
                continue
            
            if func in lda_functionals:
                category = 'LDA'
            elif func in gga_functionals:
                category = 'GGA'
            elif func in mgga_functionals:
                category = 'MGGA'
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
                    f.write(f"\\multirow{{{remaining}}}{{*}}{{\\rotatebox{{90}}{{\\emph{{{category}}}}}}}")
            
            data = func_data[func]
            # Format functional name for LaTeX
            if func == 't-HCTH':
                func_display = r'$\tau$--HCTH'
            else:
                func_display = func.replace('-', '--')
            
            def format_value(val, best_val, is_r2=False):
                if val is None:
                    return "---"
                fmt = f"{val:.3f}" if is_r2 else f"{val:.2f}"
                threshold = 0.001 if is_r2 else 0.01
                if best_val is not None and abs(val - best_val) < threshold:
                    return f"\\textbf{{{fmt}}}"
                return fmt
            
            without_d4_str = format_value(data.get('without_d4', {}).get('mad'), best_mad)
            without_d4_r2_str = format_value(data.get('without_d4', {}).get('r_squared'), best_r2, True)
            with_d4_str = format_value(data.get('with_d4', {}).get('mad'), best_mad)
            with_d4_r2_str = format_value(data.get('with_d4', {}).get('r_squared'), best_r2, True)
            
            # Format gradient and offset if available
            def format_fit_params(gradient, offset):
                if gradient is None or offset is None:
                    return "---"
                return f"{gradient:.3f}/{offset:.2f}"
            
            without_d4_fit = format_fit_params(
                data.get('without_d4', {}).get('gradient'),
                data.get('without_d4', {}).get('offset')
            )
            with_d4_fit = format_fit_params(
                data.get('with_d4', {}).get('gradient'),
                data.get('with_d4', {}).get('offset')
            )
            
            if has_linear_fit:
                if is_first_in_category and category_counts.get(category, 0) == 1:
                    f.write(f" & \\raisebox{{-0.5\\height}}{{{func_display}}} & \\raisebox{{-0.5\\height}}{{{without_d4_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_r2_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_fit}}} & \\raisebox{{-0.5\\height}}{{{with_d4_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_r2_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_fit}}} \\\\[2ex]\n")
                else:
                    f.write(f" & {func_display} & {without_d4_str} & {without_d4_r2_str} & {without_d4_fit} & {with_d4_str} & {with_d4_r2_str} & {with_d4_fit} \\\\\n")
            else:
                if is_first_in_category and category_counts.get(category, 0) == 1:
                    f.write(f" & \\raisebox{{-0.5\\height}}{{{func_display}}} & \\raisebox{{-0.5\\height}}{{{without_d4_str}}} & \\raisebox{{-0.5\\height}}{{{without_d4_r2_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_str}}} & \\raisebox{{-0.5\\height}}{{{with_d4_r2_str}}} \\\\[2ex]\n")
                else:
                    f.write(f" & {func_display} & {without_d4_str} & {without_d4_r2_str} & {with_d4_str} & {with_d4_r2_str} \\\\\n")
        
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Mean Absolute Deviation (MAD) and coefficient of determination (R²) of isomerization energies for COMPAS-3x geometries relative to revDSD-PBEP86-D4(noFC)/def2-QZVPP. All calculations performed with (99,590) grid and def2-TZVP basis set. Best values across all functionals are highlighted in bold.}\n")
        f.write("\\label{tab:compas3x_benchmarks}\n\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark functionals against revDSD-PBEP86-D4(noFC) for COMPAS-3x")
    parser.add_argument("--exess-csv", default="analysis/exess_data.csv", help="EXESS data CSV file")
    parser.add_argument("--output", default="compas3x_benchmarks.txt", help="Output text file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--reference-method", choices=['min', 'avg', 'linear_fit'], default='min',
                        help="Method for calculating isomerization energies: 'min', 'avg', or 'linear_fit' (applies linear correction)")
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
    
    functionals_to_test = ['SVWN5', 'PBE', 'BLYP', 'revPBE', 'BP86', 'BPW91', 'B97-D', 'HCTH407',
                           'TPSS', 'MN15L', 'SCAN', 'rSCAN', 'r2SCAN', 'revTPSS', 't-HCTH', 'M06-L', 'M11-L']
    all_results = []
    
    # MGGA functionals without D4 support
    mgga_no_d4 = ['M11-L', 'MN15L', 't-HCTH']
    
    for functional in functionals_to_test:
        if functional == 'SVWN5' or functional == 'HCTH407' or functional in mgga_no_d4:
            # These functionals don't have D4 support, so only test without D4
            result = process_functional_comparison(df, functional, include_d4=False, output_dir=output_dir,
                                                   reference_method=args.reference_method)
            if result:
                all_results.append(result)
        else:
            # For other functionals, test both with and without D4
            for include_d4 in [True, False]:
                result = process_functional_comparison(df, functional, include_d4=include_d4, output_dir=output_dir,
                                                       reference_method=args.reference_method)
                if result:
                    all_results.append(result)
    
    if len(all_results) == 0:
        print("\nNo results to save.")
        return
    
    # Save results to file
    with open(output_path, 'w') as f:
        f.write("COMPAS-3x Benchmark: Functionals vs revDSD-PBEP86-D4(noFC)\n")
        f.write("="*60 + "\n\n")
        for result in all_results:
            d4_label = "with D4" if result['include_d4'] else "without D4"
            f.write(f"\n{result['functional']} ({d4_label})\n")
            f.write("-"*60 + "\n")
            f.write(f"Number of matched structures: {result['n_structures']}\n\n")
            f.write("Statistics:\n")
            f.write(f"  MAD: {result['mad_kjmol']:.3f} kJ/mol\n")
            f.write(f"  RMSD: {result['rmsd']:.3f} kJ/mol\n")
            f.write(f"  R²: {result['r_squared']:.4f}\n")
            f.write(f"  MAD as percentage: {result['mad_percentage']:.2f}%\n\n")
            top_indices = result['deviations'].abs().nlargest(10).index
            f.write("Top 10 largest deviations:\n")
            for idx in top_indices:
                f.write(f"  {result['merged'].loc[idx, 'common_id']}: {result['deviations'].loc[idx]:.3f} kJ/mol\n")
            f.write("\n")
    
    # Generate LaTeX table
    latex_path = output_path.with_suffix('.tex')
    generate_latex_table(all_results, latex_path)
    print(f"\nResults saved to {output_path}")
    print(f"LaTeX table saved to {latex_path}")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
