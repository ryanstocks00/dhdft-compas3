#!/usr/bin/env python3

"""
Script to compare HOMO, LUMO, and HLG between PBE0-D4 and exess revDSD-PBEP86-D4 results.
Generates scatter plots and statistical comparisons.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots
plt.style.use(['science', 'ieee'])

# Enable LaTeX math rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'Times', 'serif']
plt.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Liberation Mono', 'Courier New', 'monospace']

HARTREE_TO_KJ_PER_MOL = 2625.5
SINGLE_COLUMN_WIDTH = 3.5


def extract_common_id(isomer_name):
    """Extract common identifier from isomer name."""
    parts = isomer_name.split('_', 1)
    if len(parts) > 1:
        common_id = parts[1]
        if common_id.startswith('hc_'):
            common_id = common_id[3:]
        return common_id
    return isomer_name


def extract_common_id_from_filename(filename):
    """Extract common identifier from Excel filename."""
    if pd.isna(filename) or not isinstance(filename, str):
        return None
    name = filename.replace('.log', '')
    parts = name.split('_', 1)
    if len(parts) > 1:
        common_id = parts[1]
        # Remove 'hc_' prefix if present (for consistency with exess data)
        if common_id.startswith('hc_'):
            common_id = common_id[3:]
        return common_id
    return name


def load_pbe0_data(excel_path):
    """Load PBE0-D4 data from Excel file (jcc70198-sup-0001-data.xlsx)."""
    # Read Excel file - same format as used in plot_compas_comparison.py
    df = pd.read_excel(excel_path, sheet_name='Table S1', header=9)
    
    # Check for required columns
    if 'homo_energy' not in df.columns or 'lumo_energy' not in df.columns:
        raise ValueError("Excel file must contain 'homo_energy' and 'lumo_energy' columns")
    
    # Extract common IDs from filename
    df['common_id'] = df['file'].apply(extract_common_id_from_filename)
    df = df.dropna(subset=['common_id', 'homo_energy', 'lumo_energy'])
    
    # Handle HLG - check if it's in eV or Hartree
    if 'band_gap_eV' in df.columns:
        # Convert eV to Hartree (1 eV = 1/27.211386245988 Hartree)
        # First convert eV to kJ/mol, then to Hartree
        EV_TO_KJ_PER_MOL = 96.485
        df['gap_hartree'] = (df['band_gap_eV'] * EV_TO_KJ_PER_MOL) / HARTREE_TO_KJ_PER_MOL
    elif 'band_gap' in df.columns:
        # Assume it's already in Hartree if column name doesn't have _eV
        df['gap_hartree'] = df['band_gap']
    else:
        # Compute gap from HOMO and LUMO
        df['gap_hartree'] = df['lumo_energy'] - df['homo_energy']
    
    # Create standardized columns
    result_df = pd.DataFrame({
        'common_id': df['common_id'],
        'isomer_name': df['file'],
        'homo_hartree': df['homo_energy'],
        'lumo_hartree': df['lumo_energy'],
        'gap_hartree': df['gap_hartree']
    })
    
    return result_df


def load_exess_data(csv_path):
    """Load exess revDSD-PBEP86-D4 data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    if 'homo_hartree' not in df.columns or 'lumo_hartree' not in df.columns:
        raise ValueError("exess_data.csv must contain 'homo_hartree' and 'lumo_hartree' columns")
    
    if 'hlg_hartree' not in df.columns:
        # Compute gap from HOMO and LUMO
        df['hlg_hartree'] = df['lumo_hartree'] - df['homo_hartree']
    
    # Extract common IDs
    df['common_id'] = df['isomer_name'].apply(extract_common_id)
    
    # Create standardized dataframe
    result_df = pd.DataFrame({
        'common_id': df['common_id'],
        'isomer_name': df['isomer_name'],
        'optimizer': df.get('optimizer', 'unknown'),
        'homo_hartree': df['homo_hartree'],
        'lumo_hartree': df['lumo_hartree'],
        'gap_hartree': df['hlg_hartree']
    })
    
    return result_df


def clone_compas_repo(gitlab_url, cache_dir=None):
    """Clone the COMPAS-3 repository from GitLab."""
    import subprocess
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / '.compas_cache'
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    repo_url = gitlab_url if gitlab_url.endswith('.git') else gitlab_url + '.git'
    repo_dir = cache_dir / 'compas'
    
    if repo_dir.exists() and (repo_dir / '.git').exists():
        subprocess.run(['git', 'pull'], cwd=repo_dir, check=False, capture_output=True)
        return repo_dir
    
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main', repo_url, str(repo_dir)], check=True)
    return repo_dir


def load_xtb_data_from_compas3x(compas3x_path):
    """
    Load xTB HOMO/LUMO/HLG data from COMPAS-3x CSV file.
    
    Parameters:
    -----------
    compas3x_path : Path or str
        Path to COMPAS-3x CSV file
    
    Returns:
    --------
    DataFrame or None
        DataFrame with xTB HOMO/LUMO/HLG data, or None if not found
    """
    try:
        df = pd.read_csv(compas3x_path)
    except Exception as e:
        print(f"Warning: Could not load COMPAS-3x data from {compas3x_path}: {e}")
        return None
    
    # COMPAS-3x has HOMO_eV, LUMO_eV, and GAP_eV columns
    if 'HOMO_eV' not in df.columns or 'LUMO_eV' not in df.columns:
        print("Warning: COMPAS-3x CSV does not contain HOMO_eV and LUMO_eV columns")
        return None
    
    # Extract common IDs from molecule column
    if 'molecule' in df.columns:
        df['common_id'] = df['molecule'].apply(extract_common_id)
    else:
        print("Warning: COMPAS-3x CSV does not contain 'molecule' column")
        return None
    
    # Convert eV to Hartree (1 eV = 1/27.211386245988 Hartree)
    EV_TO_HARTREE = 1.0 / 27.211386245988
    df['homo_hartree'] = df['HOMO_eV'] * EV_TO_HARTREE
    df['lumo_hartree'] = df['LUMO_eV'] * EV_TO_HARTREE
    
    # Use GAP_eV if available, otherwise compute from HOMO and LUMO
    if 'GAP_eV' in df.columns:
        df['gap_hartree'] = df['GAP_eV'] * EV_TO_HARTREE
    else:
        df['gap_hartree'] = df['lumo_hartree'] - df['homo_hartree']
    
    df = df.dropna(subset=['common_id', 'homo_hartree', 'lumo_hartree'])
    
    # Create standardized columns
    result_df = pd.DataFrame({
        'common_id': df['common_id'],
        'isomer_name': df['molecule'],
        'homo_hartree': df['homo_hartree'],
        'lumo_hartree': df['lumo_hartree'],
        'gap_hartree': df['gap_hartree']
    })
    
    return result_df


def load_xtb_data(excel_path=None, csv_path=None, compas3x_path=None):
    """
    Load xTB HOMO/LUMO/HLG data from various sources.
    
    Parameters:
    -----------
    excel_path : Path or str, optional
        Path to Excel file containing xTB data
    csv_path : Path or str, optional
        Path to CSV file containing xTB data
    compas3x_path : Path or str, optional
        Path to COMPAS-3x CSV file
    
    Returns:
    --------
    DataFrame or None
        DataFrame with xTB HOMO/LUMO/HLG data, or None if not found
    """
    # Try COMPAS-3x first if provided
    if compas3x_path:
        result = load_xtb_data_from_compas3x(compas3x_path)
        if result is not None:
            return result
    
    # Try CSV if provided
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not load xTB data from CSV {csv_path}: {e}")
            df = None
    else:
        df = None
    
    # Try Excel file if provided and CSV didn't work
    if df is None and excel_path:
        try:
            df = pd.read_excel(excel_path, sheet_name='Table S1', header=9)
        except Exception as e:
            print(f"Warning: Could not load xTB data from Excel {excel_path}: {e}")
            return None
    
    if df is None:
        return None
    
    # Check for xTB HOMO/LUMO columns (they might have different names)
    xtb_homo_col = None
    xtb_lumo_col = None
    
    # Try to find xTB orbital columns
    for col in df.columns:
        col_lower = col.lower()
        if 'xtb' in col_lower and 'homo' in col_lower:
            xtb_homo_col = col
        if 'xtb' in col_lower and 'lumo' in col_lower:
            xtb_lumo_col = col
    
    # If not found with 'xtb' prefix, try alternative names
    if xtb_homo_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'homo' in col_lower and ('xtb' in col_lower or 'gfnn' in col_lower):
                xtb_homo_col = col
                break
    
    if xtb_lumo_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'lumo' in col_lower and ('xtb' in col_lower or 'gfnn' in col_lower):
                xtb_lumo_col = col
                break
    
    # If still not found, return None
    if xtb_homo_col is None or xtb_lumo_col is None:
        return None
    
    # Extract common IDs
    if 'file' in df.columns:
        df['common_id'] = df['file'].apply(extract_common_id_from_filename)
    elif 'isomer_name' in df.columns:
        df['common_id'] = df['isomer_name'].apply(extract_common_id)
    else:
        print("Warning: Could not find 'file' or 'isomer_name' column for xTB data")
        return None
    
    df = df.dropna(subset=['common_id', xtb_homo_col, xtb_lumo_col])
    
    # Compute HLG from HOMO and LUMO
    df['gap_hartree'] = df[xtb_lumo_col] - df[xtb_homo_col]
    
    # Create standardized columns
    isomer_col = 'file' if 'file' in df.columns else 'isomer_name'
    result_df = pd.DataFrame({
        'common_id': df['common_id'],
        'isomer_name': df[isomer_col],
        'homo_hartree': df[xtb_homo_col],
        'lumo_hartree': df[xtb_lumo_col],
        'gap_hartree': df['gap_hartree']
    })
    
    return result_df


def plot_comparison(df1, df2, property_name, output_dir, label1, label2, folder_name):
    """
    Create scatter plot comparing a property (HOMO, LUMO, or HLG) between two datasets.
    
    Parameters:
    -----------
    df1 : DataFrame
        First dataset with common_id and property columns
    df2 : DataFrame
        Second dataset with common_id and property columns
    property_name : str
        'homo', 'lumo', or 'gap'
    output_dir : Path
        Output directory for plots
    label1 : str
        Label for first dataset (x-axis)
    label2 : str
        Label for second dataset (y-axis)
    folder_name : str
        Folder name for output plots
    """
    # Merge on common_id
    merged = df1.merge(
        df2[['common_id', f'{property_name}_hartree']],
        on='common_id',
        suffixes=('_1', '_2'),
        how='inner'
    )
    
    if len(merged) == 0:
        print(f"Warning: No matching isomers found for {property_name} comparison")
        return
    
    # Convert to kJ/mol
    vals1 = merged[f'{property_name}_hartree_1'] * HARTREE_TO_KJ_PER_MOL
    vals2 = merged[f'{property_name}_hartree_2'] * HARTREE_TO_KJ_PER_MOL
    
    # Remove NaN values
    valid_mask = ~(np.isnan(vals1) | np.isnan(vals2))
    vals1 = vals1[valid_mask]
    vals2 = vals2[valid_mask]
    
    if len(vals1) < 2:
        print(f"Warning: Not enough valid data points for {property_name} comparison (need at least 2, got {len(vals1)})")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    ax.scatter(vals1, vals2, alpha=0.3, s=6, color='#1f77b4', edgecolors='none', linewidth=0)
    
    min_val = min(min(vals1), min(vals2))
    max_val = max(max(vals1), max(vals2))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.0, 
            label='Perfect agreement')
    
    if len(vals1) > 1:
        slope, intercept = np.polyfit(vals1, vals2, 1)
        # Calculate trendline endpoints constrained to data ranges
        x_min, x_max = np.min(vals1), np.max(vals1)
        y_min, y_max = np.min(vals2), np.max(vals2)
        
        # Find intersection of line with bounding box [x_min, x_max] x [y_min, y_max]
        # Line: y = slope * x + intercept
        # We need to find where this line intersects the box boundaries
        candidates_x = []
        candidates_y = []
        
        if slope != 0:
            # Intersection with y = y_min
            x_at_ymin = (y_min - intercept) / slope
            if x_min <= x_at_ymin <= x_max:
                candidates_x.append(x_at_ymin)
                candidates_y.append(y_min)
            
            # Intersection with y = y_max
            x_at_ymax = (y_max - intercept) / slope
            if x_min <= x_at_ymax <= x_max:
                candidates_x.append(x_at_ymax)
                candidates_y.append(y_max)
        
        # Intersection with x = x_min
        y_at_xmin = slope * x_min + intercept
        if y_min <= y_at_xmin <= y_max:
            candidates_x.append(x_min)
            candidates_y.append(y_at_xmin)
        
        # Intersection with x = x_max
        y_at_xmax = slope * x_max + intercept
        if y_min <= y_at_xmax <= y_max:
            candidates_x.append(x_max)
            candidates_y.append(y_at_xmax)
        
        # Use the two points that define the line segment within the box
        if len(candidates_x) >= 2:
            # Sort by x to get endpoints
            sorted_indices = np.argsort(candidates_x)
            trendline_x = np.array([candidates_x[sorted_indices[0]], candidates_x[sorted_indices[-1]]])
            trendline_y = np.array([candidates_y[sorted_indices[0]], candidates_y[sorted_indices[-1]]])
        else:
            # Fallback: use x-range with clipped y
            trendline_x = np.array([x_min, x_max])
            trendline_y = np.clip(slope * trendline_x + intercept, y_min, y_max)
        
        ax.plot(trendline_x, trendline_y, 'black', alpha=0.8, linewidth=0.9, 
                linestyle='-', label='Linear fit', zorder=10)
    
    # Calculate statistics
    if len(vals1) >= 2:
        # Check for constant values (would cause division by zero in correlation)
        if np.std(vals1) == 0 or np.std(vals2) == 0:
            correlation = np.nan
            r_squared = np.nan
        else:
            correlation = np.corrcoef(vals1, vals2)[0, 1]
            r_squared = correlation ** 2
    else:
        correlation = np.nan
        r_squared = np.nan
    
    mad = np.mean(np.abs(vals1 - vals2))
    
    # Property labels
    property_labels = {
        'homo': 'HOMO',
        'lumo': 'LUMO',
        'gap': 'HLG'
    }
    prop_label = property_labels.get(property_name, property_name.upper())
    
    ax.set_xlabel(f'{label1} {prop_label} (kJ/mol)', fontsize=8)
    ax.set_ylabel(f'{label2} {prop_label} (kJ/mol)', fontsize=8)
    
    if np.isnan(r_squared):
        summary_text = f'$r^2$ = N/A\nMAD = {mad:.2f} kJ/mol'
    else:
        summary_text = f'$r^2$ = {r_squared:.3f}\nMAD = {mad:.2f} kJ/mol'
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, 
           horizontalalignment='right', verticalalignment='bottom',
           fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                               edgecolor='black', linewidth=0.5))
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Set equal axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    max_range = max(xrange, yrange)
    xcenter = (xlim[0] + xlim[1]) / 2
    ycenter = (ylim[0] + ylim[1]) / 2
    ax.set_xlim(xcenter - max_range/2, xcenter + max_range/2)
    ax.set_ylim(ycenter - max_range/2, ycenter + max_range/2)
    
    plt.tight_layout()
    
    # Handle offset text
    x_offset = ax.xaxis.get_offset_text()
    y_offset = ax.yaxis.get_offset_text()
    if x_offset.get_text():
        x_offset.set_visible(False)
        ax.text(0.98, 0.98, x_offset.get_text(), transform=ax.transAxes, 
               horizontalalignment='right', verticalalignment='top', 
               fontsize=8, color=x_offset.get_color())
    if y_offset.get_text():
        y_offset.set_visible(False)
        ax.text(0.02, 0.02, y_offset.get_text(), transform=ax.transAxes, 
               horizontalalignment='left', verticalalignment='bottom', 
               fontsize=8, color=y_offset.get_color())
    
    # Save plot
    folder_path = output_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    output_filename = f'{label1.lower().replace("-", "_").replace(" ", "_")}_vs_{label2.lower().replace("-", "_").replace(" ", "_")}_{property_name}.png'
    plt.savefig(folder_path / output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {folder_path / output_filename}")
    plt.close()
    
    return {
        'property': property_name,
        'n_points': len(merged),
        'r_squared': r_squared,
        'mad_kjmol': mad,
        'correlation': correlation
    }


def print_statistics(stats_list):
    """Print summary statistics table."""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"{'Property':<12} {'N':<6} {'r²':<8} {'MAD (kJ/mol)':<15}")
    print("-"*80)
    
    for stats in stats_list:
        if stats:
            r_sq_str = f"{stats['r_squared']:.3f}" if not np.isnan(stats['r_squared']) else "N/A"
            print(f"{stats['property'].upper():<12} {stats['n_points']:<6} "
                  f"{r_sq_str:<8} {stats['mad_kjmol']:<15.2f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare HOMO, LUMO, and HLG between PBE0-D4 and exess revDSD-PBEP86-D4"
    )
    parser.add_argument(
        "--pbe0-excel",
        default="jcc70198-sup-0001-data.xlsx",
        help="PBE0-D4 data Excel file (default: jcc70198-sup-0001-data.xlsx)"
    )
    parser.add_argument(
        "--exess-csv",
        default="exess_data.csv",
        help="exess revDSD-PBEP86-D4 data CSV file (default: exess_data.csv)"
    )
    parser.add_argument(
        "--xtb-csv",
        help="xTB HOMO/LUMO/HLG data CSV file (optional)"
    )
    parser.add_argument(
        "--gitlab-url",
        default="https://gitlab.com/porannegroup/compas.git",
        help="GitLab URL for COMPAS-3 repository (default: https://gitlab.com/porannegroup/compas.git)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    pbe0_path = script_dir / args.pbe0_excel if not Path(args.pbe0_excel).is_absolute() else Path(args.pbe0_excel)
    # Try script_dir first, then parent directory
    if not Path(args.exess_csv).is_absolute():
        exess_path = script_dir / args.exess_csv
        if not exess_path.exists():
            exess_path = script_dir.parent / args.exess_csv
    else:
        exess_path = Path(args.exess_csv)
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PBE0-D4 data from {pbe0_path}...")
    try:
        pbe0_df = load_pbe0_data(pbe0_path)
        print(f"Loaded {len(pbe0_df)} PBE0-D4 entries")
    except Exception as e:
        print(f"Error loading PBE0-D4 data: {e}")
        return
    
    print(f"Loading exess revDSD-PBEP86-D4 data from {exess_path}...")
    try:
        exess_df = load_exess_data(exess_path)
        print(f"Loaded {len(exess_df)} exess entries")
    except Exception as e:
        print(f"Error loading exess data: {e}")
        return
    
    # Filter exess to GFN2-xTB geometries
    exess_xtb = exess_df[exess_df['optimizer'] == 'GFN2-xTB'].copy()
    
    # Try to load xTB data from COMPAS-3x
    xtb_df = None
    if args.xtb_csv:
        xtb_csv_path = script_dir.parent / args.xtb_csv if not Path(args.xtb_csv).is_absolute() else Path(args.xtb_csv)
        print(f"\nAttempting to load GFN2-xTB data from {xtb_csv_path}...")
        xtb_df = load_xtb_data(csv_path=xtb_csv_path)
    else:
        # Try to load from COMPAS-3x CSV
        print(f"\nAttempting to load GFN2-xTB data from COMPAS-3x...")
        repo_dir = clone_compas_repo(args.gitlab_url)
        compas3x_path = repo_dir / 'COMPAS-3' / 'compas-3x.csv'
        if compas3x_path.exists():
            xtb_df = load_xtb_data(compas3x_path=compas3x_path)
        else:
            # Fallback to Excel file
            print(f"COMPAS-3x CSV not found, trying Excel file...")
            xtb_df = load_xtb_data(excel_path=pbe0_path)
    
    if xtb_df is not None:
        print(f"Loaded {len(xtb_df)} GFN2-xTB entries")
    else:
        print("GFN2-xTB HOMO/LUMO data not found. Use --xtb-csv to specify a CSV file with GFN2-xTB data.")
    
    print("\nGenerating comparison plots...")
    
    stats_list = []
    properties = ['homo', 'lumo', 'gap']
    
    # PBE0-D4 vs exess revDSD-PBEP86-D4
    print("\nPBE0-D4 vs exess revDSD-PBEP86-D4:")
    for prop in properties:
        stats = plot_comparison(
            pbe0_df, exess_xtb, prop, output_dir,
            'PBE0-D4', 'revDSD-PBEP86-D4', 'homo_lumo_gap_comparison'
        )
        if stats:
            stats['comparison'] = 'PBE0_vs_exess'
            stats_list.append(stats)
    
    # GFN2-xTB vs PBE0-D4 (if GFN2-xTB data available)
    if xtb_df is not None:
        print("\nGFN2-xTB vs PBE0-D4:")
        for prop in properties:
            stats = plot_comparison(
                xtb_df, pbe0_df, prop, output_dir,
                'GFN2-xTB', 'PBE0-D4', 'homo_lumo_gap_comparison'
            )
            if stats:
                stats['comparison'] = 'GFN2-xTB_vs_PBE0'
                stats_list.append(stats)
    
    # GFN2-xTB vs exess revDSD-PBEP86-D4 (if GFN2-xTB data available)
    if xtb_df is not None:
        print("\nGFN2-xTB vs exess revDSD-PBEP86-D4:")
        for prop in properties:
            stats = plot_comparison(
                xtb_df, exess_xtb, prop, output_dir,
                'GFN2-xTB', 'revDSD-PBEP86-D4', 'homo_lumo_gap_comparison'
            )
            if stats:
                stats['comparison'] = 'GFN2-xTB_vs_exess'
                stats_list.append(stats)
    
    # Print summary statistics
    print_statistics(stats_list)
    
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()

