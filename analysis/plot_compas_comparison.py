#!/usr/bin/env python3
"""Generate comparison plots from exess_data.csv with COMPAS-3 and Gregory PBE0 data."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "inputs"))
from common import clone_compas_repo

from plotting_utils import (
    create_scatter_plot, extract_common_id, calculate_stats, format_axis_offsets,
    HARTREE_TO_KJ_PER_MOL, SINGLE_COLUMN_WIDTH
)

GREGORY_PBE0_URL = "https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fjcc.70198&file=jcc70198-sup-0001-Data.xlsx"
label_map = {'total': r'Total', 'mp2': r'RI-PT2', 'b_formation': r'$B_{\mu\nu}^P$', 'ri_fock': r'RI-HF', 'xc': r'XC'}


def extract_common_id_from_filename(filename):
    """Extract common identifier from Gregory PBE0 filename."""
    if pd.isna(filename) or not isinstance(filename, str):
        return None
    name = filename.replace('.log', '')
    parts = name.split('_', 1)
    return parts[1] if len(parts) > 1 else name


def plot_compas_comparison(exess_df, original_df, compas_type, output_dir):
    """Compare EXESS energies with original COMPAS-3 energies."""
    exess_df = exess_df.copy()
    exess_df['common_id'] = exess_df['isomer_name'].apply(extract_common_id)
    
    configs = {
        '3x': ('xTB', 'GFN2-xTB', r'revDSD-PBEP86-D4(noFC)/def2QZVPP'),
        '3D': ('DFT', 'CAM-B3LYP-D3BJ/aug-cc-pVDZ', r'revDSD-PBEP86-D4(noFC)/def2QZVPP')
    }
    if compas_type not in configs:
        return
    
    optimizer, original_label, exess_label = configs[compas_type]
    exess_data = exess_df[exess_df['optimizer'] == optimizer].copy()
    
    original_df = original_df.copy()
    original_df['common_id'] = (original_df['molecule'] if 'molecule' in original_df.columns
                               else original_df['isomer_name'].apply(extract_common_id) if 'isomer_name' in original_df.columns else None)
    if original_df['common_id'] is None:
        return
    
    merged = exess_data.merge(original_df, on='common_id', suffixes=('_exess', '_original'), how='inner')
    if len(merged) == 0:
        return
    
    exess_rel = merged['isomerization_energy_hartree'] * HARTREE_TO_KJ_PER_MOL
    energy_col = 'erel_ev' if 'erel_ev' in merged.columns else 'Erel_eV'
    if energy_col not in merged.columns:
        return
    original_rel = merged[energy_col] * 96.485
    
    output_path = output_dir / f"compas_{compas_type}_comparison" / f'exess_vs_compas{compas_type}.png'
    create_scatter_plot(exess_rel, original_rel, f'{exess_label} $\Delta E$ (kJ/mol)',
                       f'{original_label} $\Delta E$ (kJ/mol)', output_path)


def plot_gregory_pbe0_comparison(exess_df, gregory_pbe0_df, energy_type, output_dir, geometry_type='xtb'):
    """Compare EXESS energies with Gregory PBE0 data (PBE0-D4 or xTB)."""
    exess_df = exess_df.copy()
    exess_df['common_id'] = exess_df['isomer_name'].apply(extract_common_id)
    gregory_pbe0_data = gregory_pbe0_df.copy()
    
    if energy_type == 'pbe0':
        optimizer = 'xTB' if geometry_type == 'xtb' else 'DFT'
        exess_label = (r'xTB geometry, revDSD-PBEP86-D4(noFC)/def2QZVPP' if geometry_type == 'xtb'
                      else r'CAM-B3LYP-D3BJ geometry, revDSD-PBEP86-D4(noFC)/def2QZVPP')
        original_label = 'PBE0-D4/6-31G(2df,p)'
        gregory_pbe0_data['original_rel_kjmol'] = gregory_pbe0_data['D4_rel_energy'] * 4.184
        folder_suffix = f'{geometry_type}_geom'
    elif energy_type == 'xtb':
        optimizer = 'xTB'
        exess_label, original_label = r'xTB', 'xTB'
        gregory_pbe0_data['original_rel_kjmol'] = gregory_pbe0_data['xtb_rel_energy_kcal_mol'] * 4.184
        folder_suffix = ''
    else:
        return
    
    exess_data = exess_df[exess_df['optimizer'] == optimizer].copy()
    merged = exess_data.merge(gregory_pbe0_data[['common_id', 'original_rel_kjmol']], on='common_id', how='inner')
    if len(merged) == 0:
        return
    
    exess_rel = merged['isomerization_energy_hartree'] * HARTREE_TO_KJ_PER_MOL
    original_rel = merged['original_rel_kjmol']
    
    folder_name = f"gregory_pbe0_{energy_type}_comparison{f'_{folder_suffix}' if folder_suffix else ''}"
    filename = f'exess_vs_{energy_type}{f"_{folder_suffix}" if folder_suffix else ""}.png'
    output_path = output_dir / folder_name / filename
    create_scatter_plot(exess_rel, original_rel, f'{exess_label} $\Delta E$ (kJ/mol)',
                       f'{original_label} $\Delta E$ (kJ/mol)', output_path)


def plot_gregory_pbe0_xtb_vs_compas_xtb(exess_df, compas3x_df, gregory_pbe0_df, output_dir):
    """Compare Gregory PBE0 xTB energies with COMPAS-3x xTB energies."""
    compas3x_df = compas3x_df.copy()
    compas3x_df['common_id'] = (compas3x_df['molecule'] if 'molecule' in compas3x_df.columns
                               else compas3x_df['isomer_name'].apply(extract_common_id))
    
    gregory_pbe0_data = gregory_pbe0_df.copy()
    gregory_pbe0_data['gregory_pbe0_xtb_kjmol'] = gregory_pbe0_data['xtb_rel_energy_kcal_mol'] * 4.184
    
    energy_col = 'erel_ev' if 'erel_ev' in compas3x_df.columns else 'Erel_eV'
    if energy_col not in compas3x_df.columns:
        return
    compas3x_df['compas_xtb_kjmol'] = compas3x_df[energy_col] * 96.485
    
    merged = compas3x_df[['common_id', 'compas_xtb_kjmol']].merge(
        gregory_pbe0_data[['common_id', 'gregory_pbe0_xtb_kjmol']], on='common_id', how='inner')
    if len(merged) == 0:
        return
    
    compas_xtb, gregory_pbe0_xtb = merged['compas_xtb_kjmol'], merged['gregory_pbe0_xtb_kjmol']
    
    deviations = gregory_pbe0_xtb - compas_xtb
    top_indices = np.abs(deviations).nlargest(10).index
    print(f"\nTop 10 biggest deviations for Gregory PBE0 xTB vs COMPAS xTB:")
    for idx in top_indices:
        print(f"  {merged.loc[idx, 'common_id']}: Gregory PBE0={gregory_pbe0_xtb.loc[idx]:.2f}, "
              f"COMPAS={compas_xtb.loc[idx]:.2f}, Dev={deviations.loc[idx]:.2f} kJ/mol")
    
    output_path = output_dir / "gregory_pbe0_xtb_vs_compas_xtb_comparison" / 'gregory_pbe0_xtb_vs_compas_xtb.png'
    create_scatter_plot(gregory_pbe0_xtb, compas_xtb, r'Gregory PBE0 xTB $\Delta E$ (kJ/mol)',
                       r'COMPAS-3x xTB $\Delta E$ (kJ/mol)', output_path)


def plot_by_basis(df, output_dir, plot_type='timings'):
    """Plot timings or TFLOP/s by number of basis functions with error bars."""
    cols = (['total_time_s', 'mp2_time_s', 'b_formation_time_s', 'ri_fock_time_s', 'xc_time_s'] if plot_type == 'timings'
           else ['total_tflop/s', 'mp2_tflop/s', 'b_formation_tflop/s', 'ri_fock_tflop/s', 'xc_tflop/s'])
    basis_col = 'n_primary_basis_functions'
    
    df_plot = df[[basis_col] + cols].dropna()
    df_plot = df_plot[df_plot[basis_col] >= 1400]
    
    grouped = df_plot.groupby(basis_col)
    means, stds = grouped[cols].mean(), grouped[cols].std()
    basis_funcs = sorted(means.index)
    
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.75))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '8']
    
    for i, col in enumerate(cols):
        label = label_map[col.replace('_time_s', '').replace('_tflop/s', '')]
        ax.errorbar(basis_funcs, means.loc[basis_funcs, col], yerr=stds.loc[basis_funcs, col],
                   marker=markers[i % len(markers)], label=label, color=colors[i],
                   capsize=1, capthick=0.3, elinewidth=0.3, linewidth=0.8,
                   markersize=3, markerfacecolor='none', markeredgewidth=1)
    
    if plot_type == 'tflops':
        ax.axhline(y=78.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                  label=r'Theoretical peak ($4 \times$ A100)')
        ax.set_ylim(bottom=0)
    else:
        ax.set_yscale('log')
    
    ax.set_xlabel(r'Number of Basis Functions', fontsize=9)
    ax.set_ylabel(r'Time (seconds)' if plot_type == 'timings' else r'TFLOP/s', fontsize=9)
    ax.legend(loc='upper left', fontsize=7, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    
    output_path = output_dir / f'{plot_type}_by_basis_functions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_energy_comparison_by_system_size(df, output_dir):
    """Compare xTB and CAM-B3LYP optimized geometry energies for each system size."""
    df = df.copy()
    df['optimizer_clean'] = df['optimizer'].map({'xTB': 'xTB', 'DFT': 'CAM-B3LYP'})
    df['common_id'] = df['isomer_name'].apply(extract_common_id)
    df['system_size'] = df.apply(lambda r: f"C{int(r['n_carbons'])}H{int(r['n_hydrogens'])}", axis=1)
    
    for system_size in sorted(df['system_size'].unique()):
        df_size = df[df['system_size'] == system_size]
        pairs = []
        
        for common_id in sorted(df_size['common_id'].unique()):
            id_data = df_size[df_size['common_id'] == common_id]
            xtb_data = id_data[id_data['optimizer_clean'] == 'xTB']
            cam_data = id_data[id_data['optimizer_clean'] == 'CAM-B3LYP']
            
            if len(xtb_data) > 0 and len(cam_data) > 0:
                energy_col = 'isomerization_energy_hartree' if 'isomerization_energy_hartree' in id_data.columns else 'total_energy_hartree'
                pairs.append((xtb_data[energy_col].iloc[0], cam_data[energy_col].iloc[0]))
        
        if not pairs:
            continue
        
        xtb_energies = np.array([e[0] for e in pairs]) * HARTREE_TO_KJ_PER_MOL
        cam_energies = np.array([e[1] for e in pairs]) * HARTREE_TO_KJ_PER_MOL
        
        output_path = output_dir / f'energy_comparison_{system_size}.png'
        create_scatter_plot(xtb_energies, cam_energies, 'GFN2-xTB Optimized (kJ/mol)',
                          'CAM-B3LYP Optimized (kJ/mol)', output_path)


def load_gregory_pbe0_data(gregory_pbe0_path):
    """Load data from Gregory PBE0 file."""
    df = pd.read_excel(gregory_pbe0_path, sheet_name='Table S1', header=9)
    df['common_id'] = df['file'].apply(extract_common_id_from_filename)
    df = df.dropna(subset=['common_id', 'D4_rel_energy', 'xtb_rel_energy_kcal_mol'])
    return df[['common_id', 'D4_rel_energy', 'xtb_rel_energy_kcal_mol']]


def load_compas_data_from_repo(repo_dir):
    """Load COMPAS-3x and COMPAS-3D data from the cloned repository."""
    compas3x_path = repo_dir / 'COMPAS-3' / 'compas-3x.csv'
    compas3d_path = repo_dir / 'COMPAS-3' / 'compas-3D.csv'
    
    compas3x_df = pd.read_csv(compas3x_path) if compas3x_path.exists() else None
    compas3d_df = pd.read_csv(compas3d_path) if compas3d_path.exists() else None
    
    if compas3x_df is not None:
        print(f"Loaded {len(compas3x_df)} rows from COMPAS-3x")
    if compas3d_df is not None:
        print(f"Loaded {len(compas3d_df)} rows from COMPAS-3D")
    
    return compas3x_df, compas3d_df


def check_gregory_pbe0_file(filepath):
    """Check if Gregory PBE0 file exists and provide instructions if not."""
    if not filepath.exists():
        print(f"\nWarning: {filepath.name} not found.")
        print(f"Please download it manually from:\n  {GREGORY_PBE0_URL}\nand save it as: {filepath}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare optimized energies with original COMPAS-3 results")
    parser.add_argument("-i", "--input", default="exess_data.csv", help="Input CSV file (default: exess_data.csv)")
    parser.add_argument("-o", "--output-dir", default="plots", help="Output directory (default: plots)")
    parser.add_argument("--gitlab-url", default="https://gitlab.com/porannegroup/compas.git", help="GitLab URL")
    parser.add_argument("--gregory-pbe0-file", default="jcc70198-sup-0001-data.xlsx", help="Gregory PBE0 file path")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / args.input if args.input.startswith('../') else script_dir / args.input
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading optimized data from {csv_path}...")
    optimized_df = pd.read_csv(csv_path)
    print(f"Loaded {len(optimized_df)} rows")
    
    print("\nLoading original COMPAS-3 data...")
    repo_dir = clone_compas_repo(gitlab_url=args.gitlab_url)
    compas3x_data, compas3d_data = load_compas_data_from_repo(repo_dir)
    
    if compas3x_data is None or compas3d_data is None:
        print("Error: Could not load COMPAS-3 data")
        return
    
    print("\nGenerating plots...")
    plot_by_basis(optimized_df, output_dir, 'timings')
    plot_by_basis(optimized_df, output_dir, 'tflops')
    plot_energy_comparison_by_system_size(optimized_df, output_dir)
    
    print("\nGenerating COMPAS comparison plots...")
    plot_compas_comparison(optimized_df, compas3x_data, '3x', output_dir)
    plot_compas_comparison(optimized_df, compas3d_data, '3D', output_dir)
    
    gregory_pbe0_path = script_dir / args.gregory_pbe0_file if not Path(args.gregory_pbe0_file).is_absolute() else Path(args.gregory_pbe0_file)
    if check_gregory_pbe0_file(gregory_pbe0_path):
        print("\nLoading Gregory PBE0 data...")
        gregory_pbe0_data = load_gregory_pbe0_data(gregory_pbe0_path)
        print(f"Loaded {len(gregory_pbe0_data)} rows from Gregory PBE0 file")
        
        plot_gregory_pbe0_comparison(optimized_df, gregory_pbe0_data, 'pbe0', output_dir, geometry_type='xtb')
        plot_gregory_pbe0_comparison(optimized_df, gregory_pbe0_data, 'xtb', output_dir)
        plot_gregory_pbe0_xtb_vs_compas_xtb(optimized_df, compas3x_data, gregory_pbe0_data, output_dir)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
