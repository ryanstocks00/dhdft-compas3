#!/usr/bin/env python3

"""
Script to align GFN2-xTB and CAM-B3LYP-D3BJ optimized geometries and calculate RMSD.
This helps identify if outliers have particularly different geometries.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes
import sys

# Add inputs directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent / "inputs"))
from common import clone_compas_repo


def read_xyz(xyz_path):
    """Read XYZ file and return coordinates and atom symbols."""
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    coords = []
    symbols = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return np.array(coords), symbols


def center_coords(coords):
    """Center coordinates at origin."""
    return coords - coords.mean(axis=0)


def align_geometries(coords1, coords2):
    """
    Align two geometries using Kabsch algorithm (Procrustes analysis).
    Returns aligned coordinates and rotation matrix.
    """
    # Center both geometries
    coords1_centered = center_coords(coords1)
    coords2_centered = center_coords(coords2)
    
    # Compute rotation matrix using SVD (Kabsch algorithm)
    R, _ = orthogonal_procrustes(coords2_centered, coords1_centered)
    
    # Apply rotation to coords2
    coords2_aligned = coords2_centered @ R.T
    
    return coords1_centered, coords2_aligned, R


def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two coordinate sets."""
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd


def find_xyz_path(isomer_name, optimizer, df_row):
    """
    Find the XYZ file path for a given isomer and optimizer.
    Checks multiple possible locations.
    """
    base_dir = Path(__file__).parent.parent
    
    # Ensure COMPAS repo is cloned before accessing xyzs
    clone_compas_repo()
    
    # Extract common identifier from isomer name
    if optimizer == 'GFN2-xTB':
        # compas3x_hc_c40h22_0pent_583 -> hc_c40h22_0pent_583
        common_id = isomer_name.split('_', 1)[1] if '_' in isomer_name else isomer_name
        xyz_dirs = [
            base_dir / 'inputs' / 'compas3x-xyzs',
            base_dir / '.compas_cache' / 'compas' / 'COMPAS-3' / 'compas-3x' / 'xyzs',
        ]
    else:  # DFT
        # compas3D_hc_c40h22_0pent_583 -> hc_c40h22_0pent_583
        common_id = isomer_name.split('_', 1)[1] if '_' in isomer_name else isomer_name
        xyz_dirs = [
            base_dir / 'inputs' / 'compas3D-xyzs',
            base_dir / '.compas_cache' / 'compas' / 'COMPAS-3' / 'compas-3D' / 'xyzs',
        ]
    
    # Try to find the XYZ file in multiple directories
    possible_names = [
        f"{common_id}.xyz",
        f"{isomer_name}.xyz",
        common_id.replace('_', '-') + '.xyz',
        common_id.replace('_', '_') + '.xyz',
    ]
    
    # Also try with different extensions
    for xyz_dir in xyz_dirs:
        if not xyz_dir.exists():
            continue
            
        # Try exact matches first
        for name in possible_names:
            xyz_path = xyz_dir / name
            if xyz_path.exists():
                return xyz_path
        
        # Try pattern matching
        # Look for files containing parts of the common_id
        id_parts = common_id.split('_')
        if len(id_parts) >= 2:
            # Try matching on the last part (e.g., "583")
            last_part = id_parts[-1]
            for xyz_file in xyz_dir.glob(f"*{last_part}*.xyz"):
                return xyz_file
        
        # Try matching on system size (e.g., "c40h22")
        for part in id_parts:
            if 'c' in part.lower() and 'h' in part.lower():
                for xyz_file in xyz_dir.glob(f"*{part}*.xyz"):
                    return xyz_file
    
    # Try to find from input JSON files
    try:
        filename = df_row['filename'].iloc[0] if hasattr(df_row, 'iloc') else df_row.get('filename', '')
        if filename:
            # Extract batch name from filename
            # e.g., global_u2_r_ryans_dhdft_inputs_exess_exess_inputs_isomers_6220-6239_topology_1.json
            if 'isomers_' in filename:
                batch_part = filename.split('isomers_')[1].split('_topology')[0]
                input_json = base_dir / 'inputs' / 'exess' / f'exess_inputs_isomers_{batch_part}.json'
                if input_json.exists():
                    with open(input_json) as f:
                        input_data = json.load(f)
                    if 'topologies' in input_data:
                        # Find topology by index
                        topology_idx = int(filename.split('_topology_')[1].split('.')[0]) if '_topology_' in filename else 0
                        if topology_idx < len(input_data['topologies']):
                            xyz_path_str = input_data['topologies'][topology_idx].get('xyz', '')
                            if xyz_path_str:
                                xyz_path = Path(xyz_path_str)
                                if xyz_path.exists():
                                    return xyz_path
    except Exception as e:
        pass  # Silently continue if this fails
    
    return None


def calculate_rmsd_for_pair(xtb_row, dft_row, csv_path):
    """
    Calculate RMSD for a pair of xTB and DFT optimized geometries.
    """
    xtb_name = xtb_row['isomer_name'].iloc[0]
    dft_name = dft_row['isomer_name'].iloc[0]
    
    print(f"\nProcessing: {xtb_name} (GFN2-xTB) vs {dft_name} (CAM-B3LYP-D3BJ)")
    
    # Find XYZ files
    xtb_xyz = find_xyz_path(xtb_name, 'GFN2-xTB', xtb_row)
    dft_xyz = find_xyz_path(dft_name, 'CAM-B3LYP-D3BJ', dft_row)
    
    if xtb_xyz is None:
        print(f"  Warning: Could not find GFN2-xTB XYZ file for {xtb_name}")
        return None
    if dft_xyz is None:
        print(f"  Warning: Could not find CAM-B3LYP-D3BJ XYZ file for {dft_name}")
        return None
    
    print(f"  GFN2-xTB XYZ: {xtb_xyz}")
    print(f"  CAM-B3LYP-D3BJ XYZ: {dft_xyz}")
    
    # Read geometries
    try:
        xtb_coords, xtb_symbols = read_xyz(xtb_xyz)
        dft_coords, dft_symbols = read_xyz(dft_xyz)
    except Exception as e:
        print(f"  Error reading XYZ files: {e}")
        return None
    
    # Check if number of atoms matches
    if len(xtb_coords) != len(dft_coords):
        print(f"  Warning: Number of atoms mismatch: xTB={len(xtb_coords)}, DFT={len(dft_coords)}")
        return None
    
    # Check if atom symbols match (order might be different)
    if set(xtb_symbols) != set(dft_symbols):
        print(f"  Warning: Atom types mismatch")
        print(f"    GFN2-xTB atoms: {set(xtb_symbols)}")
        print(f"    DFT atoms: {set(dft_symbols)}")
        # Continue anyway, might just be ordering
    
    # Align geometries
    xtb_aligned, dft_aligned, R = align_geometries(xtb_coords, dft_coords)
    
    # Calculate RMSD
    rmsd = calculate_rmsd(xtb_aligned, dft_aligned)
    
    print(f"  RMSD: {rmsd:.4f} Å")
    
    return {
        'xtb_name': xtb_name,
        'dft_name': dft_name,
        'xtb_xyz': str(xtb_xyz),
        'dft_xyz': str(dft_xyz),
        'rmsd': rmsd,
        'n_atoms': len(xtb_coords),
        'xtb_energy': xtb_row['total_energy_hartree'].iloc[0],
        'dft_energy': dft_row['total_energy_hartree'].iloc[0],
        'energy_diff_kjmol': (dft_row['total_energy_hartree'].iloc[0] - xtb_row['total_energy_hartree'].iloc[0]) * 2625.5
    }


def main():
    parser = argparse.ArgumentParser(
        description="Align xTB and DFT optimized geometries and calculate RMSD"
    )
    parser.add_argument(
        "-i", "--input",
        default="exess_data.csv",
        help="Input CSV file path (default: exess_data.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="rmsd_results.csv",
        help="Output CSV file path (default: rmsd_results.csv)"
    )
    parser.add_argument(
        "--outlier-only",
        action="store_true",
        help="Only process the known outlier (c40h22_0pent_583)"
    )
    parser.add_argument(
        "--system-size",
        help="Only process a specific system size (e.g., C40H22)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    if Path(args.input).is_absolute():
        csv_path = Path(args.input)
    else:
        csv_path = script_dir / args.input
    
    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        output_path = script_dir / args.output
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Extract common identifier from isomer name
    def extract_common_id(name):
        parts = name.split('_', 1)
        if len(parts) > 1:
            return parts[1]
        return name
    
    df['common_id'] = df['isomer_name'].apply(extract_common_id)
    df['system_size'] = df.apply(
        lambda row: f"C{int(row['n_carbons'])}H{int(row['n_hydrogens'])}", axis=1
    )
    
    # Filter data
    if args.outlier_only:
        # Process only the known outlier
        df_filtered = df[df['common_id'].str.contains('c40h22_0pent_583', case=False)]
    elif args.system_size:
        df_filtered = df[df['system_size'] == args.system_size]
    else:
        df_filtered = df
    
    # Get unique common IDs
    common_ids = sorted(df_filtered['common_id'].unique())
    
    print(f"\nProcessing {len(common_ids)} isomer(s)...")
    
    results = []
    
    for common_id in common_ids:
        id_data = df_filtered[df_filtered['common_id'] == common_id]
        xtb_data = id_data[id_data['optimizer'] == 'GFN2-xTB']
        dft_data = id_data[id_data['optimizer'] == 'CAM-B3LYP-D3BJ']
        
        if len(xtb_data) > 0 and len(dft_data) > 0:
            result = calculate_rmsd_for_pair(xtb_data, dft_data, csv_path)
            if result:
                results.append(result)
        else:
            print(f"\nSkipping {common_id}: missing GFN2-xTB or CAM-B3LYP-D3BJ data")
    
    if results:
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('rmsd', ascending=False)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {output_path}")
        print(f"{'='*60}")
        print("\nSummary:")
        print(f"  Total pairs processed: {len(results)}")
        print(f"  Mean RMSD: {results_df['rmsd'].mean():.4f} Å")
        print(f"  Median RMSD: {results_df['rmsd'].median():.4f} Å")
        print(f"  Max RMSD: {results_df['rmsd'].max():.4f} Å")
        print(f"  Min RMSD: {results_df['rmsd'].min():.4f} Å")
        
        print("\nTop 10 highest RMSD:")
        print(results_df[['xtb_name', 'dft_name', 'rmsd', 'energy_diff_kjmol']].head(10).to_string(index=False))
        
        if args.outlier_only or 'c40h22_0pent_583' in results_df['xtb_name'].str.cat(results_df['dft_name']):
            print("\nOutlier (c40h22_0pent_583) RMSD:")
            outlier = results_df[results_df['xtb_name'].str.contains('c40h22_0pent_583', case=False) | 
                                results_df['dft_name'].str.contains('c40h22_0pent_583', case=False)]
            if len(outlier) > 0:
                print(outlier[['xtb_name', 'dft_name', 'rmsd', 'energy_diff_kjmol']].to_string(index=False))
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()

