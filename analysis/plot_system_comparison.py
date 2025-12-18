#!/usr/bin/env python3
"""Compare performance across EXESS (8×H200), EXESS (4×A100), and ORCA systems."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
import importlib.util
from matplotlib.transforms import blended_transform_factory

sys.path.append((Path(__file__).parent.parent / "inputs").as_posix())
try:
    from plotting_utils import SINGLE_COLUMN_WIDTH
except:
    SINGLE_COLUMN_WIDTH = 3.5

# Power consumption in watts
POWER_CONSUMPTION = {
    r"EXESS (8 $\times$ H200)": 5800,
    r"EXESS (4 $\times$ A100)": 1800,
    r"ORCA (104 cores, 2 $\times$ Sapphire Rapid)": 700,
}


def load_size_profiling_isomers():
    """Load isomer names from size_profiling.json."""
    path = Path(__file__).parent.parent / "inputs" / "size_profiling.json"
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return [Path(topo["xyz"]).stem for topo in json.load(f).get("topologies", [])]


def extract_carbon_count(name):
    """Extract carbon count from isomer name."""
    match = re.search(r'c(\d+)h', name)
    return int(match.group(1)) if match else None


def extract_gpu_data(folder, system_name, max_idx=None):
    """Extract data from H200 or A100 JSON files."""
    rows = []
    for json_file in sorted(Path(folder).glob("size_profiling_topology_*.json")):
        match = re.search(r'topology_(\d+)', json_file.name)
        if not match or (idx := int(match.group(1))) == 0 or (max_idx and idx > max_idx):
            continue
        try:
            qmmbe = json.load(open(json_file)).get("qmmbe", {})
            if not (qmmbe and qmmbe.get("nmers") and (n_basis := qmmbe["nmers"][0][0].get("num_basis_fns"))):
                continue
            rows.append({
                "system": system_name,
                "n_basis_functions": int(n_basis),
                "total_time_s": float(qmmbe.get("total_time", np.nan)),
                "scf_time_s": float(qmmbe.get("scf_time", np.nan)),
                "mp2_time_s": float(qmmbe.get("mp2_time", np.nan)),
                "total_tflop/s": float(qmmbe.get("tflops", np.nan)),
                "scf_tflop/s": float(qmmbe.get("scf_tflops", np.nan)),
                "mp2_tflop/s": float(qmmbe.get("mp2_tflops", np.nan)),
            })
        except:
            continue
    return pd.DataFrame(rows)


def extract_orca_data(folder, isomers=None):
    """Extract timing data from ORCA output files."""
    extract_path = Path(__file__).parent / "extract_orca_data.py"
    spec = importlib.util.spec_from_file_location("extract_orca", extract_path)
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    
    rows = []
    for prop_file in sorted(Path(folder).glob("*.property.txt")):
        filename = prop_file.stem.replace(".property", "")
        if "compas3x" not in filename.lower() or "qz_riri" not in filename:
            continue
        try:
            sections = extract.parse_property_file(prop_file)
            if sections.get("Calculation_Status", {}).get("STATUS") != "NORMAL TERMINATION":
                continue
            if not (n_basis := int(sections.get("Calculation_Info", {}).get("NUMOFBASISFUNCTS", 0))):
                continue
            out_file = prop_file.parent / (filename + ".out")
            if not out_file.exists():
                continue
            metrics = extract.parse_out_metrics(out_file)
            if not (total_time := metrics.get("scf_time_s", 0) + metrics.get("mp2_time_s", 0)):
                continue
            match = re.search(r'compas3[dx]_?(hc_c\d+h\d+_0pent_\d+)', filename, re.IGNORECASE)
            if isomers and (not match or match.group(1) not in isomers):
                continue
            rows.append({
                "system": r"ORCA (104 cores, 2 $\times$ Sapphire Rapid)",
                "n_basis_functions": n_basis,
                "total_time_s": total_time,
                "scf_time_s": metrics.get("scf_time_s", np.nan),
                "mp2_time_s": metrics.get("mp2_time_s", np.nan),
                "total_tflop/s": np.nan,
                "scf_tflop/s": np.nan,
                "mp2_tflop/s": np.nan,
            })
        except:
            continue
    return pd.DataFrame(rows)


def get_basis_to_carbon_mapping(isomers, h200_folder, a100_folder, orca_folder):
    """Map basis function counts to carbon atom counts."""
    if not isomers:
        return {}
    mapping = {}
    topology_to_carbon = {idx + 1: extract_carbon_count(name) for idx, name in enumerate(isomers)}
    isomer_to_carbon = {name: extract_carbon_count(name) for name in isomers}
    
    # From GPU data
    for folder in [f for f in [h200_folder, a100_folder] if f is not None]:
        for json_file in Path(folder).glob("size_profiling_topology_*.json"):
            match = re.search(r'topology_(\d+)', json_file.name)
            if match and (idx := int(match.group(1))) in topology_to_carbon:
                try:
                    n_basis = json.load(open(json_file)).get("qmmbe", {}).get("nmers", [[{}]])[0][0].get("num_basis_fns")
                    if n_basis:
                        mapping[int(n_basis)] = topology_to_carbon[idx]
                except:
                    pass
    
    # From ORCA data
    extract_path = Path(__file__).parent / "extract_orca_data.py"
    spec = importlib.util.spec_from_file_location("extract_orca", extract_path)
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    for prop_file in Path(orca_folder).glob("*.property.txt"):
        filename = prop_file.stem.replace(".property", "")
        if "compas3x" not in filename.lower() or "qz_riri" not in filename:
            continue
        match = re.search(r'compas3[dx]_?(hc_c\d+h\d+_0pent_\d+)', filename, re.IGNORECASE)
        if not match or (isomer_name := match.group(1)) not in isomer_to_carbon:
            continue
        try:
            n_basis = int(extract.parse_property_file(prop_file).get("Calculation_Info", {}).get("NUMOFBASISFUNCTS", 0))
            if n_basis:
                mapping[n_basis] = isomer_to_carbon[isomer_name]
        except:
            pass
    return mapping


def extract_exess_energy(json_file, orca_n_basis, orca_isomer, d4_energies):
    """Extract EXESS energy from JSON file."""
    try:
        qmmbe = json.load(open(json_file)).get("qmmbe", {})
        if not (qmmbe and qmmbe.get("nmers")):
            return None, None
        n_basis = qmmbe["nmers"][0][0].get("num_basis_fns")
        if orca_n_basis and n_basis and int(n_basis) != orca_n_basis:
            return None, None
        hf = qmmbe.get("expanded_hf_energy")
        if hf is None:
            return None, None
        mp2_os = float(qmmbe.get("expanded_mp2_os_correction", 0.0))
        mp2_ss = float(qmmbe.get("expanded_mp2_ss_correction", 0.0))
        d4 = d4_energies.get(orca_isomer, 0.0) if orca_isomer else 0.0
        return float(hf) + mp2_os + mp2_ss + d4, {"HF": float(hf), "MP2_OS": mp2_os, "MP2_SS": mp2_ss, "D4": d4}
    except:
        return None, None


def plot_comparison(df, output_path, plot_type='timings', basis_to_carbon=None):
    """Create comparison plot."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH * 1.2, SINGLE_COLUMN_WIDTH * 0.9))
    systems = sorted(df["system"].unique(), key=lambda x: (not ("ORCA" in x), "H200" in x, "A100" in x))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    metric = "total_time_s" if plot_type == 'timings' else "total_tflop/s"
    ylabel = "Total Time (seconds)" if plot_type == 'timings' else "TFLOP/s"
    
    system_data = {}
    for i, system in enumerate(systems):
        data = df[df["system"] == system].dropna(subset=[metric, "n_basis_functions"])
        if len(data) == 0:
            continue
        system_data[system] = data
        sorted_data = data.sort_values("n_basis_functions")
        ax.plot(sorted_data["n_basis_functions"], sorted_data[metric], color=colors[i], linewidth=1.0, alpha=0.5, zorder=0)
        ax.scatter(data["n_basis_functions"], data[metric], marker=markers[i], label=system, color=colors[i],
                  s=20, alpha=0.7, edgecolors='none', zorder=2)
    
    # Find and label largest speedup
    if plot_type == 'timings':
        orca_system = next((s for s in systems if "ORCA" in s), None)
        gpu_systems = [s for s in systems if "EXESS" in s]
        orca_data = system_data.get(orca_system) if orca_system else None
        if orca_system and gpu_systems and orca_data is not None:
            max_speedup, max_points = 0, None
            for gpu_system in gpu_systems:
                gpu_data = system_data.get(gpu_system)
                if gpu_data is None:
                    continue
                for _, orca_row in orca_data.iterrows():
                    basis = orca_row["n_basis_functions"]
                    # Only consider systems with more than 2000 basis functions
                    if basis <= 2000:
                        continue
                    matching = gpu_data[gpu_data["n_basis_functions"] == basis]
                    if len(matching) > 0 and (speedup := orca_row[metric] / matching[metric].iloc[0]) > max_speedup:
                        max_speedup = speedup
                        max_points = ((basis, orca_row[metric]), (basis, matching[metric].iloc[0]), gpu_system)
            
            if max_points:
                (x_orca, y_orca), (x_gpu, y_gpu), gpu_system = max_points
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                x_bar = x_gpu + x_range * 0.03
                cap_len = x_range * 0.008
                ax.plot([x_bar, x_bar], [y_gpu, y_orca], 'k-', linewidth=2, zorder=1)
                ax.plot([x_bar - cap_len, x_bar + cap_len], [y_orca, y_orca], 'k-', linewidth=2, zorder=1)
                ax.plot([x_bar - cap_len, x_bar + cap_len], [y_gpu, y_gpu], 'k-', linewidth=2, zorder=1)
                
                # Calculate relative energy (power × time)
                orca_power = POWER_CONSUMPTION.get(orca_system, 700)
                gpu_power = POWER_CONSUMPTION.get(gpu_system, 1800)
                relative_power = gpu_power / orca_power
                # Energy = power × time, so relative energy = relative_power / speedup
                # Invert to show energy efficiency (higher is better)
                relative_energy_inv = max_speedup / relative_power
                power_label = f'{max_speedup:.1f}$\\times$ time\n({relative_energy_inv:.2f}$\\times$ energy)'
                
                # Position label at middle of bar, accounting for log scale if needed
                mid_y = (y_orca + y_gpu) / 2
                if plot_type == 'timings':
                    # For log scale, adjust using geometric mean and raise further
                    mid_y = np.sqrt(y_orca * y_gpu) * 1.5
                # Center the text box on the vertical bar (horizontally and vertically)
                ax.text(x_bar, mid_y, power_label, ha='center', va='center',
                       fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       edgecolor='black', linewidth=0.5), zorder=3)
                orca_color = colors[systems.index(orca_system)]
                gpu_color = colors[systems.index(gpu_system)]
                ax.text(x_orca - x_range * 0.02, y_orca * 1.15, f'{y_orca/60:.1f} min',
                       ha='right', va='bottom', fontsize=8, fontweight='bold', color=orca_color)
                ax.text(x_gpu + x_range * 0.02, y_gpu * 0.85, f'{y_gpu:.1f} s',
                       ha='left', va='top', fontsize=8, fontweight='bold', color=gpu_color)
    
    ax.set_xlabel("Number of Basis Functions", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_yscale('log' if plot_type == 'timings' else 'linear')
    if plot_type == 'timings':
        ax.set_ylim(top=5000)
    
    # Add carbon labels
    if basis_to_carbon:
        all_basis = sorted(df["n_basis_functions"].dropna().unique())
        carbon_to_basis = {}
        for basis in all_basis:
            if carbon := basis_to_carbon.get(basis):
                carbon_to_basis.setdefault(carbon, []).append(basis)
        
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.tick_params(axis='x', length=0, which='both')
        ax_top.tick_params(axis='y', length=0)
        
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        tick_positions = []
        for carbon in sorted(carbon_to_basis.keys()):
            basis_list = sorted(carbon_to_basis[carbon])
            tick_positions.extend([b for b in basis_list if b not in tick_positions])
        
        ax_top.set_xticks(tick_positions)
        ax_top.set_xticklabels([""] * len(tick_positions))
        plt.tight_layout()
        
        y_top, y_bottom, y_label = 1.00, 0.98, 1.015
        for tick_pos in tick_positions:
            ax.plot([tick_pos, tick_pos], [y_bottom, y_top], 'k-', linewidth=1.5, clip_on=False, zorder=10, transform=trans)
        
        for carbon in sorted(carbon_to_basis.keys()):
            basis_list = sorted(carbon_to_basis[carbon])
            min_basis, max_basis = min(basis_list), max(basis_list)
            ax.plot([min_basis, max_basis], [y_top, y_top], 'k-', linewidth=1.5, clip_on=False, zorder=10, transform=trans)
            if min_basis not in tick_positions:
                ax.plot([min_basis, min_basis], [y_bottom, y_top], 'k-', linewidth=1.5, clip_on=False, zorder=10, transform=trans)
            if max_basis not in tick_positions:
                ax.plot([max_basis, max_basis], [y_bottom, y_top], 'k-', linewidth=1.5, clip_on=False, zorder=10, transform=trans)
            ax.text((min_basis + max_basis) / 2, y_label, f"C$_{{{carbon}}}$", ha='center', va='bottom',
                   fontsize=7, zorder=10, transform=trans, clip_on=False)
    else:
        plt.tight_layout()
    
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    script_dir = Path(__file__).parent
    outputs_dir = script_dir.parent / "outputs"
    h200_folder = outputs_dir / "h200_8_gpu_uncompressed"
    a100_folder = outputs_dir / "a100_4gpu_compressed"
    orca_folder = outputs_dir / "orca"
    output_dir = script_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    isomers = load_size_profiling_isomers()
    if not isomers:
        print("Warning: Could not load size_profiling.json")
        return
    print(f"Filtering to {len(isomers)} isomers from size_profiling.json")
    
    print("Extracting data...")
    h200_df = extract_gpu_data(h200_folder, r"EXESS (8 $\times$ H200)", len(isomers) - 1)
    a100_df = extract_gpu_data(a100_folder, r"EXESS (4 $\times$ A100)", len(isomers) - 1)
    orca_df = extract_orca_data(orca_folder, isomers)
    print(f"  H200: {len(h200_df)}, A100: {len(a100_df)}, ORCA: {len(orca_df)}")
    
    all_data = pd.concat([h200_df, a100_df, orca_df], ignore_index=True)
    if len(all_data) == 0:
        print("Error: No data found!")
        return
    
    basis_to_carbon = get_basis_to_carbon_mapping(isomers, h200_folder, a100_folder, orca_folder)
    print(f"  Mapped {len(basis_to_carbon)} basis function counts to carbon atoms")
    
    # Load D4 energies
    d4_csv_path = outputs_dir / "dftd4" / "dftd4_results.csv"
    d4_energies = {}
    if d4_csv_path.exists():
        d4_df = pd.read_csv(d4_csv_path)
        d4_energies = {row['system']: float(row['d4_energy_hartree']) 
                      for _, row in d4_df[d4_df['functional'] == 'revDSD-PBEP86-D4'].iterrows()}
        print(f"  Loaded {len(d4_energies)} D4 energies from CSV (revDSD-PBEP86-D4)")
    
    # Load ORCA energies
    print("\nCalculating energy differences...")
    topology_to_isomer = {idx: name for idx, name in enumerate(isomers, 1) if idx > 0}
    orca_csv_path = script_dir / "orca_data.csv"
    orca_energies_by_topology = {}
    orca_isomers_by_topology = {}
    orca_n_basis_by_topology = {}
    
    if orca_csv_path.exists():
        orca_df = pd.read_csv(orca_csv_path)
        orca_df_compas3x = orca_df[(orca_df['isomer'].str.contains('compas3x', case=False, na=False)) &
                                   (orca_df['basis_combo_id'] == 'qz_riri')]
        used_orca_indices = set()
        for topology_idx, isomer_name in topology_to_isomer.items():
            matching = orca_df_compas3x[orca_df_compas3x['isomer'] == f"compas3x_{isomer_name}"]
            if len(matching) > 0:
                for idx, row in matching.iterrows():
                    if idx not in used_orca_indices:
                        orca_energies_by_topology[topology_idx] = float(row['total_energy_hartree'])
                        orca_isomers_by_topology[topology_idx] = row['isomer']
                        orca_n_basis_by_topology[topology_idx] = int(row['num_basis_functions'])
                        used_orca_indices.add(idx)
                        break
        print(f"  Loaded {len(orca_energies_by_topology)} ORCA energies from CSV (matched by exact isomer name, qz_riri basis only)")
    
    # Compare energies
    energy_diffs = []
    for topology_idx in range(1, len(isomers) + 1):
        energies = {}
        energy_components = {}
        exess_isomer = topology_to_isomer.get(topology_idx, "unknown")
        
        # Extract EXESS energies
        for folder, system_name in [(h200_folder, r"EXESS (8 $\times$ H200)"),
                                    (a100_folder, r"EXESS (4 $\times$ A100)")]:
            json_file = folder / f"size_profiling_topology_{topology_idx}.json"
            if json_file.exists():
                orca_n_basis = orca_n_basis_by_topology.get(topology_idx)
                orca_isomer = orca_isomers_by_topology.get(topology_idx)
                total_energy, components = extract_exess_energy(json_file, orca_n_basis, orca_isomer, d4_energies)
                if total_energy is not None:
                    energies[system_name] = total_energy
                    energy_components[system_name] = {"isomer": exess_isomer, **components}
        
        # Extract ORCA energy
        if topology_idx in orca_energies_by_topology:
            orca_isomer = orca_isomers_by_topology.get(topology_idx, "unknown")
            energies[r"ORCA (104 cores, 2 $\times$ Sapphire Rapid)"] = orca_energies_by_topology[topology_idx]
            energy_components[r"ORCA (104 cores, 2 $\times$ Sapphire Rapid)"] = {"isomer": orca_isomer}
        
        if len(energies) >= 2:
            energy_values = list(energies.values())
            max_diff = max(energy_values) - min(energy_values)
            avg_diff = np.mean([abs(energy_values[i] - energy_values[j]) 
                              for i in range(len(energy_values)) for j in range(i+1, len(energy_values))])
            energy_diffs.append((topology_idx, max_diff, avg_diff))
            
            threshold_hartree = 0.5 / 2625.5
            if max_diff > threshold_hartree:
                print(f"  Topology {topology_idx}: *** WARNING: Large difference *** - "
                      f"difference = {max_diff:.6e} Hartree ({max_diff * 2625.5:.4f} kJ/mol)")
                print(f"    Energies: {energies}")
                for system_name, components in energy_components.items():
                    print(f"    {system_name}:")
                    print(f"      Isomer: {components.get('isomer', 'unknown')}")
                    if 'HF' in components:
                        print(f"      HF: {components['HF']:.10f} Hartree")
                        print(f"      MP2_OS: {components['MP2_OS']:.10f} Hartree")
                        print(f"      MP2_SS: {components['MP2_SS']:.10f} Hartree")
                        print(f"      D4: {components['D4']:.10f} Hartree")
                        print(f"      Total: {energies[system_name]:.10f} Hartree")
                    else:
                        print(f"      Total: {energies[system_name]:.10f} Hartree")
    
    if energy_diffs:
        max_energy_diff = max(diff[1] for diff in energy_diffs)
        avg_energy_diff = np.mean([diff[2] for diff in energy_diffs])
        print(f"  Maximum energy difference: {max_energy_diff:.6e} Hartree")
        print(f"  Average energy difference: {avg_energy_diff:.6e} Hartree")
    
    print("\nGenerating plots...")
    plot_comparison(all_data, output_dir / "system_comparison_timings.png", 'timings', basis_to_carbon)
    plot_comparison(all_data, output_dir / "system_comparison_tflops.png", 'tflops', basis_to_carbon)
    print("Done!")


if __name__ == "__main__":
    main()
