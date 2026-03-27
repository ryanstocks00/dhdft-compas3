#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

import sys

sys.path.append((Path(__file__).parent.parent / "inputs").as_posix())
import common

from exess_csv_rounding import round_exess_csv_row
from exess_plane_displacement import plane_metrics_from_xyz


def parse_qmmbe_json(path: Path, d4_energy) -> Dict[str, Any]:
    # Use json.load with file handle for better performance
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qmmbe = data.get("qmmbe", {})

    scf_energy = float(qmmbe.get("expanded_hf_energy"))
    pt2_os_raw = qmmbe.get("expanded_mp2_os_correction")
    pt2_ss_raw = qmmbe.get("expanded_mp2_ss_correction")
    # For total energy calculation, use 0.0 if MP2 corrections are missing
    pt2_os_energy = pt2_os_raw if pt2_os_raw is not None else 0.0
    pt2_ss_energy = pt2_ss_raw if pt2_ss_raw is not None else 0.0

    n_basis_functions = qmmbe.get("nmers")[0][0].get("num_basis_fns")

    # Use 0.0 if D4 energy is None (e.g., for functionals without D4 support like HCTH407, SVWN5)
    d4_energy_value = d4_energy if d4_energy is not None else 0.0
    dhdft_total_energy = scf_energy + pt2_os_energy + pt2_ss_energy + d4_energy_value

    row = {
        "total_energy_hartree": dhdft_total_energy,
        "scf_energy_hartree": scf_energy,
        "pt2_os_correction_hartree": (pt2_os_raw if pt2_os_raw is not None else None),
        "pt2_ss_correction_hartree": (pt2_ss_raw if pt2_ss_raw is not None else None),
        "d4_energy_hartree": d4_energy if d4_energy is not None else None,
        "xc_energy_hartree": float(qmmbe.get("expanded_xc_energy")),
        "nuc_repulsion_energy_hartree": float(
            qmmbe.get("expanded_nuc_repulsion_energy")
        ),
        "elec_energy_hartree": float(qmmbe.get("expanded_elec_energy")),
        "n_primary_basis_functions": int(n_basis_functions),
        "n_atoms": int(qmmbe.get("n_atoms")),
        "n_scf_iterations": int(qmmbe.get("num_iters")),
        "total_time_s": float(qmmbe.get("total_time")),
        "scf_time_s": float(qmmbe.get("scf_time")),
        "mp2_time_s": (
            float(qmmbe.get("mp2_time")) if qmmbe.get("mp2_time") is not None else 0.0
        ),
        "b_formation_time_s": float(qmmbe.get("b_formation_time")),
        "diag_time_s": float(qmmbe.get("diag_time")),
        "ri_fock_time_s": float(qmmbe.get("ri_fock_time")),
        "xc_time_s": float(qmmbe.get("xc_time")),
        "basis_transforms_time_s": float(qmmbe.get("transforms_time")),
        "total_tflop/s": float(qmmbe.get("tflops")),
        "scf_tflop/s": float(qmmbe.get("scf_tflops")),
        "mp2_tflop/s": (
            float(qmmbe.get("mp2_tflops"))
            if qmmbe.get("mp2_tflops") is not None
            else 0.0
        ),
        "b_formation_tflop/s": float(qmmbe.get("b_tflops")),
        "ri_fock_tflop/s": float(qmmbe.get("ri_tflops")),
        "xc_tflop/s": float(qmmbe.get("xc_tflops")),
        "homo_hartree": (
            float(qmmbe.get("homo")) if qmmbe.get("homo") is not None else None
        ),
        "lumo_hartree": (
            float(qmmbe.get("lumo")) if qmmbe.get("lumo") is not None else None
        ),
    }
    if row["homo_hartree"] is not None:
        row["hlg_hartree"] = row["lumo_hartree"] - row["homo_hartree"]
    return row


def build_json_lookup(exess_output_folder: Path) -> Dict[Tuple[str, str, str, int], Path]:
    """Build a lookup dictionary for all JSON files.
    
    Returns: {(functional, basis, batch_name, topology_idx): Path}
    """
    lookup = {}
    json_logs_dirs = list(exess_output_folder.rglob("json_logs"))
    
    print(f"Scanning {len(json_logs_dirs)} json_logs directories...")
    for json_logs_dir in json_logs_dirs:
        # Extract functional and basis from path: .../functional/basis/json_logs
        parts = json_logs_dir.parts
        try:
            # Find the exess folder index
            exess_idx = parts.index("exess")
            if exess_idx + 2 < len(parts):
                functional = parts[exess_idx + 1]
                basis = parts[exess_idx + 2]
            else:
                continue
        except (ValueError, IndexError):
            continue
        
        # Pattern: *exess_inputs_{functional}_{basis}_{batch_name}_topology_{idx}.json
        # The filename has a variable prefix (e.g., lustre_orion_..._exess_inputs_)
        pattern = re.compile(
            rf".*exess_inputs_{re.escape(functional)}_{re.escape(basis)}_(.+?)_topology_(\d+)\.json$"
        )
        
        for json_file in json_logs_dir.glob("*.json"):
            match = pattern.match(json_file.name)
            if match:
                batch_name = match.group(1)
                topology_idx = int(match.group(2))
                key = (functional, basis, batch_name, topology_idx)
                if key in lookup:
                    print(f"Warning: Duplicate JSON file found for {key}: {json_file} and {lookup[key]}")
                lookup[key] = json_file
    
    print(f"Built lookup with {len(lookup)} JSON files")
    if len(lookup) == 0:
        print("Warning: No JSON files found in lookup! Check the pattern matching.")
        # Show a sample filename for debugging
        sample_files = list(exess_output_folder.rglob("*.json"))[:3]
        if sample_files:
            print(f"Sample JSON filenames found:")
            for f in sample_files:
                print(f"  {f.name}")
    return lookup


def process_batch(
    exess_batch,
    d4_energies: Dict[Tuple[str, str], float],
    json_lookup: Dict[Tuple[str, str, str, int], Path],
):
    """Process a single EXESS batch and return rows."""
    batch_name = exess_batch.name()
    functional = getattr(exess_batch, "functional", "revDSD-PBEP86-D4")
    basis = getattr(exess_batch, "basis", "def2-QZVPP")
    
    rows = []
    for idx in range(len(exess_batch.isomers)):
        # Use lookup instead of globbing
        lookup_key = (functional, basis, batch_name, idx)
        matched = json_lookup.get(lookup_key)
        
        if matched is None:
            print(
                f"Warning: No JSON file found for {batch_name} topology {idx} (isomer {exess_batch.isomers[idx].name})"
            )
            continue

        d4_key = (exess_batch.isomers[idx].name, functional)
        d4_energy = d4_energies.get(d4_key)
        if d4_energy is None:
            print(
                f"Warning: No D4 energy found for {exess_batch.isomers[idx].name} with functional {functional}"
            )
        extracted_data = parse_qmmbe_json(matched, d4_energy)
        extracted_data["functional"] = functional
        extracted_data["basis_set"] = basis
        extracted_data["isomer_name"] = exess_batch.isomers[idx].name
        # Map optimizer labels: xTB -> GFN2-xTB, DFT -> CAM-B3LYP-D3BJ
        optimizer_raw = exess_batch.isomers[idx].optimizer
        optimizer_map = {
            "xTB": "GFN2-xTB",
            "DFT": "CAM-B3LYP-D3BJ",
        }
        extracted_data["optimizer"] = optimizer_map.get(optimizer_raw, optimizer_raw)
        extracted_data["n_carbons"] = exess_batch.isomers[idx].carbons
        extracted_data["n_hydrogens"] = exess_batch.isomers[idx].hydrogens
        extracted_data["id"] = exess_batch.isomers[idx].id
        rows.append(extracted_data)
    
    return rows


def _build_isomer_plane_metrics(batches) -> Dict[str, Optional[Tuple[float, float]]]:
    """One (max |d|, mean |d|) per isomer name from its XYZ; None on failure."""
    out: Dict[str, Optional[Tuple[float, float]]] = {}
    for batch in batches:
        for isomer in batch.isomers:
            if isomer.name in out:
                continue
            try:
                out[isomer.name] = plane_metrics_from_xyz(isomer.xyz_path)
            except Exception as e:
                print(
                    f"Warning: plane metrics for {isomer.name} ({isomer.xyz_path}): {e}"
                )
                out[isomer.name] = None
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract EXESS JSON data to CSV")
    ap.add_argument(
        "-o", "--output", default="exess_data.csv", help="Output CSV filename"
    )
    ap.add_argument(
        "-j", "--jobs", type=int, default=None,
        help="Number of parallel jobs (default: number of CPU cores)"
    )
    ap.add_argument(
        "--include-pah335", action="store_true",
        help="Include PAH335 (G4(MP2)) results (excluded by default)"
    )
    args = ap.parse_args()

    exess_output_folder = Path(__file__).parent.parent / "outputs" / "exess"

    # Load D4 energies
    print("Loading D4 energies...")
    d4_energies: Dict[Tuple[str, str], float] = {}
    dftd4_results_path = (
        Path(__file__).parent.parent / "outputs" / "dftd4" / "dftd4_results.csv"
    )
    with open(dftd4_results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_functional_column = "functional" in reader.fieldnames
        for r in reader:
            # Handle both old format (no functional column) and new format (with functional column)
            if has_functional_column:
                functional = r["functional"]
            else:
                # Default to revDSD-PBEP86-D4 if functional column is missing
                functional = "revDSD-PBEP86-D4"
            key = (r["system"], functional)
            d4_energies[key] = float(r["d4_energy_hartree"])
    print(f"Loaded {len(d4_energies)} D4 energies")

    # Build JSON file lookup (this is the key optimization - do it once upfront)
    print("Building JSON file lookup...")
    json_lookup = build_json_lookup(exess_output_folder)

    # Collect all batches
    all_batches = []
    # Include PAH335 (G4(MP2)) batches only if explicitly requested
    if args.include_pah335:
        all_batches.extend(common.exess_pah335_batches)
        all_batches.extend(common.exess_pah335_pbe_batches)
        print("Including PAH335 (G4(MP2)) batches")
    else:
        print("Excluding PAH335 (G4(MP2)) batches (use --include-pah335 to include)")
    
    # Always include COMPAS batches
    all_batches.extend(common.exess_batches)
    all_batches.extend(common.exess_svwn_batches)
    all_batches.extend(common.exess_gga_batches)
    all_batches.extend(common.exess_mgga_batches)
    
    print(f"Processing {len(all_batches)} batches...")
    
    # Process batches in parallel (with fallback to sequential if multiprocessing fails)
    rows: List[Dict[str, Any]] = []
    num_jobs = args.jobs if args.jobs else cpu_count()
    
    try:
        process_func = partial(process_batch, d4_energies=d4_energies, json_lookup=json_lookup)
        with Pool(processes=num_jobs) as pool:
            results = pool.map(process_func, all_batches)
            for batch_rows in results:
                rows.extend(batch_rows)
        print(f"Processed {len(rows)} rows using {num_jobs} parallel workers")
    except Exception as e:
        print(f"Warning: Multiprocessing failed ({e}), falling back to sequential processing")
        for exess_batch in all_batches:
            batch_rows = process_batch(exess_batch, d4_energies, json_lookup)
            rows.extend(batch_rows)
        print(f"Processed {len(rows)} rows sequentially")

    print("Computing max/mean out-of-plane displacement from XYZ geometries...")
    z_by_name = _build_isomer_plane_metrics(all_batches)
    for r in rows:
        zm = z_by_name.get(r["isomer_name"])
        if zm is None:
            r["max_z_displacement"] = None
            r["mean_z_displacement"] = None
        else:
            r["max_z_displacement"], r["mean_z_displacement"] = zm

    # Find minimum energy isomer for each (C, H) group using revDSD-PBEP86-D4
    # Store the isomer_name that has minimum energy
    min_isomer_by_ch_revdsd: Dict[Tuple[int, int], str] = {}
    min_energy_by_ch_revdsd: Dict[Tuple[int, int], float] = {}
    for r in rows:
        functional = r.get("functional", "revDSD-PBEP86-D4")
        if functional != "revDSD-PBEP86-D4":
            continue
        key = (r["n_carbons"], r["n_hydrogens"])
        e = r.get("total_energy_hartree")
        if e is None:
            continue
        if key not in min_energy_by_ch_revdsd or e < min_energy_by_ch_revdsd[key]:
            min_energy_by_ch_revdsd[key] = e
            min_isomer_by_ch_revdsd[key] = r["isomer_name"]

    # Build lookup: (functional, basis_set, isomer_name) -> energy
    energy_lookup: Dict[Tuple[str, str, str], float] = {}
    for r in rows:
        key = (r.get("functional", "revDSD-PBEP86-D4"), r.get("basis_set", "def2-QZVPP"), r["isomer_name"])
        e = r.get("total_energy_hartree")
        if e is not None:
            energy_lookup[key] = e

    # Add isomerization energy = total_energy - energy of min_isomer (calculated with same functional/basis)
    for r in rows:
        key_ch = (r["n_carbons"], r["n_hydrogens"])
        min_isomer = min_isomer_by_ch_revdsd.get(key_ch)
        if min_isomer is None:
            r["isomerization_energy_hartree"] = None
            continue
        
        # Find the energy of the minimum isomer calculated with the same functional/basis
        functional = r.get("functional", "revDSD-PBEP86-D4")
        basis_set = r.get("basis_set", "def2-QZVPP")
        lookup_key = (functional, basis_set, min_isomer)
        ref_energy = energy_lookup.get(lookup_key)
        
        total_e = r.get("total_energy_hartree")
        r["isomerization_energy_hartree"] = (
            (total_e - ref_energy) if (ref_energy is not None and total_e is not None) else None
        )

    field_order = [
        "isomer_name",
        "optimizer",
        "functional",
        "basis_set",
        "id",
        "total_energy_hartree",
        "isomerization_energy_hartree",
        "scf_energy_hartree",
        "pt2_os_correction_hartree",
        "pt2_ss_correction_hartree",
        "d4_energy_hartree",
        "xc_energy_hartree",
        "nuc_repulsion_energy_hartree",
        "elec_energy_hartree",
        "homo_hartree",
        "lumo_hartree",
        "hlg_hartree",
        "n_primary_basis_functions",
        "n_atoms",
        "max_z_displacement",
        "mean_z_displacement",
        "n_carbons",
        "n_hydrogens",
        "n_scf_iterations",
        "total_time_s",
        "scf_time_s",
        "mp2_time_s",
        "b_formation_time_s",
        "diag_time_s",
        "ri_fock_time_s",
        "xc_time_s",
        "basis_transforms_time_s",
        "total_tflop/s",
        "scf_tflop/s",
        "mp2_tflop/s",
        "b_formation_tflop/s",
        "ri_fock_tflop/s",
        "xc_tflop/s",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            out = {fn: r.get(fn) for fn in field_order}
            writer.writerow(round_exess_csv_row(out))

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
