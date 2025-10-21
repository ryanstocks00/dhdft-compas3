#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import sys

sys.path.append((Path(__file__).parent.parent / "inputs").as_posix())
import common


def parse_qmmbe_json(path: Path, d4_energy) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    qmmbe = data.get("qmmbe", {})

    scf_energy = float(qmmbe.get("expanded_hf_energy"))
    pt2_os_energy = float(qmmbe.get("expanded_mp2_os_correction"))
    pt2_ss_energy = float(qmmbe.get("expanded_mp2_ss_correction"))

    n_basis_functions = qmmbe.get("nmers")[0][0].get("num_basis_fns")

    dhdft_total_energy = scf_energy + pt2_os_energy + pt2_ss_energy + d4_energy

    row = {
        "filename": path.name,
        "total_energy_hartree": dhdft_total_energy,
        "scf_energy_hartree": scf_energy,
        "pt2_os_correction_hartree": pt2_os_energy,
        "pt2_ss_correction_hartree": pt2_ss_energy,
        "d4_energy_hartree": d4_energy,
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
        "mp2_time_s": float(qmmbe.get("mp2_time")),
        "b_formation_time_s": float(qmmbe.get("b_formation_time")),
        "diag_time_s": float(qmmbe.get("diag_time")),
        "ri_fock_time_s": float(qmmbe.get("ri_fock_time")),
        "xc_time_s": float(qmmbe.get("xc_time")),
        "basis_transforms_time_s": float(qmmbe.get("transforms_time")),
        "total_tflop/s": float(qmmbe.get("tflops")),
        "scf_tflop/s": float(qmmbe.get("scf_tflops")),
        "mp2_tflop/s": float(qmmbe.get("mp2_tflops")),
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


def main():
    ap = argparse.ArgumentParser(description="Extract EXESS JSON data to CSV")
    ap.add_argument(
        "-o", "--output", default="exess_data.csv", help="Output CSV filename"
    )
    args = ap.parse_args()

    exess_output_folder = Path(__file__).parent.parent / "outputs" / "exess"
    jsons_folder = exess_output_folder / "json_logs"
    json_files = sorted([p for p in jsons_folder.glob("*.json") if p.is_file()])

    rows: List[Dict[str, Any]] = []

    d4_energies: Dict[str, float] = {}
    dftd4_results_path = (
        Path(__file__).parent.parent / "outputs" / "dftd4" / "dftd4_results.csv"
    )
    with open(dftd4_results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d4_energies[r["system"]] = float(r["d4_energy_hartree"])

    for i, exess_batch in enumerate(common.exess_batches):
        batch_name = exess_batch.name()

        with open(exess_batch.input_file_path(), "r", encoding="utf-8") as f:
            input_json = json.load(f)
        for idx in range(len(exess_batch.isomers)):
            json_re = rf".*exess_inputs_{batch_name}_topology_{idx}.json$"
            matched = [p for p in json_files if re.match(json_re, p.name)]
            if len(matched) == 0:
                print(
                    f"Warning: No JSON file found for {batch_name} topology {idx} (isomer {exess_batch.isomers[idx].name})"
                )
            elif len(matched) > 1:
                print(
                    f"Warning: Multiple JSON files found for {batch_name} topology {idx}: {matched}"
                )
            elif (
                input_json.get("topologies")[idx]["xyz"]
                != exess_batch.isomers[idx].xyz_path.as_posix()
            ):
                print(
                    f"Warning: Mismatch in input XYZ for {batch_name} topology {idx}: {input_json.get('topologies')[idx]['xyz']} vs {exess_batch.isomers[idx].xyz_path.as_posix()}"
                )
                exit(1)
            else:
                d4_energy = d4_energies.get(exess_batch.isomers[idx].name)
                extracted_data = parse_qmmbe_json(matched[0], d4_energy)
                extracted_data["topology_index"] = idx
                extracted_data["batch_name"] = batch_name
                extracted_data["isomer_name"] = exess_batch.isomers[idx].name
                extracted_data["optimizer"] = exess_batch.isomers[idx].optimizer
                extracted_data["n_carbons"] = exess_batch.isomers[idx].carbons
                extracted_data["n_hydrogens"] = exess_batch.isomers[idx].hydrogens
                extracted_data["id"] = exess_batch.isomers[idx].id
                rows.append(extracted_data)

    field_order = [
        "isomer_name",
        "optimizer",
        "batch_name",
        "topology_index",
        "id",
        "filename",
        "total_energy_hartree",
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
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
