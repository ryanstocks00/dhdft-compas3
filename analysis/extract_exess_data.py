#!/usr/bin/env python3
"""
qmmbe_json_to_csv.py

Scan a folder for QMMBE JSON result files and extract key energy terms
into a single CSV.

Fields captured (when present):
- filename
- calculation_time_s
- expanded_hf_energy_hartree
- expanded_mp2_os_correction_hartree
- expanded_mp2_ss_correction_hartree
- expanded_mp2_total_hartree            (hf + os + ss)
- num_iters
- basis_fns_total                       (sum across nmers)

Usage:
  python qmmbe_json_to_csv.py /path/to/folder -o qmmbe_energies.csv
"""

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


import sys

sys.path.append((Path(__file__).parent.parent / "inputs").as_posix())
import common


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _flatten_nmers(nmers_field) -> List[Dict[str, Any]]:
    """
    Your example shows nmers as a list of lists, each inner list
    containing 1 dict. This flattens to a simple list of dicts.
    Works if it's already flat too.
    """
    if not isinstance(nmers_field, list):
        return []
    out = []
    for item in nmers_field:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    out.append(sub)
    return out


def parse_qmmbe_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    qmmbe = data.get("qmmbe", {}) or {}
    calc_time = data.get("calculation_time")

    # Top-level “expanded_*” energies (preferred)
    hf = qmmbe.get("expanded_hf_energy")
    os = qmmbe.get("expanded_mp2_os_correction")
    ss = qmmbe.get("expanded_mp2_ss_correction")

    # Iterations (top-level inside qmmbe)
    num_iters = qmmbe.get("num_iters")

    # Per-nmer details (for context metrics)
    nmers = _flatten_nmers(qmmbe.get("nmers"))
    basis_total = 0
    basis_vals = []

    for n in nmers:
        nb = n.get("num_basis_fns")
        if isinstance(nb, int):
            basis_total += nb
            basis_vals.append(nb)

    basis_avg = (sum(basis_vals) / len(basis_vals)) if basis_vals else float("nan")

    # Compute MP2 total from expanded components if available
    hf_f = _safe_float(hf) if hf is not None else float("nan")
    os_f = _safe_float(os) if os is not None else float("nan")
    ss_f = _safe_float(ss) if ss is not None else float("nan")

    mp2_total = hf_f + os_f + ss_f

    row = {
        "filename": path.name,
        "calculation_time_s": _safe_float(calc_time)
        if calc_time is not None
        else float("nan"),
        "expanded_hf_energy_hartree": hf_f,
        "expanded_mp2_os_correction_hartree": os_f,
        "expanded_mp2_ss_correction_hartree": ss_f,
        "expanded_mp2_total_hartree": mp2_total,
        "num_iters": int(num_iters) if isinstance(num_iters, int) else "",
        "basis_fns_total": basis_total if basis_total else "",
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="Extract EXESS JSON data to CSV")
    ap.add_argument(
        "-o", "--output", default="exess_data.csv", help="Output CSV filename"
    )
    args = ap.parse_args()

    folder = Path(__file__).parent.parent / "outputs" / "exess"
    jsons_folder = Path(__file__).parent.parent / "outputs" / "exess" / "json_logs"
    json_files = sorted([p for p in jsons_folder.glob("*.json") if p.is_file()])

    rows: List[Dict[str, Any]] = []

    for exess_batch in common.exess_batches:
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
                try:
                    extracted_data = parse_qmmbe_json(matched[0])
                    extracted_data["topology_index"] = idx
                    extracted_data["batch_name"] = batch_name
                    extracted_data["isomer_name"] = exess_batch.isomers[idx].name
                    rows.append(extracted_data)
                except Exception as e:
                    print(f"Error parsing {matched[0]}: {e}")

    field_order = [
        "isomer_name",
        "batch_name",
        "topology_index",
        "filename",
        "calculation_time_s",
        "expanded_hf_energy_hartree",
        "expanded_mp2_os_correction_hartree",
        "expanded_mp2_ss_correction_hartree",
        "expanded_mp2_total_hartree",
        "num_iters",
        "basis_fns_total",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
