#!/usr/bin/env python3
"""
qmmbe_json_to_csv.py

Scan a folder for QMMBE JSON result files and extract key energy terms
into a single CSV.

Fields captured (when present):
- filename
- schema_version
- calculation_time_s
- method
- expanded_hf_energy_hartree
- expanded_mp2_os_correction_hartree
- expanded_mp2_ss_correction_hartree
- expanded_mp2_total_hartree            (hf + os + ss)
- num_iters
- nmers_count                           (flattened count)
- fragments_count_total                 (sum across nmers)
- basis_fns_total                       (sum across nmers)
- basis_fns_avg                         (avg across nmers)

Usage:
  python qmmbe_json_to_csv.py /path/to/folder -o qmmbe_energies.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float('nan')

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
    schema_version = data.get("schema_version") or qmmbe.get("schema_version") or ""
    calc_time = data.get("calculation_time")

    # Top-level “expanded_*” energies (preferred)
    hf = qmmbe.get("expanded_hf_energy")
    os = qmmbe.get("expanded_mp2_os_correction")
    ss = qmmbe.get("expanded_mp2_ss_correction")
    method = qmmbe.get("method") or ""

    # Iterations (top-level inside qmmbe)
    num_iters = qmmbe.get("num_iters")

    # Per-nmer details (for context metrics)
    nmers = _flatten_nmers(qmmbe.get("nmers"))
    nmers_count = len(nmers)
    fragments_count_total = 0
    basis_total = 0
    basis_vals = []

    for n in nmers:
        # fragments is a list of ints (indices); sum their length
        frags = n.get("fragments")
        if isinstance(frags, list):
            fragments_count_total += len(frags)

        nb = n.get("num_basis_fns")
        if isinstance(nb, int):
            basis_total += nb
            basis_vals.append(nb)

    basis_avg = (sum(basis_vals) / len(basis_vals)) if basis_vals else float('nan')

    # Compute MP2 total from expanded components if available
    hf_f = _safe_float(hf) if hf is not None else float('nan')
    os_f = _safe_float(os) if os is not None else float('nan')
    ss_f = _safe_float(ss) if ss is not None else float('nan')

    mp2_total = hf_f + os_f + ss_f

    row = {
        "filename": path.name,
        "schema_version": schema_version,
        "calculation_time_s": _safe_float(calc_time) if calc_time is not None else float('nan'),
        "method": method,
        "expanded_hf_energy_hartree": hf_f,
        "expanded_mp2_os_correction_hartree": os_f,
        "expanded_mp2_ss_correction_hartree": ss_f,
        "expanded_mp2_total_hartree": mp2_total,
        "num_iters": int(num_iters) if isinstance(num_iters, int) else "",
        "nmers_count": nmers_count,
        "fragments_count_total": fragments_count_total,
        "basis_fns_total": basis_total if basis_total else "",
        "basis_fns_avg": basis_avg,
    }
    return row

def main():
    ap = argparse.ArgumentParser(description="Extract QMMBE JSON energies to CSV")
    ap.add_argument("folder", help="Folder containing *.json files")
    ap.add_argument("-o", "--output", default="qmmbe_energies.csv", help="Output CSV filename")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    files = sorted([p for p in folder.glob("*.json") if p.is_file()])

    if not files:
        print(f"No *.json files found in {folder}")
        return

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            rows.append(parse_qmmbe_json(fp))
        except Exception as e:
            # If one file is malformed, keep going but note it
            rows.append({
                "filename": fp.name,
                "schema_version": "",
                "calculation_time_s": float('nan'),
                "method": "",
                "expanded_hf_energy_hartree": float('nan'),
                "expanded_mp2_os_correction_hartree": float('nan'),
                "expanded_mp2_ss_correction_hartree": float('nan'),
                "expanded_mp2_total_hartree": float('nan'),
                "num_iters": "",
                "nmers_count": "",
                "fragments_count_total": "",
                "basis_fns_total": "",
                "basis_fns_avg": float('nan'),
            })

    field_order = [
        "filename",
        "schema_version",
        "calculation_time_s",
        "method",
        "expanded_hf_energy_hartree",
        "expanded_mp2_os_correction_hartree",
        "expanded_mp2_ss_correction_hartree",
        "expanded_mp2_total_hartree",
        "num_iters",
        "nmers_count",
        "fragments_count_total",
        "basis_fns_total",
        "basis_fns_avg",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()

