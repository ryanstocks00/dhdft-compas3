#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import sys

sys.path.append((Path(__file__).parent.parent / "inputs").as_posix())
import common


SECTION_START = re.compile(r"^\s*\$(\S+)")
SECTION_END = re.compile(r"^\s*\$End\s*$")
NUM_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
STR_RE = r'"([^"]*)"'
VALUE_FINDER = re.compile(rf"{NUM_RE}|{STR_RE}")

KEY_FINDER = re.compile(r"^\s*&([A-Za-z0-9_]+)")


def parse_property_file(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse an ORCA .property.txt file into a dict:
      sections[section_name][key] = value (string).
    Values are extracted from the text AFTER the trailing ']' of the [&Type ...] block,
    preferring the first number there; if no number exists, the first quoted string.
    """
    sections: Dict[str, Dict[str, str]] = {}
    current_section = None

    SECTION_START = re.compile(r"^\s*\$(\S+)")
    SECTION_END = re.compile(r"^\s*\$End\s*$")
    KEY_FINDER = re.compile(r"^\s*&([A-Za-z0-9_]+)\b")
    NUMBER_FINDER = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    STRING_FINDER = re.compile(r'"([^"]*)"')

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mstart = SECTION_START.match(line)
            if mstart:
                current_section = mstart.group(1)
                sections.setdefault(current_section, {})
                continue

            if SECTION_END.match(line):
                current_section = None
                continue

            if not current_section:
                continue

            km = KEY_FINDER.match(line)
            if not km:
                continue

            key = km.group(1)

            # Slice the line to AFTER the last ']' so we skip [&Type "..."] etc.
            pos = line.rfind("]")
            tail = line[pos + 1 :] if pos != -1 else line[km.end() :]

            # Prefer the first number in the tail; if none, use the first quoted string
            nm = NUMBER_FINDER.search(tail)
            if nm:
                val = nm.group(1)
            else:
                sm = STRING_FINDER.search(tail)
                val = sm.group(1) if sm else None

            if val is not None:
                sections[current_section][key] = val

    return sections


def to_float(maybe_str: str) -> float:
    try:
        return float(maybe_str)
    except Exception:
        return float("nan")


def extract_fields(sections: Dict[str, Dict[str, str]]) -> Dict[str, float]:
    """
    Map ORCA sections/keys to the output fields.
    """
    out: Dict[str, float] = {}

    # Meta info (kept as strings for the CSV)
    out_meta: Dict[str, str] = {}
    calc_status = sections.get("Calculation_Status", {})
    out_meta["version"] = calc_status.get("VERSION", "")
    out_meta["progname"] = calc_status.get("PROGNAME", "")
    out_meta["status"] = calc_status.get("STATUS", "")

    # SCF energy
    scf = sections.get("SCF_Energy", {})
    out["scf_energy_hartree"] = to_float(scf.get("SCF_ENERGY", "nan"))

    # DFT energy / exchange-correlation pieces
    dft = sections.get("DFT_Energy", {})
    out["dft_eexchange_hartree"] = to_float(dft.get("EEXCHANGE", "nan"))
    out["dft_ecorr_hartree"] = to_float(dft.get("ECORR", "nan"))
    out["dft_exc_hartree"] = to_float(dft.get("EXC", "nan"))
    out["dft_finalen_hartree"] = to_float(dft.get("FINALEN", "nan"))

    # MP2 energies
    mp2 = sections.get("MP2_Energies", {})
    out["mp2_ref_hartree"] = to_float(mp2.get("REFENERGY", "nan"))
    out["mp2_corr_hartree"] = to_float(mp2.get("CORRENERGY", "nan"))
    out["mp2_total_hartree"] = to_float(mp2.get("TOTALENERGY", "nan"))

    # VdW/D4 correction (ORCA prints this under $VdW_Correction -> &VDW)
    vdw = sections.get("VdW_Correction", {})
    out["vdw_correction_hartree"] = to_float(vdw.get("VDW", "nan"))

    # Calculation info / totals
    info = sections.get("Calculation_Info", {})
    out["total_energy_hartree"] = to_float(info.get("TOTALENERGY", "nan"))

    # Useful context (ints)
    def to_int(s: str) -> int:
        try:
            return int(s)
        except Exception:
            return 0

    out_meta["mult"] = str(to_int(info.get("MULT", "")))
    out_meta["charge"] = str(to_int(info.get("CHARGE", "")))
    out_meta["natoms"] = str(to_int(info.get("NUMOFATOMS", "")))

    # Merge meta (as strings) into the dict we’ll later write
    # We’ll keep numbers as floats and meta as strings; the CSV writer will handle it.
    combined: Dict[str, float | str] = {}
    combined.update(out_meta)
    combined.update(out)
    return combined  # type: ignore[return-value]


def main():
    parser = argparse.ArgumentParser(
        description="Extract ORCA .property.txt energies to CSV"
    )
    parser.add_argument(
        "-o", "--output", default="orca_data.csv", help="Output CSV filename"
    )
    args = parser.parse_args()

    folder = Path(__file__).parent.parent / "outputs" / "orca"
    files = sorted(folder.glob("*.property.txt"))

    if not files:
        print(f"No *.property.txt files found in {folder}")
        return

    rows: List[Dict[str, float | str]] = []

    for orca_calc in common.orca_calculations:
        fp = orca_calc.output_filepath().with_suffix(".property.txt")
        if (not fp.exists() or not fp.is_file()):
            print(
                f"Warning: Expected file {fp} for {orca_calc.input_filename} not found."
            )
            continue
        sections = parse_property_file(fp)
        record = extract_fields(sections)
        # Add filename (base) for traceability
        record = {"filename": fp.name, **record}
        record["isomer"] = orca_calc.isomer.name
        record["primary_basis"] = orca_calc.primary_basis
        record["scf_aux_basis"] = orca_calc.scf_aux_basis or ""
        record["ri_aux_basis"] = orca_calc.ri_aux_basis or ""
        rows.append(record)

    # Determine CSV field order (stable and readable)
    preferred_order = [
        "isomer",
        "primary_basis",
        "scf_aux_basis",
        "ri_aux_basis",
        "filename",
        "version",
        "progname",
        "status",
        "mult",
        "charge",
        "natoms",
        "scf_energy_hartree",
        "dft_eexchange_hartree",
        "dft_ecorr_hartree",
        "dft_exc_hartree",
        "dft_finalen_hartree",
        "mp2_ref_hartree",
        "mp2_corr_hartree",
        "mp2_total_hartree",
        "vdw_correction_hartree",
        "total_energy_hartree",
    ]
    # Include any extra keys that happened to appear (future-proofing)
    all_keys = set().union(*(r.keys() for r in rows))
    fieldnames = preferred_order + [
        k for k in sorted(all_keys) if k not in preferred_order
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
