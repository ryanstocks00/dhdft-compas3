#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math
import sys

# Make 'inputs/common.py' importable
sys.path.append((Path(__file__).parent.parent / "inputs").as_posix())
import common  # noqa: E402

# ----------------- regex used for .property.txt parsing -----------------

SECTION_START = re.compile(r"^\s*\$(\S+)")
SECTION_END = re.compile(r"^\s*\$End\s*$")
KEY_FINDER = re.compile(r"^\s*&([A-Za-z0-9_]+)")
NUM_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
STR_RE = r'"([^"]*)"'
VALUE_FINDER = re.compile(rf"{NUM_RE}|{STR_RE}")

# ----------------- property file parser -----------------

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

    if calc_status.get("STATUS", "") != "NORMAL TERMINATION":
        print("Not normal termination:", calc_status.get("STATUS", ""))
        raise ValueError("Calculation did not terminate normally.")

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

    out_meta["multiplicity"] = str(to_int(info.get("MULT", "")))
    out_meta["charge"] = str(to_int(info.get("CHARGE", "")))
    out_meta["num_atoms"] = str(to_int(info.get("NUMOFATOMS", "")))

    # NEW: total number of basis functions (no aux*)
    out_meta["num_basis_functions"] = str(to_int(info.get("NUMOFBASISFUNCTS", "")))

    # Merge meta (as strings) into the dict we’ll later write
    combined: Dict[str, float | str] = {}
    combined.update(out_meta)
    combined.update(out)
    return combined  # type: ignore[return-value]

# --- helpers to get (C,H) counts and compute group energies -----------------

CH_PAT = re.compile(r"[Cc]\s*(\d+)\s*[Hh]\s*(\d+)")

def infer_ch_counts(isomer_name: str, isomer_obj: Any) -> Tuple[int | None, int | None]:
    """
    Try to infer (num_carbons, num_hydrogens) from the isomer name like 'C6H6_benzene'
    or from attributes on the isomer object (num_carbons/num_hydrogens).
    Returns (None, None) if unavailable.
    """
    if isinstance(isomer_name, str):
        m = CH_PAT.search(isomer_name)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                pass

    # Fallback to attributes if present
    nC = getattr(isomer_obj, "num_carbons", None)
    nH = getattr(isomer_obj, "num_hydrogens", None)
    try:
        nC = int(nC) if nC is not None else None
    except Exception:
        nC = None
    try:
        nH = int(nH) if nH is not None else None
    except Exception:
        nH = None

    return nC, nH

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# --- .out parsing: timings & iterations -----------------------------------

RE_FLOAT_SEC = r"([0-9]+(?:\.[0-9]+)?)\s*sec"

def _find_float_after(label_regex: str, lines: List[str]) -> float:
    rx = re.compile(label_regex, re.IGNORECASE)
    for ln in lines:
        m = rx.search(ln)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return float("nan")

def _parse_total_run_time(text: str) -> float:
    """
    Parse 'TOTAL RUN TIME: 0 days 0 hours 3 minutes 17 seconds 996 msec'
    Return seconds as float.
    Also supports 'Total time .... 82.900 sec' as fallback.
    """
    m = re.search(
        r"TOTAL RUN TIME:\s*(?:(\d+)\s*days?)?\s*(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*minutes?)?\s*(?:(\d+)\s*seconds?)?\s*(?:(\d+)\s*msec)?",
        text,
        flags=re.IGNORECASE,
    )
    if m and any(g is not None for g in m.groups()):
        days = int(m.group(1) or 0)
        hours = int(m.group(2) or 0)
        minutes = int(m.group(3) or 0)
        seconds = int(m.group(4) or 0)
        msec = int(m.group(5) or 0)
        return days * 86400 + hours * 3600 + minutes * 60 + seconds + (msec / 1000.0)

    # Fallback: a single "Total time .... XX sec" (SCF block etc.)
    m2 = re.search(r"^\s*Total time\s*\.*\s*([0-9.]+)\s*sec", text, flags=re.IGNORECASE | re.MULTILINE)
    if m2:
        return float(m2.group(1))

    return float("nan")

def _parse_scf_iterations(text: str) -> int | None:
    m = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def parse_out_metrics(path: Path) -> Dict[str, float | int | None]:
    """
    Extract timings & iteration counts from an ORCA .out file.
    Returns numeric seconds for timings and an int for scf_iterations (or None).
    Unknown fields -> NaN / None.
    """
    if not path.exists():
        return {
            "startup_time_s": float("nan"),
            "scf_time_s": float("nan"),
            "property_time_s": float("nan"),
            "rij_k_time_s": float("nan"),
            "xc_integration_time_s": float("nan"),
            "mp2_time_s": float("nan"),
            "total_time_s": float("nan"),
            "scf_iterations": None,
        }

    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    # End-of-file module timings
    startup_s = _find_float_after(r"Startup\s+calculation\s*\.*\s*" + RE_FLOAT_SEC, lines)
    scf_iter_s = _find_float_after(r"SCF\s+iterations\s*\.*\s*" + RE_FLOAT_SEC, lines)
    prop_s = _find_float_after(r"Property\s+calculations\s*\.*\s*" + RE_FLOAT_SEC, lines)
    mp2_s = _find_float_after(r"MP2\s+module\s*\.*\s*" + RE_FLOAT_SEC, lines)

    # SCF sub-timings (from Fock matrix formation breakdown)
    rij_k_s = _find_float_after(r"RI-JK\s*\.*\s*" + RE_FLOAT_SEC, lines)
    xc_int_s = _find_float_after(r"XC\s+integration\s*\.*\s*" + RE_FLOAT_SEC, lines)

    total_run_s = _parse_total_run_time(txt)
    scf_cycles = _parse_scf_iterations(txt)

    return {
        "startup_time_s": startup_s,
        "scf_time_s": scf_iter_s,
        "property_time_s": prop_s,
        "rij_k_time_s": rij_k_s,
        "xc_integration_time_s": xc_int_s,
        "mp2_time_s": mp2_s,
        "total_time_s": total_run_s,
        "scf_iterations": scf_cycles,
    }

# ----------------- main ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract ORCA .property.txt energies to CSV (+ .out timings)"
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

    rows: List[Dict[str, Any]] = []

    for orca_calc in common.orca_calculations:
        fp_prop = orca_calc.output_filepath().with_suffix(".property.txt")
        if not fp_prop.exists() or not fp_prop.is_file():
            print(f"Warning: Expected file {fp_prop} for {orca_calc.input_filename} not found.")
            continue

        sections = parse_property_file(fp_prop)
        record = extract_fields(sections)

        # Add filename (base) for traceability
        record = {"filename": fp_prop.name, **record}
        record["isomer"] = orca_calc.isomer.name
        record["basis_combo_id"] = orca_calc.basis_id
        record["primary_basis"] = orca_calc.primary_basis
        record["scf_aux_basis"] = orca_calc.scf_aux_basis or ""
        record["ri_aux_basis"] = orca_calc.ri_aux_basis or ""

        # Group key (C,H) if we can infer it
        nC, nH = infer_ch_counts(record["isomer"], orca_calc.isomer)
        record["num_carbons"] = nC
        record["num_hydrogens"] = nH

        # Timings / iterations from the .out file
        fp_out = orca_calc.output_filepath().with_suffix(".out")
        out_metrics = parse_out_metrics(fp_out)
        record.update(out_metrics)

        rows.append(record)

    # Group by (C,H,basis_combo) for relative energies
    groups: Dict[Tuple[Any, Any, Any], List[float]] = {}
    for r in rows:
        nC = r.get("num_carbons")
        nH = r.get("num_hydrogens")
        basis_combo = r.get("basis_combo_id")
        E = safe_float(r.get("total_energy_hartree", float("nan")))
        groups.setdefault((nC, nH, basis_combo), []).append(E)

    group_min: Dict[Tuple[Any, Any, Any], float] = {}
    group_mean: Dict[Tuple[Any, Any, Any], float] = {}
    for key, vals in groups.items():
        clean = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        if clean:
            group_min[key] = min(clean)
            group_mean[key] = sum(clean) / len(clean)
        else:
            group_min[key] = float("nan")
            group_mean[key] = float("nan")

    for r in rows:
        nC = r.get("num_carbons")
        nH = r.get("num_hydrogens")
        basis_combo = r.get("basis_combo_id")
        E = safe_float(r.get("total_energy_hartree"))
        key = (nC, nH, basis_combo)
        Emin = group_min.get(key, float("nan"))
        Eavg = group_mean.get(key, float("nan"))
        r["isomerization_energy_hartree"] = (
            E - Emin if not math.isnan(E) and not math.isnan(Emin) else float("nan")
        )
        r["relative_energy_hartree"] = (
            E - Eavg if not math.isnan(E) and not math.isnan(Eavg) else float("nan")
        )

    # CSV column order
    preferred_order = [
        "isomer",
        "num_carbons",
        "num_hydrogens",
        "basis_combo_id",
        "primary_basis",
        "scf_aux_basis",
        "ri_aux_basis",
        "filename",
        "multiplicity",
        "charge",
        "num_atoms",
        # NEW
        "num_basis_functions",
        # timings / iterations
        "scf_iterations",
        "startup_time_s",
        "scf_time_s",
        "property_time_s",
        "rij_k_time_s",
        "xc_integration_time_s",
        "mp2_time_s",
        "total_time_s",
        # energies
        "total_energy_hartree",
        "isomerization_energy_hartree",
        "relative_energy_hartree",
        "scf_energy_hartree",
        "dft_eexchange_hartree",
        "dft_ecorr_hartree",
        "dft_exc_hartree",
        "dft_finalen_hartree",
        "mp2_ref_hartree",
        "mp2_corr_hartree",
        "mp2_total_hartree",
        "vdw_correction_hartree",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=preferred_order, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
