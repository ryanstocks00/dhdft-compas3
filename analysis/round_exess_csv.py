#!/usr/bin/env python3
"""
Re-write an EXESS-style CSV with reduced numeric precision (energies, z displacement,
times, TFLOP/s). Streams row-by-row; safe for large files.

Example:
  python analysis/round_exess_csv.py analysis/exess_data.csv -o analysis/exess_data_rounded.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from exess_csv_rounding import round_exess_csv_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Input CSV path")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite input)",
    )
    args = ap.parse_args()
    in_path = args.input
    out_path = args.output if args.output is not None else in_path
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    with in_path.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            print("ERROR: CSV has no header", file=sys.stderr)
            sys.exit(1)
        fieldnames = list(reader.fieldnames)

        # Write to temp file if in-place (same path)
        if out_path == in_path:
            fd, tmp_name = tempfile.mkstemp(
                suffix=".csv", prefix="exess_round_", dir=in_path.parent
            )
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as fout:
                    writer = csv.DictWriter(
                        fout, fieldnames=fieldnames, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(round_exess_csv_row(dict(row)))
                os.replace(tmp_name, out_path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        else:
            with out_path.open("w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(
                    fout, fieldnames=fieldnames, extrasaction="ignore"
                )
                writer.writeheader()
                for row in reader:
                    writer.writerow(round_exess_csv_row(dict(row)))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
