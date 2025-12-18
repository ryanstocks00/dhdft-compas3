#!/usr/bin/env python3
"""Generate supporting information document for DFT functional comparison paper."""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    exess_csv = project_root / "analysis" / "exess_data.csv"
    plots_dir = script_dir / "plots"
    table_file = script_dir / "compas3x_benchmarks.tex"
    latex_file = script_dir / "supporting_information.tex"
    
    # Check if EXESS CSV exists
    if not exess_csv.exists():
        print(f"ERROR: EXESS data file not found at {exess_csv}")
        sys.exit(1)
    
    # Step 1: Generate plots with min reference method (no linear correction)
    print("Generating benchmark plots (using minimum reference method)...")
    benchmark_script = script_dir / "benchmark_compas3x.py"
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--exess-csv", str(exess_csv),
        "--output-dir", str(plots_dir),
        "--output", str(table_file.with_suffix('.txt')),
        "--reference-method", "min"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Failed to generate plots")
        sys.exit(1)
    
    # Save plots directory before second run (which will overwrite plots)
    plots_backup = script_dir / "plots_backup"
    if plots_backup.exists():
        shutil.rmtree(plots_backup)
    shutil.copytree(plots_dir, plots_backup)
    
    # Step 2: Generate table with linear_fit to get gradient/offset columns
    print("Generating statistics table with gradient and offset...")
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--exess-csv", str(exess_csv),
        "--output-dir", str(plots_dir),
        "--output", str(table_file.with_suffix('.txt')),
        "--reference-method", "linear_fit"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Failed to generate table")
        sys.exit(1)
    
    # Restore plots from min run (overwrite the linear_fit plots)
    print("Restoring plots with minimum reference method...")
    for plot_file in plots_backup.glob("compas3x_*_vs_revdsd.png"):
        shutil.copy2(plot_file, plots_dir / plot_file.name)
    
    # Clean up backup
    shutil.rmtree(plots_backup)
    
    # Step 3: Compile LaTeX document
    print("\nCompiling LaTeX document...")
    latex_dir = latex_file.parent
    latex_filename = latex_file.name
    
    # Run pdflatex twice to resolve references
    for run_num in [1, 2]:
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory", str(latex_dir),
            str(latex_filename)
        ]
        subprocess.run(cmd)
    
    pdf_file = latex_file.with_suffix('.pdf')
    if pdf_file.exists():
        print(f"\nSUCCESS: Supporting information PDF generated at {pdf_file}")
    else:
        print(f"\nWARNING: PDF file not found. Check LaTeX compilation output.")


if __name__ == "__main__":
    main()
