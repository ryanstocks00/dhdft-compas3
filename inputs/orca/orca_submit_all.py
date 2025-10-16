import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

for calc in common.orca_calculations:
    print("Submitting calculation for", calc)
    input_file = calc.input_filepath()
    output_file = calc.output_filepath()
    input_folder = input_file.parent
    output_folder = output_file.parent
    qsub_command = f"""qsub -v OUTPUT_FOLDER={output_folder.resolve()},INPUT_FOLDER={input_folder.resolve()},INPUT_FILE={input_file.stem},OUTPUT_FILE={output_file.name} orca_pbs.sh"""
    print(qsub_command)
    os.system(qsub_command)
