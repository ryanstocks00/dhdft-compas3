import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

for input_file in Path(__file__).parent.glob("orca_inputs/**/*.inp"):
    print("Submitting calculation for", input_file)
    qsub_command = f"""qsub -v OUTPUT_FOLDER={(Path(__file__).parent.parent.parent / 'outputs' / 'orca').resolve()},INPUT_FOLDER={input_file.parent.resolve()},INPUT_FILE={input_file.stem},OUTPUT_FILE={input_file.stem}.out orca_pbs.sh"""
    print(qsub_command)
    os.system(qsub_command)
