import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

for input_file in Path(__file__).parent.glob("exess_inputs/*.json"):
    print("Submitting calculation for", input_file)
    sbatch_command = f"""sbatch --export=OUT_DIR={(Path(__file__).parent.parent.parent / 'outputs' / 'exess').resolve()},INPUT_FILENAME={input_file.resolve()},OUTPUT_FILENAME={input_file.stem}.out  exess_submit_80.sh"""
    print(sbatch_command)
    os.system(sbatch_command)
