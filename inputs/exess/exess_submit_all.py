import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

for batch in common.exess_batches:
    input_file = batch.input_file_path()
    output_file = batch.output_file_path()
    output_folder = output_file.parent
    print("Batch", batch.name(), "with", len(batch.isomers), "isomers")
    sbatch_command = f"""sbatch --export=OUT_DIR={output_folder.resolve()},INPUT_FILENAME={input_filename.resolve()},OUTPUT_FILENAME={output_file.name}  exess_submit_80.sh"""
    print(sbatch_command)
    os.system(sbatch_command)
