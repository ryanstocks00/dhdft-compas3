import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

machine = "perlmutter"

# for batch in common.exess_batches:
# for batch in common.exess_pah335_batches:
# for batch in common.exess_batches + common.exess_pah335_batches:
for batch in common.exess_pah335_pbe_batches:
    input_file = batch.input_file_path()
    output_file = batch.output_file_path()
    output_folder = output_file.parent
    print("Batch", batch.name(), "with", len(batch.isomers), "isomers")
    if machine == "perlmutter":
        submit_command = f"""sbatch --export=OUT_DIR={output_folder.resolve()},INPUT_FILENAME={input_file.resolve()},OUTPUT_FILENAME={output_file.name}  exess_submit_80.sh"""
    elif machine == "gadi_a100":
        submit_command = f"""qsub -v OUT_DIR={output_folder.resolve()},INPUT_FILENAME={input_file.resolve()},OUTPUT_FILENAME={output_file.name}  exess_submit_gadi_4_a100.sh"""
    else:
        raise ValueError(f"Unknown machine {machine}")
    print(submit_command)
    os.system(submit_command)
