import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
import common

machine = "frontier"
use_multi_node = True  # Set to True to submit 256 batches in a single 256-node job
max_batches = 256  # Maximum number of batches to submit in multi-node mode

# for batch in common.exess_batches:
# for batch in common.exess_pah335_batches:
# for batch in common.exess_batches + common.exess_pah335_batches:
# for batch in common.exess_pah335_pbe_batches:
# batches = list(common.exess_svwn_batches)
batches = list(common.exess_gga_batches)

if machine == "frontier" and use_multi_node:
    # Multi-node mode: split batches into chunks of 256 and submit separate jobs
    script_dir = Path(__file__).parent
    total_batches = len(batches)
    num_jobs = (total_batches + max_batches - 1) // max_batches  # Ceiling division
    
    print(f"Total batches: {total_batches}")
    print(f"Submitting {num_jobs} job(s) with {max_batches} nodes each...")
    
    # Split batches into chunks and submit a job for each chunk
    for job_idx in range(num_jobs):
        start_idx = job_idx * max_batches
        end_idx = min(start_idx + max_batches, total_batches)
        chunk_batches = batches[start_idx:end_idx]
        
        # Create batch list file for this chunk
        batch_list_file = script_dir / f"batch_list_256_job{job_idx + 1}.txt"
        
        with open(batch_list_file, "w") as f:
            for batch in chunk_batches:
                input_file = batch.input_file_path()
                output_file = batch.output_file_path()
                output_folder = output_file.parent
                # Format: OUT_DIR:INPUT_FILENAME:OUTPUT_FILENAME
                f.write(f"{output_folder.resolve()}:{input_file.resolve()}:{output_file.name}\n")
        
        num_batches_in_chunk = len(chunk_batches)
        print(f"\nJob {job_idx + 1}/{num_jobs}: Created batch list file with {num_batches_in_chunk} batches: {batch_list_file}")
        
        if num_batches_in_chunk < max_batches:
            print(f"  Note: This job will use {num_batches_in_chunk} nodes (some nodes will be idle)")
        
        submit_command = f"""sbatch --export=BATCH_LIST_FILE={batch_list_file.resolve()} exess_submit_frontier.sh"""
        print(f"  Submitting: {submit_command}")
        os.system(submit_command)
else:
    # Single-batch mode: submit each batch as separate job
    for batch in batches:
        input_file = batch.input_file_path()
        output_file = batch.output_file_path()
        output_folder = output_file.parent
        print("Batch", batch.name(), "with", len(batch.isomers), "isomers")
        if machine == "perlmutter":
            submit_command = f"""sbatch --export=OUT_DIR={output_folder.resolve()},INPUT_FILENAME={input_file.resolve()},OUTPUT_FILENAME={output_file.name}  exess_submit_80.sh"""
        elif machine == "frontier":
            submit_command = f"""sbatch --export=OUT_DIR={output_folder.resolve()},INPUT_FILENAME={input_file.resolve()},OUTPUT_FILENAME={output_file.name}  exess_submit_frontier.sh"""
        elif machine == "gadi_a100":
            submit_command = f"""qsub -v OUT_DIR={output_folder.resolve()},INPUT_FILENAME={input_file.resolve()},OUTPUT_FILENAME={output_file.name}  exess_submit_gadi_4_a100.sh"""
        else:
            raise ValueError(f"Unknown machine {machine}")
        print(submit_command)
        os.system(submit_command)
