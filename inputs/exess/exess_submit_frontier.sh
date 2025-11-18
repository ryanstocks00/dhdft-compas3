#!/bin/bash
#SBATCH -A CHM213
#SBATCH --gpus-per-node=8
#SBATCH --partition=batch
#SBATCH --time=90:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9

export EXESS_PATH="/lustre/orion/scratch/ryan/chm213/exess"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation

source ${EXESS_PATH}/tools/frontier_env_setup_6.3.sh

mkdir -p "$OUT_DIR"
pushd "$OUT_DIR" || exit
echo "Running input $INPUT_FILENAME.json and saving to $OUTPUT_FILENAME"
srun --export=ALL $EXESS_PATH/build/exess "$INPUT_FILENAME" 2>&1 | tee "$OUTPUT_FILENAME"
popd || exit
