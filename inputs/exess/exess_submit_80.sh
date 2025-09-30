#!/bin/bash
#SBATCH -A m4265
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 30:00
#SBATCH -N 1

export EXESS_PATH="/global/homes/r/ryans/HERMES"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation

mkdir -p "$OUT_DIR"
pushd "$OUT_DIR" || exit
echo "Running input $INPUT_FILENAME.json and saving to $OUTPUT_FILENAME"
srun -N 1 --export=ALL --ntasks-per-node=5 --gpus-per-node=4 $EXESS_PATH/build/exess "$INPUT_FILENAME" 2>&1 | tee "$OUTPUT_FILENAME"
popd || exit
