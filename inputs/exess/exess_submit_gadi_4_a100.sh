#!/bin/bash
#PBS -P kx58
#PBS -q dgxa100
#PBS -l walltime=0:45:00
#PBS -l mem=1000GB
#PBS -l jobfs=100GB
#PBS -l ncpus=64
#PBS -l ngpus=4
#PBS -l storage=scratch/kx58

module load cuda/12.8.0 openmpi/4.1.7 hdf5/1.12.1 intel-mkl/2023.2.0

export EXESS_PATH="/scratch/kx58/rs2200/exess"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation

mkdir -p "$OUT_DIR"
pushd "$OUT_DIR" || exit
echo "Running input $INPUT_FILENAME.json and saving to $OUTPUT_FILENAME"
mpirun -np 5 $EXESS_PATH/build/exess "$INPUT_FILENAME" 2>&1 | tee "$OUTPUT_FILENAME"
popd || exit
