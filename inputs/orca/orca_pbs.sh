#!/bin/bash
#PBS -P kx58
#PBS -q normalsr
#PBS -l walltime=0:30:00
#PBS -l mem=512GB
#PBS -l jobfs=400GB
#PBS -l software=orca
#PBS -l ncpus=104
#PBS -l storage=scratch/kx58

module load orca/6.0.1

# Set up working directory on job filesystem for better performance
WORK_DIR="$PBS_JOBFS"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR" || exit

# Copy input file to jobfs
cp "$INPUT_FOLDER"/"$INPUT_FILE".inp "$INPUT_FILE".inp

# Run ORCA in jobfs
$ORCA_PATH/orca "$INPUT_FILE".inp 2>&1 | tee -p "$INPUT_FILE".out

# Copy important output files back to gdata output folder
mkdir -p "$OUTPUT_FOLDER"
cp "$INPUT_FILE".out "$OUTPUT_FOLDER"/"$OUTPUT_FILE"

# Copy property files if they exist
if [ -f "$INPUT_FILE".property.txt ]; then
    cp "$INPUT_FILE".property.txt "$OUTPUT_FOLDER"/
fi

# Clean up jobfs
rm -f "$INPUT_FILE".*
