#!/bin/bash
#PBS -P kx58
#PBS -q normalsr
#PBS -l walltime=48:00:00
#PBS -l mem=512GB
#PBS -l jobfs=400GB
#PBS -l software=orca
#PBS -l ncpus=104
#PBS -l storage=scratch/kx58

module load orca/6.0.1

mkdir -p "$OUTPUT_FOLDER"
pushd "$OUTPUT_FOLDER" || exit

rm "$INPUT_FILE".*

cp "$INPUT_FOLDER"/"$INPUT_FILE".inp "$INPUT_FILE".inp

$ORCA_PATH/orca "$INPUT_FILE".inp | tee "$OUTPUT_FILE"

rm "$INPUT_FILE.gbw" "$INPUT_FILE.densities" "$INPUT_FILE.bibtex" "$INPUT_FILE.densitiesinfo" "$INPUT_FILE".inp "$INPUT_FILE*.tmp*" "$INPUT_FILE".bas*
