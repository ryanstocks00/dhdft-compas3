#!/bin/bash
#PBS -P kx58
#PBS -q hugemem
#PBS -l walltime=48:00:00
#PBS -l mem=700GB
#PBS -l jobfs=700GB
#PBS -l software=orca
#PBS -l ncpus=24
#PBS -l storage=scratch/kx58

module load orca/6.0.1

mkdir -p "$OUTPUT_FOLDER"
pushd "$OUTPUT_FOLDER" || exit

rm "$INPUT_FILE".*

cp "$INPUT_FOLDER"/"$INPUT_FILE".inp "$INPUT_FILE".inp

$ORCA_PATH/orca "$INPUT_FILE".inp 2>&1 | tee -p "$OUTPUT_FILE"

rm "$INPUT_FILE.gbw" "$INPUT_FILE.densities" "$INPUT_FILE.bibtex" "$INPUT_FILE.densitiesinfo" "$INPUT_FILE".inp "$INPUT_FILE*.tmp*" "$INPUT_FILE".bas*
