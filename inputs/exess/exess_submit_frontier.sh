#!/bin/bash
#SBATCH -A CHM213
#SBATCH --partition=batch
#SBATCH --nodes=256
#SBATCH --gpus-per-node=8
#SBATCH --time=90:00
#SBATCH --ntasks-per-node=9

export EXESS_PATH="/lustre/orion/scratch/ryan/chm213/exess"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation

source "${EXESS_PATH}/tools/frontier_env_setup_6.3.sh"

# Get list of nodes in this allocation
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )

i=0
for node in "${nodes[@]}"; do
    # One line per node in BATCH_LIST_FILE: OUT_DIR:INPUT_FILENAME:OUTPUT_FILENAME
    if [ -n "$BATCH_LIST_FILE" ] && [ -f "$BATCH_LIST_FILE" ]; then
        BATCH_LINE=$(sed -n "$((i + 1))p" "$BATCH_LIST_FILE")
        if [ -z "$BATCH_LINE" ]; then
            echo "Node $i ($node): no batch assignment, skipping."
            ((i++))
            continue
        fi
        IFS=":" read -r OUT_DIR INPUT_FILENAME OUTPUT_FILENAME <<< "$BATCH_LINE"
    else
        echo "Error: BATCH_LIST_FILE must be set for per-node batches"
        exit 1
    fi

    mkdir -p "$OUT_DIR"

    echo "Node $i ($node): running input $INPUT_FILENAME -> $OUTPUT_FILENAME"

    # Launch one step per node, in background
    srun --nodes=1 \
         --ntasks=9 \
         --ntasks-per-node=9 \
         --gpus-per-node=8 \
         --exclusive \
         --export=ALL \
         -w "$node" \
         /bin/bash -lc "cd '$OUT_DIR' && '$EXESS_PATH/build/exess' '$INPUT_FILENAME' 2>&1 | tee '$OUTPUT_FILENAME'" &

    ((i++))
done

wait
