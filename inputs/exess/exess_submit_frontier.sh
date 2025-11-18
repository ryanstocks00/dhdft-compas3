#!/bin/bash
#SBATCH -A CHM213
#SBATCH --gpus-per-node=8
#SBATCH --partition=batch
#SBATCH --time=90:00
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=9

export EXESS_PATH="/lustre/orion/scratch/ryan/chm213/exess"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation

source ${EXESS_PATH}/tools/frontier_env_setup_6.3.sh

# Get node and task IDs
NODE_ID=${SLURM_NODEID:-0}
LOCAL_TASK_ID=${SLURM_LOCALID:-0}

# If BATCH_LIST_FILE is provided, use it to distribute batches across nodes
# Otherwise, use the single batch mode with INPUT_FILENAME and OUTPUT_FILENAME
if [ -n "$BATCH_LIST_FILE" ] && [ -f "$BATCH_LIST_FILE" ]; then
    # Multi-batch mode: read batch assignments from file
    # Format: one line per node: OUT_DIR:INPUT_FILENAME:OUTPUT_FILENAME
    # All tasks on the node need to read the batch assignment for srun
    BATCH_LINE=$(sed -n "$((NODE_ID + 1))p" "$BATCH_LIST_FILE")
    
    if [ -z "$BATCH_LINE" ]; then
        if [ "$LOCAL_TASK_ID" -eq 0 ]; then
            echo "No batch assignment for node $NODE_ID"
        fi
        exit 0
    fi
    
    IFS=':' read -r OUT_DIR INPUT_FILENAME OUTPUT_FILENAME <<< "$BATCH_LINE"
else
    # Single batch mode: use environment variables
    # This allows backward compatibility with existing submission scripts
    if [ -z "$INPUT_FILENAME" ] || [ -z "$OUTPUT_FILENAME" ]; then
        if [ "$LOCAL_TASK_ID" -eq 0 ]; then
            echo "Error: Either BATCH_LIST_FILE or INPUT_FILENAME/OUTPUT_FILENAME must be set"
            exit 1
        else
            exit 1
        fi
    fi
fi

# All tasks participate in srun, but only task 0 does setup and writes output
if [ "$LOCAL_TASK_ID" -eq 0 ]; then
    mkdir -p "$OUT_DIR"
    pushd "$OUT_DIR" || exit
    echo "Node $NODE_ID: Running input $INPUT_FILENAME and saving to $OUTPUT_FILENAME"
fi

# Use srun to properly allocate GPUs - all 9 tasks on this node participate
# --ntasks=9 uses all 9 tasks on this node (to utilize all 8 GPUs)
# --ntasks-per-node=9 ensures we use all tasks on the current node
# --exclusive ensures this task gets exclusive access to the node's resources
# All tasks call srun, and srun coordinates to run exess using all 9 tasks
if [ "$LOCAL_TASK_ID" -eq 0 ]; then
    srun --ntasks=9 --ntasks-per-node=9 --exclusive --export=ALL $EXESS_PATH/build/exess "$INPUT_FILENAME" 2>&1 | tee "$OUTPUT_FILENAME"
    popd || exit
else
    # Tasks 1-8 also call srun - srun will coordinate and use all 9 tasks
    srun --ntasks=9 --ntasks-per-node=9 --exclusive --export=ALL $EXESS_PATH/build/exess "$INPUT_FILENAME" > /dev/null 2>&1
fi
