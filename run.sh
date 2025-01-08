#!/bin/bash

# Define hyperparameter ranges
lambda_1_values=(0.0001 0.00001 0.000001 0.000001)
margin_values=(1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
beta_values=(0.1 0.2 0.3 0.4 0.5)
loss_modes=("mixed-margin")

# Output directory for logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Function to run the program with specific hyperparameters
run_experiment() {
    lambda_1=$1
    margin_value=$2
    beta_value=$3
    loss_mode=$4

    log_file="$LOG_DIR/lambda1_${lambda_1}_margin_${margin_value}_beta_${beta_value}_loss_${loss_mode}.log"

    echo "Starting experiment with lambda_1=$lambda_1, margin_value=$margin_value, beta_value=$beta_value, loss_mode=$loss_mode"

    python3 main.py \
        --lambda_1 $lambda_1 \
        --margin_value $margin_value \
        --beta $beta_value \
        --loss_mode $loss_mode > "$log_file" 2>&1

    echo "Experiment completed for lambda_1=$lambda_1, margin_value=$margin_value, beta_value=$beta_value, loss_mode=$loss_mode"
}

# Export the function for parallel to use
export -f run_experiment
export LOG_DIR

# Generate all combinations of hyperparameters
params=()
for lambda_1 in "${lambda_1_values[@]}"; do
    for margin_value in "${margin_values[@]}"; do
        for beta_value in "${beta_values[@]}"; do
            for loss_mode in "${loss_modes[@]}"; do
                params+=("$lambda_1 $margin_value $beta_value $loss_mode")
            done
        done
    done
done

# Automatically detect number of CPU cores and set parallelism
NUM_CORES=$(nproc)  # Detect total cores
PARALLEL_JOBS=$((NUM_CORES - 12))  # Leave 12 cores for system processes

# Run experiments in parallel
printf "%s\n" "${params[@]}" | xargs -n 4 -P $PARALLEL_JOBS bash -c 'run_experiment "$@"' _
