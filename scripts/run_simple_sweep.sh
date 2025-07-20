#!/bin/bash

# Simple bash script for running SAE hyperparameter sweeps
# Usage: ./scripts/run_simple_sweep.sh [SAE_TYPE] [MAX_PARALLEL]

set -e  # Exit on any error

# Configuration
SAE_TYPE=${1:-"relu"}
MAX_PARALLEL=${2:-1}
BASE_CONFIG="example_configs/tinystories-relu.yaml"
WANDB_PROJECT="tinystories-sweeps"

# Hyperparameter grids
declare -a LR_VALUES=(1e-4 3e-4 1e-3)
declare -a DICT_RATIOS=(8.0 16.0 32.0 64.0)
declare -a SPARSITY_COEFFS=(0.01 0.03 0.1 0.3)

# SAE-specific parameters
declare -a AUX_COEFFS=(0.01 0.03125 0.1)
declare -a INITIAL_BETAS=(0.1 0.5 1.0)
declare -a GATE_TYPES=(true false)

# Create temporary config directory
TEMP_DIR="temp_sweep_configs"
mkdir -p "$TEMP_DIR"

# Function to create a config file
create_config() {
    local config_file="$1"
    local lr="$2"
    local dict_ratio="$3" 
    local sparsity_coeff="$4"
    local aux_coeff="$5"
    local initial_beta="$6"
    local input_gates="$7"
    
    # Start with base config
    cp "$BASE_CONFIG" "$config_file"
    
    # Update parameters using sed (simple approach)
    # Update SAE type
    sed -i "s/sae_type: .*/sae_type: \"$SAE_TYPE\"/" "$config_file"
    sed -i "s/name: .*/name: \"${SAE_TYPE}_sae\"/" "$config_file"
    
    # Update learning rate
    sed -i "s/lr: .*/lr: $lr/" "$config_file"
    
    # Update dict ratio
    sed -i "s/dict_size_to_input_ratio: .*/dict_size_to_input_ratio: $dict_ratio/" "$config_file"
    
    # Update sparsity coefficient
    sed -i "s/sparsity_coeff: .*/sparsity_coeff: $sparsity_coeff/" "$config_file"
    
    # Update wandb project
    sed -i "s/wandb_project: .*/wandb_project: $WANDB_PROJECT/" "$config_file"
    
    # Create run name
    local run_name="${SAE_TYPE}_lr_${lr}_dict_${dict_ratio}_sparse_${sparsity_coeff}"
    
    # SAE-specific parameters
    case "$SAE_TYPE" in
        "gated"|"gated_hard_concrete")
            # Add aux_coeff if not present
            if ! grep -q "aux_coeff:" "$config_file"; then
                echo "  aux_coeff: $aux_coeff" >> "$config_file"
            else
                sed -i "s/aux_coeff: .*/aux_coeff: $aux_coeff/" "$config_file"
            fi
            run_name="${run_name}_aux_${aux_coeff}"
            ;;
    esac
    
    case "$SAE_TYPE" in
        "hard_concrete"|"gated_hard_concrete")
            # Add initial_beta if not present
            if ! grep -q "initial_beta:" "$config_file"; then
                echo "  initial_beta: $initial_beta" >> "$config_file"
            else
                sed -i "s/initial_beta: .*/initial_beta: $initial_beta/" "$config_file"
            fi
            run_name="${run_name}_beta_${initial_beta}"
            
            if [ "$SAE_TYPE" = "hard_concrete" ]; then
                # Add input_dependent_gates if not present
                if ! grep -q "input_dependent_gates:" "$config_file"; then
                    echo "  input_dependent_gates: $input_gates" >> "$config_file"
                else
                    sed -i "s/input_dependent_gates: .*/input_dependent_gates: $input_gates/" "$config_file"
                fi
                run_name="${run_name}_gates_${input_gates}"
            fi
            ;;
    esac
    
    # Update run name
    sed -i "s/wandb_run_name: .*/wandb_run_name: $run_name/" "$config_file"
}

# Function to run a single experiment
run_experiment() {
    local config_file="$1"
    local experiment_name=$(basename "$config_file" .yaml)
    
    echo "Starting experiment: $experiment_name"
    
    if python run.py "$config_file"; then
        echo "✅ Completed: $experiment_name"
        return 0
    else
        echo "❌ Failed: $experiment_name"
        return 1
    fi
}

# Function to run experiments in parallel
run_parallel() {
    local max_jobs="$1"
    shift
    local configs=("$@")
    
    local running_jobs=0
    local completed=0
    local failed=0
    
    for config in "${configs[@]}"; do
        # Wait if we've reached max parallel jobs
        while [ $running_jobs -ge $max_jobs ]; do
            wait -n  # Wait for any background job to complete
            local exit_code=$?
            running_jobs=$((running_jobs - 1))
            
            if [ $exit_code -eq 0 ]; then
                completed=$((completed + 1))
            else
                failed=$((failed + 1))
            fi
        done
        
        # Start new job in background
        run_experiment "$config" &
        running_jobs=$((running_jobs + 1))
    done
    
    # Wait for remaining jobs
    while [ $running_jobs -gt 0 ]; do
        wait -n
        local exit_code=$?
        running_jobs=$((running_jobs - 1))
        
        if [ $exit_code -eq 0 ]; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo "Summary: $completed completed, $failed failed"
}

# Generate configs and run experiments
main() {
    echo "Running sweep for $SAE_TYPE SAE with max $MAX_PARALLEL parallel jobs"
    
    local configs=()
    local config_count=0
    
    # Generate all parameter combinations
    for lr in "${LR_VALUES[@]}"; do
        for dict_ratio in "${DICT_RATIOS[@]}"; do
            for sparsity_coeff in "${SPARSITY_COEFFS[@]}"; do
                case "$SAE_TYPE" in
                    "relu")
                        config_file="$TEMP_DIR/relu_${config_count}.yaml"
                        create_config "$config_file" "$lr" "$dict_ratio" "$sparsity_coeff" "" "" ""
                        configs+=("$config_file")
                        config_count=$((config_count + 1))
                        ;;
                    "hard_concrete")
                        for initial_beta in "${INITIAL_BETAS[@]}"; do
                            for input_gates in "${GATE_TYPES[@]}"; do
                                config_file="$TEMP_DIR/hard_concrete_${config_count}.yaml"
                                create_config "$config_file" "$lr" "$dict_ratio" "$sparsity_coeff" "" "$initial_beta" "$input_gates"
                                configs+=("$config_file")
                                config_count=$((config_count + 1))
                            done
                        done
                        ;;
                    "gated")
                        for aux_coeff in "${AUX_COEFFS[@]}"; do
                            config_file="$TEMP_DIR/gated_${config_count}.yaml"
                            create_config "$config_file" "$lr" "$dict_ratio" "$sparsity_coeff" "$aux_coeff" "" ""
                            configs+=("$config_file")
                            config_count=$((config_count + 1))
                        done
                        ;;
                    "gated_hard_concrete")
                        for aux_coeff in "${AUX_COEFFS[@]}"; do
                            for initial_beta in "${INITIAL_BETAS[@]}"; do
                                config_file="$TEMP_DIR/gated_hard_concrete_${config_count}.yaml"
                                create_config "$config_file" "$lr" "$dict_ratio" "$sparsity_coeff" "$aux_coeff" "$initial_beta" ""
                                configs+=("$config_file")
                                config_count=$((config_count + 1))
                            done
                        done
                        ;;
                    *)
                        echo "Unknown SAE type: $SAE_TYPE"
                        exit 1
                        ;;
                esac
            done
        done
    done
    
    echo "Generated ${#configs[@]} configurations"
    
    # Run experiments
    if [ $MAX_PARALLEL -eq 1 ]; then
        echo "Running experiments sequentially..."
        local completed=0
        local failed=0
        
        for config in "${configs[@]}"; do
            if run_experiment "$config"; then
                completed=$((completed + 1))
            else
                failed=$((failed + 1))
            fi
        done
        
        echo "Summary: $completed completed, $failed failed"
    else
        echo "Running experiments with max $MAX_PARALLEL parallel jobs..."
        run_parallel $MAX_PARALLEL "${configs[@]}"
    fi
    
    # Cleanup
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    
    echo "Sweep completed!"
}

# Check if base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config file not found: $BASE_CONFIG"
    exit 1
fi

# Run main function
main 