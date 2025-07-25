#!/bin/bash
# Run ReLU SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 0 is visible

# Get the current Python interpreter path
PYTHON_PATH=$(which python)

# Kill existing session if it exists
tmux kill-session -t "relu_sweep" 2>/dev/null

# Create tmux session with dynamically determined python path
tmux new-session -d -s "relu_sweep" "CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH run_experiments.py --base_config configs/tinystories/tinystories-relu.yaml --sweep_config configs/tinystories/sweep/relu_sweep.yaml --output_dir experiment_outputs/tinystories-relu-sweep"

echo "Started tmux session 'relu_sweep'"
echo "Attach with: tmux attach -t relu_sweep"
echo "Monitor with: tmux list-sessions"