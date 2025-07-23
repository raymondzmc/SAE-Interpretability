#!/bin/bash
# Run ReLU SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 0 is visible

# Kill existing session if it exists
tmux kill-session -t "relu_sweep" 2>/dev/null

# Create tmux session with single-line command (no backslashes)
tmux new-session -d -s "relu_sweep" 'CUDA_VISIBLE_DEVICES=0 python run_experiments.py --base_config configs/tinystories/tinystories-relu.yaml --sweep_config configs/tinystories/sweep/relu_sweep.yaml --output_dir experiment_outputs/tinystories-relu-sweep'

echo "Started tmux session 'relu_sweep'"
echo "Attach with: tmux attach -t relu_sweep"
echo "Monitor with: tmux list-sessions"