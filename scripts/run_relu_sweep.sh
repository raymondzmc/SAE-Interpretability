#!/bin/bash
# Run ReLU SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 0 is visible

# Check if config files exist
if [ ! -f "configs/tinystories/tinystories-relu.yaml" ]; then
    echo "Error: configs/tinystories/tinystories-relu.yaml not found"
    exit 1
fi

if [ ! -f "configs/tinystories/sweep/relu_sweep.yaml" ]; then
    echo "Error: configs/tinystories/sweep/relu_sweep.yaml not found"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "relu_sweep" 2>/dev/null

# Create tmux session
tmux new-session -d -s "relu_sweep"

# Send command to the session
tmux send-keys -t "relu_sweep" "CUDA_VISIBLE_DEVICES=0 python run_experiments.py --base_config configs/tinystories/tinystories-relu.yaml --sweep_config configs/tinystories/sweep/relu_sweep.yaml --output_dir experiment_outputs/tinystories-relu-sweep" Enter

echo "Started tmux session 'relu_sweep'"
echo "Attach with: tmux attach -t relu_sweep"
echo "Monitor with: tmux list-sessions"

# Show session status
sleep 1
if tmux has-session -t "relu_sweep" 2>/dev/null; then
    echo "✅ Session is running"
else
    echo "❌ Session exited - check for errors in the command"
fi