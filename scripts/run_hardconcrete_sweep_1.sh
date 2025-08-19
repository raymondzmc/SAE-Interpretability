#!/bin/bash
# Run Hard Concrete SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 4 is visible

# Get the current Python interpreter path
PYTHON_PATH=$(which python)

tmux new-session -d -s "hardconcrete_sweep" "CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH run_experiments.py --base_config configs/tinystories/tinystories-hardconcrete.yaml --sweep_config configs/tinystories/sweep/hardconcrete_sweep_1.yaml --output_dir experiment_outputs/hardconcrete_sweep"

echo "Started tmux session 'hardconcrete_sweep'"
echo "Attach with: tmux attach -t hardconcrete_sweep"
echo "Monitor with: tmux list-sessions"