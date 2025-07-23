#!/bin/bash
# Run Hard Concrete SAE (no learned gates) hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 5 is visible

# Get the current Python interpreter path
PYTHON_PATH=$(which python)

tmux new-session -d -s "hardconcrete_no_gates_sweep" "CUDA_VISIBLE_DEVICES=3 $PYTHON_PATH run_experiments.py --base_config configs/tinystories/tinystories-hardconcrete-no-learned-gates.yaml --sweep_config configs/tinystories/sweep/hardconcrete_no_gates_sweep.yaml --output_dir experiment_outputs/hardconcrete_no_gates_sweep"

echo "Started tmux session 'hardconcrete_no_gates_sweep'"
echo "Attach with: tmux attach -t hardconcrete_no_gates_sweep"
echo "Monitor with: tmux list-sessions"