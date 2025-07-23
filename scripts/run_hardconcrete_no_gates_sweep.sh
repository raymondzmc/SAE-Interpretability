#!/bin/bash
# Run Hard Concrete SAE (no learned gates) hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 5 is visible
tmux new-session -d -s "hardconcrete_no_gates_sweep" 'CUDA_VISIBLE_DEVICES=3 python run_experiments.py --base_config configs/tinystories/tinystories-hardconcrete-no-learned-gates.yaml --sweep_config configs/tinystories/sweep/hardconcrete_no_gates_sweep.yaml --output_dir experiment_outputs/hardconcrete_no_gates_sweep'

echo "Started tmux session 'hardconcrete_no_gates_sweep'"
echo "Attach with: tmux attach -t hardconcrete_no_gates_sweep"
echo "Monitor with: tmux list-sessions"