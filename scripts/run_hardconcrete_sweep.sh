#!/bin/bash
# Run Hard Concrete SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 4 is visible
tmux new-session -d -s "hardconcrete_sweep" 'CUDA_VISIBLE_DEVICES=3 python run_experiments.py --base_config configs/tinystories/tinystories-hardconcrete.yaml --sweep_config configs/tinystories/sweep/hardconcrete_sweep.yaml --output_dir experiment_outputs/hardconcrete_sweep'

echo "Started tmux session 'hardconcrete_sweep'"
echo "Attach with: tmux attach -t hardconcrete_sweep"
echo "Monitor with: tmux list-sessions"