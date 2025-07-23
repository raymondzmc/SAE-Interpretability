#!/bin/bash
# Run Gated Hard Concrete SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 0 is visible
tmux new-session -d -s "gated_hardconcrete_sweep" 'CUDA_VISIBLE_DEVICES=2 python run_experiments.py --base_config configs/tinystories/tinystories-gated-hardconcrete.yaml --sweep_config configs/tinystories/sweep/gated_hardconcrete_sweep.yaml --output_dir experiment_outputs/gated_hardconcrete_sweep'

echo "Started tmux session 'gated_hardconcrete_sweep'"
echo "Attach with: tmux attach -t gated_hardconcrete_sweep"
echo "Monitor with: tmux list-sessions"