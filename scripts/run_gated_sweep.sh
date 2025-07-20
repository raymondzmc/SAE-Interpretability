#!/bin/bash
# Run Gated SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to ensure only GPU 1 is visible
tmux new-session -d -s "gated_sweep" \
"CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-gated.yaml \
--sweep_config configs/tinystories/sweep/gated_sweep.yaml \
--output_dir experiment_outputs/gated_sweep"

echo "Started tmux session 'gated_sweep'"
echo "Attach with: tmux attach -t gated_sweep"
echo "Monitor with: tmux list-sessions"