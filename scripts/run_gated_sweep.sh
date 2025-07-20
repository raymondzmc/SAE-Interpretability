#!/bin/bash
# Run Gated SAE hyperparameter sweep
CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-gated.yaml \
--sweep_config configs/tinystories/sweep/gated_sweep.yaml \
--output_dir experiment_outputs/gated_sweep \
--devices cuda:0 &