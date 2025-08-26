#!/bin/bash
conda activate sae

# CUDA_VISIBLE_DEVICES=2 python run_experiments.py \
# --base_config configs/tinystories/tinystories-lagrangian-hardconcrete.yaml \
# --sweep_config configs/tinystories/sweep/lagrangian_hardconcrete_sweep.yaml \
# --output_dir experiment_outputs/lagrangian_hardconcrete_sweep

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--wandb_project raymondl/tinystories-1m-lagrangian-hardconcrete \
--filter_runs_by_name standard_lagrangian_hardconcrete