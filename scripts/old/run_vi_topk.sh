#!/bin/bash
conda activate sae

# Run VI TopK SAE training
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories-test/tinystories-vi-topk.yaml \
--sweep_config configs/tinystories-test/sweep/vi-topk_sweep.yaml \
--output_dir experiment_outputs/vi-topk_sweep

CUDA_VISIBLE_DEVICES=3 python evaluation.py \
--wandb_project raymondl/tinystories-1m-test \
--filter_runs_by_name vi-topk 