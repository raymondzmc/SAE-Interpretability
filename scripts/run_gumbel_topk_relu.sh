#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-gumbel-topk.yaml \
--sweep_config configs/tinystories/sweep/gumbel_topk_relu_sweep.yaml \
--output_dir experiment_outputs/gumbel_topk_relu_sweep

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--wandb_project raymondl/tinystories-1m-gumbel-topk \
--filter_runs_by_name relu_gumbel_topk