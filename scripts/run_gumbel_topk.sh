#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories/tinystories-gumbel-topk.yaml \
--sweep_config configs/tinystories/sweep/gumbel_topk_sweep.yaml \
--output_dir experiment_outputs/gumbel_topk_sweep

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--wandb_project raymondl/tinystories-1m-gumbel-topk \
--filter_runs_by_name standard_gumbel_topk