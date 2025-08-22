#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories/tinystories-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/hardconcrete_sweep.yaml \
--output_dir experiment_outputs/hardconcrete_sweep

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--wandb_project raymondl/tinystories-1m-hardconcrete \
--n_eval_samples 10000 \
--filter_runs_by_name magnitude_activation_relu