#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/hardconcrete_sweep_2.yaml \
--output_dir experiment_outputs/hardconcrete_sweep

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--wandb_project raymondl/tinystories-1m-hardconcrete \
--n_eval_samples 10000 \
--filter_runs_by_name apply_relu_to_magnitude_false