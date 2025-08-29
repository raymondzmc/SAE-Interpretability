#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/hardconcrete_sweep.yaml \
--output_dir experiment_outputs/hardconcrete_sweep

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--wandb_project raymondl/tinystories-1m-hardconcrete \
--filter_runs_by_name hard_concrete_sparsity_coeff_5e-03_initial_beta_1.0