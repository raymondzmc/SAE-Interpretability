#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories/tinystories-lagrangian-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/hardconcrete_lagrangian-sweep.yaml \
--output_dir experiment_outputs/hardconcrete_lagrangian_sweep

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--wandb_project raymondl/tinystories-1m-lagrangian-hardconcrete \
--filter_runs_by_name lagrangean_hard_concrete_sparsity_coeff_5e-03_initial_beta_3.0_final_beta_0.3