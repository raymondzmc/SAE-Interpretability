#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories/tinystories-lagrangian-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/lagrangian_hardconcrete/softplus_sweep_with_lb.yaml \
--output_dir experiment_outputs/softplus_lagrangian_hardconcrete_sweep_with_lb

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--wandb_project raymondl/tinystories-1m-lagrangian-hardconcrete \
--filter_runs_by_name softplus_lagrangian_hardconcrete_with_lb