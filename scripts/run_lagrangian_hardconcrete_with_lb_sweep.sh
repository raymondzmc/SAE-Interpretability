#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-lagrangian-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/lagrangian_hardconcrete/default_sweep_with_lb.yaml \
--output_dir experiment_outputs/lagrangian_hardconcrete_sweep_with_lb

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--wandb_project raymondl/tinystories-1m-lagrangian-hardconcrete \
--filter_runs_by_name standard_lagrangian_hardconcrete_with_lb