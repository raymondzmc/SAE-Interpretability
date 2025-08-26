#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=2 python run_experiments.py \
--base_config configs/tinystories/tinystories-lagrangian-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/lagrangian_hardconcrete/relu_sweep.yaml \
--output_dir experiment_outputs/relu_lagrangian_hardconcrete_sweep

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
--wandb_project raymondl/tinystories-1m-lagrangian-hardconcrete \
--filter_runs_by_name relu_lagrangian_hardconcrete