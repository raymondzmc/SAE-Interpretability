#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories/tinystories-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/hardconcrete_sweep_1.yaml \
--output_dir experiment_outputs/hardconcrete_sweep