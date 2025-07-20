#!/bin/bash
# Run ReLU SAE hyperparameter sweep using CUDA_VISIBLE_DEVICES to hide cuda:0
CUDA_VISIBLE_DEVICES=1 python run_experiments.py \
--base_config configs/tinystories/tinystories-relu.yaml \
--sweep_config configs/tinystories/sweep/relu_sweep.yaml \
--output_dir experiment_outputs/tinystories-relu-sweep