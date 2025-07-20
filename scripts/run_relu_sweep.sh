#!/bin/bash
# Run ReLU SAE hyperparameter sweep
python run_experiments.py \
--base_config configs/tinystories/tinystories-relu.yaml \
--sweep_config configs/tinystories/sweep/relu_sweep.yaml \
--output_dir experiment_outputs/relu_sweep \
--devices cuda:3 &