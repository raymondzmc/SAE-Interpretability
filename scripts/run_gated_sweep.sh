#!/bin/bash
# Run Gated SAE hyperparameter sweep
python run_experiments.py --base_config configs/tinystories/tinystories-gated.yaml --sweep_config configs/tinystories/sweep/gated_sweep.yaml 