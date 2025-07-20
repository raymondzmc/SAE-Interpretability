#!/bin/bash
# Run Hard Concrete SAE (no learned gates) hyperparameter sweep
python run_experiments.py --base_config configs/tinystories/tinystories-hardconcrete-no-learned-gates.yaml --sweep_config configs/tinystories/sweep/hardconcrete_no_gates_sweep.yaml 