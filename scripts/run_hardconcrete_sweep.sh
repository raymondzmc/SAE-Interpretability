#!/bin/bash
# Run Hard Concrete SAE hyperparameter sweep
python run_experiments.py \
--base_config configs/tinystories/tinystories-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/hardconcrete_sweep.yaml \
--output_dir experiment_outputs/hardconcrete_sweep \
--devices cuda:0 &