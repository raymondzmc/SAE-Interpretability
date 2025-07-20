#!/bin/bash
# Run Gated Hard Concrete SAE hyperparameter sweep
python run_experiments.py \
--base_config configs/tinystories/tinystories-gated-hardconcrete.yaml \
--sweep_config configs/tinystories/sweep/gated_hardconcrete_sweep.yaml \
--output_dir experiment_outputs/gated_hardconcrete_sweep \
--devices cuda:2