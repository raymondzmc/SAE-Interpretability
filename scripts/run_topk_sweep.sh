
#!/bin/bash
conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
--base_config configs/tinystories/tinystories-topk.yaml \
--sweep_config configs/tinystories/sweep/topk_sweep.yaml \
--output_dir experiment_outputs/topk_sweep

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--wandb_project raymondl/tinystories-1m \
--filter_runs_by_name topk