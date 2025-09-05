
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=5

python run_experiments.py \
--base_config configs/tinystories-test/tinystories-hc_topk.yaml \
--sweep_config configs/tinystories-test/sweep/hc_topk_sweep5.yaml \
--output_dir experiment_outputs/hc_topk_sweep5

python evaluation.py \
--wandb_project raymondl/tinystories-1m-test \
--filter_runs_by_name score_method_magnitude_detached_straight_through_false