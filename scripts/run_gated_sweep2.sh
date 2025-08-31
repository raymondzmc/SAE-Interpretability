
CUDA_VISIBLE_DEVICES=2 python run_experiments.py \
--base_config configs/tinystories-test/tinystories-gated.yaml \
--sweep_config configs/tinystories-test/sweep/gated_sweep2.yaml \
--output_dir experiment_outputs/gated_sweep2

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
--wandb_project raymondl/tinystories-1m-test \
--filter_runs_by_name separate