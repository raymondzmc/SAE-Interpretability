
CUDA_VISIBLE_DEVICES=3 python run_experiments.py \
--base_config configs/tinystories-test/tinystories-gated.yaml \
--sweep_config configs/tinystories-test/sweep/gated_sweep_3.yaml \
--output_dir experiment_outputs/gated_sweep_3

CUDA_VISIBLE_DEVICES=3 python evaluation.py \
--wandb_project raymondl/tinystories-1m-test \
--filter_runs_by_name decoder_transpose