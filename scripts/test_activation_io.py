#!/usr/bin/env python3
import argparse
import time
from typing import Tuple

import torch
import wandb

from settings import settings
from utils.io import save_activation_data_to_wandb, load_activation_data_from_wandb


def build_sample_activation_data(window_size: int = 64) -> Tuple[dict, list[list[str]]]:
    sae_pos = "blocks.0.mlp.hook_post"
    # Create small sample tensors
    n_examples = 3
    nonzero_activations = torch.randn(n_examples, window_size, dtype=torch.float16)
    data_indices = torch.arange(n_examples, dtype=torch.long)
    neuron_indices = torch.tensor([5, 10, 9], dtype=torch.long)

    accumulated_data = {
        sae_pos: {
            "nonzero_activations": nonzero_activations,
            "data_indices": data_indices,
            "neuron_indices": neuron_indices,
        }
    }

    # Build token ids
    all_token_ids: list[list[str]] = []
    for i in range(n_examples):
        seq = [f"tok{i}_{j}" for j in range(window_size)]
        all_token_ids.append(seq)

    return accumulated_data, all_token_ids


def main():
    parser = argparse.ArgumentParser(description="Test saving and loading activation_data to/from a Wandb run")
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                        help="Wandb project in format 'entity/project'")
    parser.add_argument("--run_id", type=str, default=None, help="Specific run ID to attach to")
    parser.add_argument("--filter_by_name", type=str, default=None, help="Filter runs by name to pick first match")
    parser.add_argument("--window_size", type=int, default=64, help="Sequence length for activations")

    args = parser.parse_args()

    # Login
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()

    # Resolve run
    run_id = args.run_id
    if run_id is None:
        runs = api.runs(args.wandb_project)
        if args.filter_by_name:
            runs = [r for r in runs if args.filter_by_name in r.name]
        if not runs:
            raise RuntimeError("No runs found for project/filter")
        run = runs[0]
        run_id = run.id
        print(f"Using run {run_id} ({run.name})")

    entity, project_name = args.wandb_project.split("/")
    run = wandb.init(project=project_name, entity=entity, id=run_id, resume="allow")

    # Build sample data
    accumulated_data, all_token_ids = build_sample_activation_data(window_size=args.window_size)

    # Save to run files
    save_activation_data_to_wandb(accumulated_data=accumulated_data, all_token_ids=all_token_ids)

    # Nudge sync
    run.log({"_activation_io_test_marker": time.time()})
    wandb.finish()

    # Validate by loading back
    loaded_data, loaded_tokens = load_activation_data_from_wandb(run_id=run_id, project=args.wandb_project)

    # Basic checks
    sae_pos = "blocks.0.mlp.hook_post"
    assert sae_pos in loaded_data, f"Missing key {sae_pos} in loaded activation data"

    orig = accumulated_data[sae_pos]
    back = loaded_data[sae_pos]

    ok_shapes = (
        orig["nonzero_activations"].shape == back["nonzero_activations"].shape and
        orig["data_indices"].shape == back["data_indices"].shape and
        orig["neuron_indices"].shape == back["neuron_indices"].shape
    )
    print("Shape check:", ok_shapes, orig["nonzero_activations"].shape)

    # Compare a few values
    diffs = (orig["nonzero_activations"].float() - back["nonzero_activations"].float()).abs().max().item()
    eq_data_idx = torch.equal(orig["data_indices"], back["data_indices"])
    eq_neuron_idx = torch.equal(orig["neuron_indices"], back["neuron_indices"])

    print("Max activation diff:", diffs)
    print("Data indices equal:", eq_data_idx)
    print("Neuron indices equal:", eq_neuron_idx)

    # Token IDs
    print("Token IDs loaded:", loaded_tokens is not None)
    if loaded_tokens is not None:
        print("Token length match:", len(loaded_tokens) == len(all_token_ids))
        print("First token seq match:", loaded_tokens[0] == all_token_ids[0])

    # Cleanup: delete activation_data/* from this run
    run_api = api.run(f"{args.wandb_project}/{run_id}")
    deleted = []
    for f in run_api.files():
        if f.name.startswith("activation_data/"):
            try:
                f.delete()
                deleted.append(f.name)
            except Exception:
                pass

    print("Deleted:", deleted)


if __name__ == "__main__":
    main() 