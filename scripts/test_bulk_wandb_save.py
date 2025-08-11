#!/usr/bin/env python3
import argparse
import time
from datetime import datetime, timezone

import wandb

from settings import settings
from utils.io import save_metrics_to_wandb, save_explanations_to_wandb


def save_and_cleanup_for_run(project: str, run_id: str) -> dict:
    entity, project_name = project.split("/")

    # Attach to run
    run = wandb.init(project=project_name, entity=entity, id=run_id, resume="allow")

    # Create test payloads
    ts = datetime.now(timezone.utc).isoformat()
    metrics = {"__bulk_test__": {"ts": ts}}
    explanations = {
        "test.layer_neuron_0": {
            "text": "bulk test explanation",
            "score": 0.0,
            "sae_position": "test.layer",
            "neuron_index": 0,
            "num_examples": 1,
        }
    }

    # Save files
    save_metrics_to_wandb(metrics)
    save_explanations_to_wandb(explanations)

    # Nudge sync
    run.log({"_bulk_test_marker": time.time()})
    wandb.finish()

    # Verify and then delete
    api = wandb.Api()
    run_api = api.run(f"{project}/{run_id}")

    file_names = [f.name for f in run_api.files()]
    created = [name for name in file_names if name in ("metrics.json", "explanations.json", "explanation_summary.json")]

    # Attempt deletion of created files
    deleted = []
    errors = []
    for f in run_api.files():
        if f.name in ("metrics.json", "explanations.json", "explanation_summary.json"):
            try:
                f.delete()
                deleted.append(f.name)
            except Exception as e:
                errors.append((f.name, str(e)))

    # Re-list
    file_names_after = [f.name for f in api.run(f"{project}/{run_id}").files()]

    return {
        "run_id": run_id,
        "created": created,
        "deleted": deleted,
        "remaining": [n for n in ("metrics.json", "explanations.json", "explanation_summary.json") if n in file_names_after],
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Bulk test: save and then delete test files across multiple runs")
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                        help="Wandb project in format 'entity/project'")
    parser.add_argument("--max_runs", type=int, default=3, help="Number of runs to test")
    parser.add_argument("--filter_by_name", type=str, default=None, help="Filter runs by name")

    args = parser.parse_args()

    # Login
    wandb.login(key=settings.wandb_api_key)

    api = wandb.Api()
    runs = api.runs(args.wandb_project)
    if args.filter_by_name:
        runs = [r for r in runs if args.filter_by_name in r.name]

    runs = runs[: args.max_runs]

    results = []
    for r in runs:
        print(f"Processing run: {r.id} ({r.name})")
        try:
            res = save_and_cleanup_for_run(args.wandb_project, r.id)
        except Exception as e:
            res = {"run_id": r.id, "error": str(e), "created": [], "deleted": [], "remaining": [], "errors": []}
        results.append(res)
        print(res)
        time.sleep(1)

    print("\nSummary:")
    for res in results:
        print(res)


if __name__ == "__main__":
    main() 