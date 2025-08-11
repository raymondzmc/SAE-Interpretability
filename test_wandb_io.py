#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime, timezone

import wandb

from settings import settings
from utils.io import save_metrics_to_wandb, load_metrics_from_wandb


def pick_run(api: wandb.Api, project: str, filter_by_name: str | None) -> tuple[str, str]:
    runs = api.runs(project)
    if filter_by_name:
        runs = [r for r in runs if filter_by_name in r.name]
    if not runs:
        raise RuntimeError("No runs found for the specified project/filter")
    run = runs[0]
    return run.id, run.name


def attach_to_run(project: str, run_id: str):
    entity, project_name = project.split("/")
    return wandb.init(project=project_name, entity=entity, id=run_id, resume="allow")


def list_run_files(project: str, run_id: str) -> list[str]:
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    return [f.name for f in run.files()]


def main():
    parser = argparse.ArgumentParser(description="Test saving and loading metrics.json to/from a Wandb run")
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                        help="Wandb project in format 'entity/project'")
    parser.add_argument("--run_id", type=str, default=None, help="Specific run ID to attach to")
    parser.add_argument("--filter_by_name", type=str, default=None,
                        help="Filter runs by name; first match is used if run_id not provided")
    parser.add_argument("--print_loaded", action="store_true", help="Print loaded metrics JSON")

    args = parser.parse_args()

    # Login
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()

    # Resolve run_id
    run_id = args.run_id
    run_name = None
    if run_id is None:
        run_id, run_name = pick_run(api, args.wandb_project, args.filter_by_name)

    print(f"Using run_id={run_id}{' (' + run_name + ')' if run_name else ''} in project={args.wandb_project}")

    # Attach to existing run
    run = attach_to_run(args.wandb_project, run_id)

    # Create a small test metrics payload
    timestamp = datetime.now(timezone.utc).isoformat()
    test_metrics = {
        "__test__": {
            "note": "test metrics entry created by test_wandb_io.py",
            "timestamp": timestamp,
            "random_value": time.time() % 1.0
        }
    }

    # Method A: via utils.io (temp dir + wandb.save)
    print("Saving metrics.json to run files via utils.io.save_metrics_to_wandb...")
    save_metrics_to_wandb(test_metrics)

    # Method B: write directly into run dir and save
    run_dir = wandb.run.dir
    direct_path = os.path.join(run_dir, "metrics.json")
    with open(direct_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Wrote metrics.json directly to run dir: {direct_path}")

    print("Calling wandb.save on run-dir metrics.json with policy=now...")
    wandb.save(direct_path, policy="now")

    # Log small scalar to force a sync heartbeat
    run.log({"_test_metrics_saved": time.time()})

    # Give uploader a moment
    time.sleep(2)

    # List files mid-run
    try:
        names_mid = list_run_files(args.wandb_project, run_id)
        print("Files currently on run (mid-run):", names_mid)
    except Exception as e:
        print("Warning: could not list files mid-run:", e)

    wandb.finish()

    # Wait for backend to index files
    time.sleep(3)

    # Load back
    print("Loading metrics.json back from run files via utils.io.load_metrics_from_wandb...")
    loaded = load_metrics_from_wandb(run_id=run_id, project=args.wandb_project)

    # List files after finish
    names = list_run_files(args.wandb_project, run_id)
    print("Files after finish:")
    for n in names:
        print(" -", n)

    if loaded is None:
        print("✗ Failed to load metrics.json back from run files.")
        return

    has_test_key = "__test__" in loaded
    print(f"✓ Loaded metrics.json ({'contains' if has_test_key else 'missing'} __test__ key)")

    if args.print_loaded:
        print(json.dumps(loaded, indent=2))


if __name__ == "__main__":
    main() 