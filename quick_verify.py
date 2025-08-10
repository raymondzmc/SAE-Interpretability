#!/usr/bin/env python3
"""
Quick script to verify metrics for a single run or a few runs.
This is a simplified version of verify_metrics.py for rapid testing.
"""

import argparse
import wandb
from settings import settings

def quick_check_run(run_id: str, project: str):
    """Quick check for a single run's files."""
    api = wandb.Api()
    
    print(f"Checking run: {run_id}")
    
    try:
        run = api.run(f"{project}/{run_id}")
        files = list(run.files())
        file_names = [f.name for f in files]
        
        # Check activation data
        activation_files = [f for f in file_names if f.startswith("activation_data/") and f.endswith(".pt")]
        if activation_files:
            print(f"  ✓ Activation data files found: {len(activation_files)} files")
            print(f"    - Files: {activation_files[:3]}{'...' if len(activation_files) > 3 else ''}")
        else:
            print(f"  ✗ No activation data files found")
        
        # Check metrics
        if "metrics.json" in file_names:
            print(f"  ✓ Metrics file exists")
        else:
            print(f"  ✗ Metrics file NOT found")
        
        # Check explanations
        if "explanations.json" in file_names:
            print(f"  ✓ Explanations file exists")
        else:
            print(f"  ✗ Explanations file NOT found")
            
        # Check explanation summary
        if "explanation_summary.json" in file_names:
            print(f"  ✓ Explanation summary exists")
        else:
            print(f"  ✗ Explanation summary NOT found")
            
    except wandb.errors.CommError:
        print(f"  ✗ Run NOT found")
    except Exception as e:
        print(f"  ✗ Error checking run: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quick verification of metrics for specific runs")
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project in format 'entity/project'")
    parser.add_argument("--run_ids", type=str, nargs="+", default=None,
                       help="Specific run IDs to check")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by name and check first few")
    parser.add_argument("--max_runs", type=int, default=3,
                       help="Maximum number of runs to check when filtering")
    
    args = parser.parse_args()
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    if args.run_ids:
        # Check specific run IDs
        for run_id in args.run_ids:
            quick_check_run(run_id, args.wandb_project)
            print()
    else:
        # Get runs from project
        runs = api.runs(args.wandb_project)
        
        if args.filter_runs_by_name:
            runs = [run for run in runs if args.filter_runs_by_name in run.name]
            print(f"Found {len(runs)} runs matching filter: {args.filter_runs_by_name}")
        
        # Check first few runs
        runs_to_check = runs[:args.max_runs]
        print(f"Checking first {len(runs_to_check)} runs...")
        
        for run in runs_to_check:
            print(f"\nRun: {run.name}")
            quick_check_run(run.id, args.wandb_project)

if __name__ == "__main__":
    main() 