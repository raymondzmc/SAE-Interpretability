#!/usr/bin/env python3
"""
Debug script to see exactly what files exist in a Wandb run.
"""

import argparse
import wandb
from settings import settings

def debug_run_files(run_id: str, project: str = "raymondl/tinystories-1m"):
    """List all files in a Wandb run."""
    
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    try:
        run = api.run(f"{project}/{run_id}")
        files = list(run.files())
        
        print(f"Run: {run.name} ({run_id})")
        print(f"URL: {run.url}")
        print(f"Total files: {len(files)}")
        print("\nAll files in run:")
        
        for i, file in enumerate(files):
            print(f"  {i+1}. {file.name}")
            print(f"     Size: {file.size} bytes")
            print(f"     URL: {file.url}")
            print()
            
        if not files:
            print("  No files found in this run")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Debug files in a Wandb run")
    parser.add_argument("run_id", help="Wandb run ID to check")
    parser.add_argument("--project", default="raymondl/tinystories-1m", help="Wandb project")
    
    args = parser.parse_args()
    debug_run_files(args.run_id, args.project)

if __name__ == "__main__":
    main() 