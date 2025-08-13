#!/usr/bin/env python3
"""
Quick status check for evaluation completion across different SAE types.
"""

import wandb
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from settings import settings


def check_evaluation_status(project: str = "raymondl/tinystories-1m"):
    """Check evaluation status for all runs grouped by SAE type."""
    
    print("=" * 60)
    print("EVALUATION STATUS CHECK")
    print("=" * 60)
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs
    runs = list(api.runs(project))
    
    # Group runs by SAE type
    sae_types = {
        'relu': [],
        'gated': [],
        'hard_concrete': [],
        'hardconcrete': [],  # Alternative naming
        'topk': [],
        'other': []
    }
    
    for run in runs:
        name_lower = run.name.lower()
        if 'relu' in name_lower:
            sae_types['relu'].append(run)
        elif 'gated' in name_lower:
            sae_types['gated'].append(run)
        elif 'hard_concrete' in name_lower or 'hardconcrete' in name_lower:
            sae_types['hard_concrete'].append(run)
        elif 'topk' in name_lower or 'top_k' in name_lower:
            sae_types['topk'].append(run)
        else:
            sae_types['other'].append(run)
    
    # Check each SAE type
    total_evaluated = 0
    total_runs = len(runs)
    
    for sae_type, type_runs in sae_types.items():
        if not type_runs:
            continue
            
        # Check which runs have metrics
        with_metrics = 0
        missing = []
        
        for run in type_runs:
            artifacts = list(run.logged_artifacts())
            has_metrics = any(
                a.type == "metrics" and "evaluation_metrics" in a.name 
                for a in artifacts
            )
            
            if has_metrics:
                with_metrics += 1
                total_evaluated += 1
            else:
                missing.append(run.name)
        
        # Print status for this SAE type
        if type_runs:
            completion_pct = 100 * with_metrics / len(type_runs)
            status_icon = "✅" if with_metrics == len(type_runs) else "⚠️"
            
            print(f"\n{sae_type.upper():15} {status_icon} {with_metrics}/{len(type_runs)} runs evaluated ({completion_pct:.0f}%)")
            
            if missing and len(missing) <= 3:
                for name in missing:
                    print(f"  ❌ Missing: {name}")
            elif missing:
                print(f"  ❌ Missing: {missing[0]}, {missing[1]}, ... ({len(missing)} total)")
    
    # Overall summary
    print("\n" + "=" * 60)
    overall_pct = 100 * total_evaluated / total_runs if total_runs > 0 else 0
    
    if total_evaluated == total_runs:
        print(f"✅ ALL EVALUATIONS COMPLETE: {total_evaluated}/{total_runs} runs")
    else:
        print(f"📊 OVERALL PROGRESS: {total_evaluated}/{total_runs} runs ({overall_pct:.0f}%)")
        print(f"   {total_runs - total_evaluated} runs still need evaluation")
    
    print("=" * 60)
    
    return total_evaluated == total_runs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check evaluation status")
    parser.add_argument(
        "--project",
        type=str,
        default="raymondl/tinystories-1m",
        help="Wandb project"
    )
    
    args = parser.parse_args()
    
    all_complete = check_evaluation_status(args.project)
    sys.exit(0 if all_complete else 1) 