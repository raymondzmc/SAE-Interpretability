#!/usr/bin/env python3
"""
Script to verify that all runs have metrics.json saved as artifacts and can be loaded properly.
This should be run after evaluation.py has been executed on all runs.
"""

import wandb
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from settings import settings
from utils.io import load_metrics_from_wandb


@dataclass
class RunMetricsStatus:
    """Status of metrics for a single run"""
    run_id: str
    run_name: str
    has_artifact: bool
    artifact_name: Optional[str] = None
    artifact_version: Optional[int] = None
    can_load: bool = False
    num_sae_positions: Optional[int] = None
    metrics_keys: Optional[List[str]] = None
    error: Optional[str] = None


def check_all_runs_metrics(
    project: str = "raymondl/tinystories-1m",
    filter_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check all runs in a project to verify they have metrics artifacts.
    
    Args:
        project: Wandb project in format 'entity/project'
        filter_name: Optional filter for run names (e.g., 'relu', 'gated')
    
    Returns:
        Dictionary with summary statistics and detailed results
    """
    
    print(f"=== VERIFYING METRICS FOR ALL RUNS IN {project} ===\n")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs
    runs = api.runs(project)
    
    # Apply filter if specified
    if filter_name:
        original_count = len(runs)
        runs = [r for r in runs if filter_name in r.name]
        print(f"Filtering runs by '{filter_name}': {len(runs)}/{original_count} runs\n")
    else:
        print(f"Found {len(runs)} total runs\n")
    
    # Track results
    results: List[RunMetricsStatus] = []
    
    # Check each run
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] Checking run: {run.name} ({run.id})")
        
        status = RunMetricsStatus(
            run_id=run.id,
            run_name=run.name,
            has_artifact=False
        )
        
        try:
            # Check for metrics artifacts
            artifacts = list(run.logged_artifacts())
            metrics_artifacts = [
                a for a in artifacts 
                if a.type == "metrics" and "evaluation_metrics" in a.name
            ]
            
            if metrics_artifacts:
                # Found metrics artifact(s)
                latest_artifact = max(metrics_artifacts, key=lambda x: x.version)
                status.has_artifact = True
                status.artifact_name = latest_artifact.name
                status.artifact_version = latest_artifact.version
                
                print(f"  ✓ Found metrics artifact: {latest_artifact.name} (v{latest_artifact.version})")
                
                # Try to load the metrics
                try:
                    loaded_metrics = load_metrics_from_wandb(run.id, project)
                    
                    if loaded_metrics:
                        status.can_load = True
                        status.num_sae_positions = len(loaded_metrics)
                        
                        # Get a sample of metrics keys from first SAE position
                        if loaded_metrics:
                            first_pos = next(iter(loaded_metrics))
                            status.metrics_keys = list(loaded_metrics[first_pos].keys())
                        
                        print(f"  ✓ Successfully loaded metrics for {status.num_sae_positions} SAE positions")
                        
                        # Verify expected fields are present
                        expected_fields = [
                            "alive_dict_components",
                            "alive_dict_components_proportion", 
                            "sparsity_l0",
                            "mse",
                            "explained_variance"
                        ]
                        
                        missing_fields = []
                        for pos, metrics in loaded_metrics.items():
                            missing = [f for f in expected_fields if f not in metrics]
                            if missing:
                                missing_fields.extend([(pos, f) for f in missing])
                        
                        if missing_fields:
                            print(f"  ⚠️  Missing fields: {missing_fields[:5]}...")  # Show first 5
                            status.error = f"Missing fields: {len(missing_fields)} total"
                    else:
                        status.can_load = False
                        status.error = "load_metrics_from_wandb returned None"
                        print(f"  ✗ Failed to load metrics: returned None")
                        
                except Exception as e:
                    status.can_load = False
                    status.error = str(e)
                    print(f"  ✗ Error loading metrics: {e}")
                    
            else:
                print(f"  ✗ No metrics artifact found")
                status.error = "No metrics artifact found"
                
        except Exception as e:
            status.error = f"Error checking artifacts: {e}"
            print(f"  ✗ Error checking run: {e}")
        
        results.append(status)
        print()  # Empty line between runs
    
    # Generate summary statistics
    total_runs = len(results)
    runs_with_artifacts = sum(1 for r in results if r.has_artifact)
    runs_can_load = sum(1 for r in results if r.can_load)
    runs_missing = [r for r in results if not r.has_artifact]
    runs_load_failed = [r for r in results if r.has_artifact and not r.can_load]
    
    # Print summary
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total runs checked: {total_runs}")
    print(f"Runs with metrics artifacts: {runs_with_artifacts}/{total_runs} ({100*runs_with_artifacts/total_runs:.1f}%)")
    print(f"Runs successfully loaded: {runs_can_load}/{total_runs} ({100*runs_can_load/total_runs:.1f}%)")
    
    if runs_missing:
        print(f"\n⚠️  {len(runs_missing)} runs MISSING metrics artifacts:")
        for r in runs_missing[:10]:  # Show first 10
            print(f"  - {r.run_name} ({r.run_id})")
        if len(runs_missing) > 10:
            print(f"  ... and {len(runs_missing) - 10} more")
    
    if runs_load_failed:
        print(f"\n⚠️  {len(runs_load_failed)} runs FAILED to load (but have artifacts):")
        for r in runs_load_failed[:10]:  # Show first 10
            print(f"  - {r.run_name} ({r.run_id}): {r.error}")
        if len(runs_load_failed) > 10:
            print(f"  ... and {len(runs_load_failed) - 10} more")
    
    # Check consistency of metrics structure
    if runs_can_load > 0:
        print("\n" + "=" * 60)
        print("METRICS STRUCTURE ANALYSIS")
        print("=" * 60)
        
        # Get all unique SAE positions and metrics keys
        all_sae_positions = set()
        all_metrics_keys = set()
        
        for r in results:
            if r.can_load and r.metrics_keys:
                all_metrics_keys.update(r.metrics_keys)
                
        print(f"Common metrics fields found across runs:")
        for key in sorted(all_metrics_keys):
            print(f"  - {key}")
        
        # Check SAE position counts
        sae_position_counts = {}
        for r in results:
            if r.can_load and r.num_sae_positions:
                count = r.num_sae_positions
                if count not in sae_position_counts:
                    sae_position_counts[count] = 0
                sae_position_counts[count] += 1
        
        print(f"\nSAE position counts:")
        for count, num_runs in sorted(sae_position_counts.items()):
            print(f"  - {count} positions: {num_runs} runs")
    
    # Generate detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "project": project,
        "filter": filter_name,
        "summary": {
            "total_runs": total_runs,
            "runs_with_artifacts": runs_with_artifacts,
            "runs_can_load": runs_can_load,
            "success_rate": runs_can_load / total_runs if total_runs > 0 else 0
        },
        "missing_runs": [
            {"run_id": r.run_id, "run_name": r.run_name, "error": r.error}
            for r in runs_missing
        ],
        "failed_loads": [
            {"run_id": r.run_id, "run_name": r.run_name, "error": r.error}
            for r in runs_load_failed
        ],
        "successful_runs": [
            {
                "run_id": r.run_id,
                "run_name": r.run_name,
                "artifact_version": r.artifact_version,
                "num_sae_positions": r.num_sae_positions,
                "metrics_keys": r.metrics_keys
            }
            for r in results if r.can_load
        ]
    }
    
    # Save report to file
    report_path = Path("metrics_verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Final status
    print("\n" + "=" * 60)
    if runs_can_load == total_runs:
        print("✅ SUCCESS: All runs have metrics artifacts that can be loaded!")
    elif runs_with_artifacts == total_runs:
        print("⚠️  PARTIAL SUCCESS: All runs have artifacts, but some fail to load")
    else:
        print(f"❌ INCOMPLETE: {total_runs - runs_with_artifacts} runs still need evaluation")
    print("=" * 60)
    
    return report


def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify all runs have metrics artifacts and can be loaded"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="raymondl/tinystories-1m",
        help="Wandb project in format 'entity/project'"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by name (e.g., 'relu', 'gated', 'hardconcrete')"
    )
    
    args = parser.parse_args()
    
    # Run verification
    report = check_all_runs_metrics(
        project=args.project,
        filter_name=args.filter
    )
    
    # Return exit code based on success
    if report["summary"]["success_rate"] == 1.0:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some runs missing or failed


if __name__ == "__main__":
    main() 