#!/usr/bin/env python3
"""
Generate a training summary for a specific Weights & Biases run.

This script fetches a run from wandb and generates a summary showing:
- Min, max, variation (std dev), and final value for each logged metric
- Training progress overview
- Run configuration details
"""

import argparse
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import json

from settings import settings


def extract_run_info_from_url(url: str) -> tuple[str, str, str]:
    """
    Extract entity, project, and run_id from a wandb URL.
    
    Args:
        url: Wandb run URL like https://wandb.ai/entity/project/runs/run_id
        
    Returns:
        Tuple of (entity, project, run_id)
    """
    # Remove query parameters
    url = url.split('?')[0]
    
    # Expected format: https://wandb.ai/entity/project/runs/run_id
    parts = url.split('/')
    
    if len(parts) < 6 or 'wandb.ai' not in url:
        raise ValueError(f"Invalid wandb URL format: {url}")
    
    entity = parts[-4]
    project = parts[-3] 
    run_id = parts[-1]
    
    return entity, project, run_id


def analyze_metric_history(history: list, metric_name: str) -> Dict[str, Any]:
    """
    Analyze a single metric's training history.
    
    Args:
        history: List of metric values over time
        metric_name: Name of the metric
        
    Returns:
        Dictionary with min, max, variation, final value, and other stats
    """
    if not history:
        return {
            'metric_name': metric_name,
            'num_points': 0,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'final': None,
            'initial': None,
            'trend': 'N/A'
        }
    
    values = np.array(history)
    # Filter out NaN values
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) == 0:
        return {
            'metric_name': metric_name,
            'num_points': len(history),
            'min': None,
            'max': None, 
            'mean': None,
            'std': None,
            'final': None,
            'initial': None,
            'trend': 'All NaN'
        }
    
    # Calculate trend (improving, degrading, stable)
    trend = 'stable'
    if len(clean_values) > 1:
        initial_val = clean_values[0]
        final_val = clean_values[-1]
        change_pct = ((final_val - initial_val) / abs(initial_val)) * 100 if initial_val != 0 else 0
        
        if abs(change_pct) > 5:  # More than 5% change
            # For loss metrics, decreasing is improving
            if 'loss' in metric_name.lower() or 'mse' in metric_name.lower():
                trend = 'improving' if change_pct < 0 else 'degrading'
            else:
                # For accuracy/performance metrics, increasing is improving
                trend = 'improving' if change_pct > 0 else 'degrading'
    
    return {
        'metric_name': metric_name,
        'num_points': len(history),
        'min': float(np.min(clean_values)),
        'max': float(np.max(clean_values)),
        'mean': float(np.mean(clean_values)),
        'std': float(np.std(clean_values)),
        'final': float(clean_values[-1]),
        'initial': float(clean_values[0]),
        'trend': trend,
        'change_pct': ((clean_values[-1] - clean_values[0]) / abs(clean_values[0])) * 100 if len(clean_values) > 1 and clean_values[0] != 0 else 0
    }


def generate_training_summary(entity: str, project: str, run_id: str, 
                            output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive training summary for a wandb run.
    
    Args:
        entity: Wandb entity name
        project: Wandb project name  
        run_id: Wandb run ID
        output_file: Optional path to save summary as JSON
        
    Returns:
        Dictionary containing the complete summary
    """
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Fetch the run
    run_path = f"{entity}/{project}/{run_id}"
    print(f"Fetching run: {run_path}")
    
    try:
        run = api.run(run_path)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch run {run_path}: {e}")
    
    print(f"Run name: {run.name}")
    print(f"Run state: {run.state}")
    print(f"Created: {run.created_at}")
    print(f"Runtime: {run._attrs.get('duration', 'N/A')}")
    
    # Get run history (all logged metrics over time)
    print("Fetching run history...")
    history = run.history()
    
    if history.empty:
        print("Warning: No training history found for this run")
        return {
            'run_info': {
                'entity': entity,
                'project': project,
                'run_id': run_id,
                'name': run.name,
                'state': run.state,
                'created_at': str(run.created_at),
                'url': run.url
            },
            'config': dict(run.config),
            'metrics_summary': {},
            'total_metrics': 0
        }
    
    print(f"Found {len(history)} training steps with {len(history.columns)} metrics")
    print(f"Metrics: {list(history.columns)}")
    
    # Analyze each metric
    metrics_summary = {}
    
    # Exclude non-metric columns and include only train/ and eval/ metrics
    exclude_cols = {'_step', '_runtime', '_timestamp'}
    metric_columns = [col for col in history.columns 
                     if col not in exclude_cols and (col.startswith('train/') or col.startswith('eval/'))]
    
    print(f"\nAnalyzing {len(metric_columns)} metrics...")
    
    for metric_name in metric_columns:
        metric_data = history[metric_name].dropna().tolist()
        analysis = analyze_metric_history(metric_data, metric_name)
        metrics_summary[metric_name] = analysis
        
        # Print brief summary
        if analysis['num_points'] > 0:
            print(f"  {metric_name}: {analysis['final']:.6f} (final), "
                  f"{analysis['min']:.6f}-{analysis['max']:.6f} range, "
                  f"trend: {analysis['trend']}")
        else:
            print(f"  {metric_name}: No data")
    
    # Compile full summary
    summary = {
        'run_info': {
            'entity': entity,
            'project': project,
            'run_id': run_id,
            'name': run.name,
            'state': run.state,
            'created_at': str(run.created_at),
            'url': run.url,
            'total_steps': len(history),
            'runtime_seconds': run._attrs.get('duration', None)
        },
        'config': dict(run.config),
        'metrics_summary': metrics_summary,
        'total_metrics': len(metrics_summary)
    }
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to: {output_path}")
    
    return summary


def print_formatted_summary(summary: Dict[str, Any]):
    """Print a nicely formatted version of the summary."""
    
    print("\n" + "=" * 80)
    print("WANDB RUN TRAINING SUMMARY")
    print("=" * 80)
    
    # Run info
    run_info = summary['run_info']
    print(f"\nRun Information:")
    print(f"  Name: {run_info['name']}")
    print(f"  Entity/Project: {run_info['entity']}/{run_info['project']}")
    print(f"  Run ID: {run_info['run_id']}")
    print(f"  State: {run_info['state']}")
    print(f"  Created: {run_info['created_at']}")
    print(f"  URL: {run_info['url']}")
    print(f"  Total Steps: {run_info.get('total_steps', 'N/A')}")
    print(f"  Runtime: {run_info.get('runtime_seconds', 'N/A')} seconds")
    
    # Config summary (just show key parameters)
    config = summary['config']
    print(f"\nKey Configuration:")
    key_params = ['lr', 'batch_size', 'warmup_samples', 'total_samples', 'seed']
    for param in key_params:
        if param in config:
            print(f"  {param}: {config[param]}")
    
    # Metrics summary
    print(f"\n" + "=" * 80)
    print(f"METRICS ANALYSIS ({summary['total_metrics']} metrics)")
    print("=" * 80)
    
    if not summary['metrics_summary']:
        print("No metrics data found.")
        return
    
    # Sort metrics by name for consistent output
    sorted_metrics = sorted(summary['metrics_summary'].items())
    
    print(f"\n{'Metric':<35} {'Final':<12} {'Min':<12} {'Max':<12} {'Std':<12} {'Trend':<12}")
    print("-" * 95)
    
    for metric_name, analysis in sorted_metrics:
        if analysis['num_points'] == 0:
            print(f"{metric_name:<35} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            final = f"{analysis['final']:.6f}" if analysis['final'] is not None else "N/A"
            min_val = f"{analysis['min']:.6f}" if analysis['min'] is not None else "N/A"
            max_val = f"{analysis['max']:.6f}" if analysis['max'] is not None else "N/A"
            std_val = f"{analysis['std']:.6f}" if analysis['std'] is not None else "N/A"
            trend = analysis['trend']
            
            print(f"{metric_name:<35} {final:<12} {min_val:<12} {max_val:<12} {std_val:<12} {trend:<12}")
    
    # Show detailed analysis for key metrics
    key_metrics = ['loss', 'mse', 'explained_variance', 'sparsity_l0']
    detailed_metrics = [name for name in summary['metrics_summary'].keys() 
                       if any(key in name.lower() for key in key_metrics)]
    
    if detailed_metrics:
        print(f"\n" + "=" * 80)
        print("DETAILED ANALYSIS - KEY METRICS")
        print("=" * 80)
        
        for metric_name in detailed_metrics[:5]:  # Limit to top 5 for readability
            analysis = summary['metrics_summary'][metric_name]
            if analysis['num_points'] > 0:
                print(f"\n{metric_name}:")
                print(f"  Data points: {analysis['num_points']}")
                print(f"  Final value: {analysis['final']:.8f}")
                print(f"  Range: {analysis['min']:.8f} to {analysis['max']:.8f}")
                print(f"  Mean: {analysis['mean']:.8f} ± {analysis['std']:.8f}")
                print(f"  Change: {analysis['change_pct']:.2f}% ({analysis['trend']})")


def main():
    """Main function to run the training summary generator."""
    parser = argparse.ArgumentParser(
        description="Generate training summary from Wandb run URL or run details"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Full Wandb run URL (e.g., https://wandb.ai/entity/project/runs/run_id)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="Wandb entity name (alternative to URL)"
    )
    parser.add_argument(
        "--project", 
        type=str,
        help="Wandb project name (alternative to URL)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Wandb run ID (alternative to URL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save summary as JSON (optional)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final formatted summary, suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Extract run details
    if args.url:
        try:
            entity, project, run_id = extract_run_info_from_url(args.url)
        except ValueError as e:
            print(f"Error parsing URL: {e}")
            return 1
    elif args.entity and args.project and args.run_id:
        entity, project, run_id = args.entity, args.project, args.run_id
    else:
        print("Error: Must provide either --url OR --entity, --project, and --run-id")
        return 1
    
    if not args.quiet:
        print(f"Analyzing run: {entity}/{project}/{run_id}")
    
    try:
        # Generate summary
        summary = generate_training_summary(
            entity=entity,
            project=project,
            run_id=run_id,
            output_file=args.output
        )
        
        # Print formatted summary
        print_formatted_summary(summary)
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 