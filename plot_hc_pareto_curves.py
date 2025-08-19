"""
Plot Pareto curves for HardConcrete SAE runs with different configurations.

Configurations:
- no_mag vs mag (apply_relu_to_magnitude: false vs true)
- coefficient_threshold: None vs 1e-6
"""

import json
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from settings import settings
from utils.io import load_metrics_from_wandb


def categorize_run(run_name: str) -> str:
    """Categorize a run based on its name.
    
    Returns:
        mag_category: 'mag' or 'no_mag'
    """
    # Check magnitude setting only, ignore threshold
    if 'apply_relu_to_magnitude_true' in run_name:
        return 'mag'
    elif 'apply_relu_to_magnitude_false' in run_name:
        return 'no_mag'
    else:
        return None  # Skip runs that don't match pattern


def extract_sparsity_coeff(run_name: str) -> float:
    """Extract sparsity coefficient from run name."""
    parts = run_name.split('_')
    for i, part in enumerate(parts):
        if part == 'coeff' and i + 1 < len(parts):
            coeff_str = parts[i + 1]
            # Handle scientific notation
            if 'e-' in coeff_str:
                return float(coeff_str)
            else:
                return float(coeff_str)
    return None


def load_run_metrics(run_id: str, run_name: str, project: str) -> Dict[str, Any]:
    """Load metrics for a single run."""
    print(f"  Loading metrics for run {run_name} (ID: {run_id})...")
    metrics = load_metrics_from_wandb(run_id, project=project)
    
    if metrics is None:
        print(f"  Warning: No metrics found for run {run_id}")
        return None
    
    # Extract sparsity coefficient from run name
    sparsity_coeff = extract_sparsity_coeff(run_name)
    
    return {
        'run_id': run_id,
        'run_name': run_name,
        'sparsity_coeff': sparsity_coeff,
        'metrics': metrics
    }


def compute_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute the Pareto frontier for minimizing both objectives.
    
    Args:
        points: List of (x, y) tuples where we want to minimize both x and y
    
    Returns:
        List of points on the Pareto frontier, sorted by x coordinate
    """
    if not points:
        return []
    
    # Sort points by x coordinate
    sorted_points = sorted(points, key=lambda p: p[0])
    
    pareto_points = []
    min_y_so_far = float('inf')
    
    for point in sorted_points:
        x, y = point
        # A point is on the Pareto frontier if it has the lowest y seen so far
        # (since we're iterating in order of increasing x)
        if y < min_y_so_far:
            pareto_points.append(point)
            min_y_so_far = y
    
    return pareto_points


def plot_pareto_curves(results_by_config: Dict[str, List[Dict]], layer_name: str, output_dir: Path):
    """Plot Pareto curves for different configurations."""
    
    # Set up the plot with better styling
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for each configuration
    config_styles = {
        'mag': {'color': 'blue', 'marker': 'o', 'label': 'With Magnitude (ReLU)'},
        'no_mag': {'color': 'red', 'marker': 's', 'label': 'Without Magnitude (ReLU)'},
    }
    
    # Plot each configuration
    for config_key, runs in results_by_config.items():
        if not runs:
            continue
        
        style = config_styles.get(config_key, {})
        
        # Extract points (L0, MSE) for this configuration
        points = []
        for run_data in runs:
            if layer_name in run_data['metrics']:
                layer_metrics = run_data['metrics'][layer_name]
                l0 = layer_metrics.get('sparsity_l0', None)
                mse = layer_metrics.get('mse', None)
                
                if l0 is not None and mse is not None:
                    points.append((l0, mse))
        
        if not points:
            continue
        
        # Sort points by L0 for consistent plotting
        points = sorted(points, key=lambda p: p[0])
        
        # Plot all points
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        plt.scatter(x_vals, y_vals, 
                   color=style.get('color', 'gray'),
                   marker=style.get('marker', 'o'),
                   alpha=0.6, s=50,
                   label=f"{style.get('label', str(config_key))} (points)")
        
        # Compute and plot Pareto frontier
        pareto_points = compute_pareto_frontier(points)
        if pareto_points:
            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]
            plt.plot(pareto_x, pareto_y,
                    color=style.get('color', 'gray'),
                    linewidth=2,
                    label=f"{style.get('label', str(config_key))} (Pareto)",
                    marker=style.get('marker', 'o'),
                    markersize=8)
    
    # Formatting
    plt.xlabel('L0 Sparsity', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'Pareto Curves - {layer_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # Use log scale for MSE if range is large
    y_min, y_max = plt.ylim()
    if y_max / y_min > 100:
        plt.yscale('log')
    
    # Save the plot
    output_file = output_dir / f'pareto_curves_{layer_name.replace(".", "_")}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.svg'), bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def plot_combined_pareto_curves(all_results: Dict[str, List[Dict]], output_dir: Path):
    """Plot combined Pareto curves for all layers."""
    
    # Get all unique layers across all configurations
    all_layers = set()
    for runs in all_results.values():
        for run_data in runs:
            if run_data and 'metrics' in run_data:
                all_layers.update(run_data['metrics'].keys())
    
    # Sort layers for consistent ordering
    sorted_layers = sorted(all_layers)
    
    # Create subplots
    n_layers = len(sorted_layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colors and markers for each configuration
    config_styles = {
        'mag': {'color': 'blue', 'marker': 'o', 'label': 'With ReLU'},
        'no_mag': {'color': 'red', 'marker': 's', 'label': 'Without ReLU'},
    }
    
    for idx, layer_name in enumerate(sorted_layers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot each configuration for this layer
        for config_key, runs in all_results.items():
            style = config_styles.get(config_key, {})
            
            # Extract points (L0, MSE) for this configuration and layer
            points = []
            for run_data in runs:
                if run_data and 'metrics' in run_data and layer_name in run_data['metrics']:
                    layer_metrics = run_data['metrics'][layer_name]
                    l0 = layer_metrics.get('sparsity_l0', None)
                    mse = layer_metrics.get('mse', None)
                    
                    if l0 is not None and mse is not None:
                        points.append((l0, mse))
            
            if not points:
                continue
            
            # Compute and plot Pareto frontier only
            pareto_points = compute_pareto_frontier(points)
            if pareto_points:
                pareto_x = [p[0] for p in pareto_points]
                pareto_y = [p[1] for p in pareto_points]
                ax.plot(pareto_x, pareto_y,
                       color=style.get('color', 'gray'),
                       linewidth=2,
                       label=style.get('label', str(config_key)),
                       marker=style.get('marker', 'o'),
                       markersize=6,
                       alpha=0.8)
        
        # Formatting
        ax.set_xlabel('L0 Sparsity', fontsize=10)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title(layer_name.replace('blocks.', 'Layer ').replace('.hook_resid_pre', ''), fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    # Remove empty subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    # Overall title
    fig.suptitle('HardConcrete SAE Pareto Curves - All Layers', fontsize=14, fontweight='bold', y=1.02)
    
    # Save the plot
    output_file = output_dir / 'pareto_curves_combined.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.svg'), bbox_inches='tight')
    print(f"Saved combined plot to {output_file}")
    plt.close()


def main():
    """Main function to analyze and plot HardConcrete Pareto curves."""
    
    # Setup
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get the project
    project = "raymondl/tinystories-1m-hardconcrete"
    
    print(f"Fetching runs from project: {project}")
    runs = list(api.runs(project))
    print(f"Found {len(runs)} runs")
    
    # Categorize runs by configuration (ignoring threshold)
    results_by_config = {
        'mag': [],
        'no_mag': [],
    }
    
    # Process each run
    for run in tqdm(runs, desc="Processing runs"):
        print(f"\nProcessing run: {run.name} (ID: {run.id})")
        
        # Categorize the run (ignoring threshold)
        mag_cat = categorize_run(run.name)
        if mag_cat is None:
            print(f"  Skipping run {run.name} - doesn't match expected pattern")
            continue
        
        # Load metrics for this run
        run_data = load_run_metrics(
            run_id=run.id,
            run_name=run.name,
            project=project
        )
        
        if run_data is not None:
            results_by_config[mag_cat].append(run_data)
    
    # Print summary
    print(f"\n" + "="*60)
    print("Summary by Configuration:")
    print("="*60)
    for config_key, runs in results_by_config.items():
        print(f"{config_key}: {len(runs)} runs")
    
    # Create output directory
    output_dir = Path('plots/hardconcrete_pareto')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all unique layers
    all_layers = set()
    for runs in results_by_config.values():
        for run_data in runs:
            if run_data and 'metrics' in run_data:
                all_layers.update(run_data['metrics'].keys())
    
    # Plot individual Pareto curves for each layer
    print(f"\n" + "="*60)
    print("Plotting Pareto curves...")
    print("="*60)
    
    for layer_name in sorted(all_layers):
        print(f"Plotting {layer_name}...")
        plot_pareto_curves(results_by_config, layer_name, output_dir)
    
    # Plot combined view
    print("\nPlotting combined view...")
    plot_combined_pareto_curves(results_by_config, output_dir)
    
    # Save raw data for future analysis
    output_data = {
        'project': project,
        'configurations': {
            config_key: [
                {
                    'run_id': r['run_id'],
                    'run_name': r['run_name'],
                    'sparsity_coeff': r['sparsity_coeff'],
                    'metrics': r['metrics']
                }
                for r in runs
            ]
            for config_key, runs in results_by_config.items()
        }
    }
    
    output_file = output_dir / 'pareto_data.json'
    print(f"\nSaving raw data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nAll plots saved to {output_dir}/")
    print("Done!")
    
    return results_by_config


if __name__ == "__main__":
    results = main() 