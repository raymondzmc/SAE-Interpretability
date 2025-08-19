#!/usr/bin/env python3
"""
Plot Pareto curves for ReLU and Gated SAEs - separate plots for each layer.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from settings import settings
from utils.io import load_metrics_from_wandb

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9


def collect_metrics_data(project: str = "raymondl/tinystories-1m") -> Dict[str, List[Dict]]:
    """
    Collect metrics data for ReLU and Gated runs with per-layer information.
    
    Returns:
        Dictionary with 'relu' and 'gated' keys, each containing list of run data
    """
    print("Collecting metrics data from Wandb...")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs
    runs = list(api.runs(project))
    
    # Collect data by SAE type
    data = {
        'relu': [],
        'gated': []
    }
    
    # Track layer names
    all_layers = set()
    
    for run in runs:
        name_lower = run.name.lower()
        
        # Determine SAE type
        sae_type = None
        if 'relu' in name_lower:
            sae_type = 'relu'
        elif 'gated' in name_lower:
            sae_type = 'gated'
        else:
            continue  # Skip other types
        
        # Load metrics for this run
        print(f"  Loading metrics for {run.name} ({run.id})...")
        metrics = load_metrics_from_wandb(run.id, project)
        
        if metrics:
            # Extract sparsity coefficient from run name
            sparsity_coeff = None
            if 'sparsity_coeff_' in run.name:
                coeff_str = run.name.split('sparsity_coeff_')[1]
                try:
                    sparsity_coeff = float(coeff_str.replace('e-', 'e-'))
                except:
                    sparsity_coeff = None
            
            # Store per-layer metrics
            run_data = {
                'run_name': run.name,
                'run_id': run.id,
                'sparsity_coeff': sparsity_coeff,
                'layers': {}
            }
            
            # Process each layer
            for layer_name, layer_metrics in metrics.items():
                all_layers.add(layer_name)
                run_data['layers'][layer_name] = {
                    'l0': layer_metrics['sparsity_l0'],
                    'mse': layer_metrics['mse'],
                    'explained_variance': layer_metrics['explained_variance'],
                    'alive_dict_components': layer_metrics['alive_dict_components'],
                    'alive_dict_proportion': layer_metrics['alive_dict_components_proportion']
                }
            
            data[sae_type].append(run_data)
    
    # Sort by average L0 sparsity for consistency
    for sae_type in data:
        data[sae_type] = sorted(data[sae_type], 
                                key=lambda x: np.mean([m['l0'] for m in x['layers'].values()]))
    
    print(f"Collected data for {len(data['relu'])} ReLU runs and {len(data['gated'])} Gated runs")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    
    return data, sorted(all_layers)


def find_pareto_frontier(x_values: np.ndarray, y_values: np.ndarray, 
                        minimize_x: bool = True, minimize_y: bool = True) -> np.ndarray:
    """
    Find the Pareto frontier for 2D data.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        minimize_x: If True, prefer smaller x values
        minimize_y: If True, prefer smaller y values
    
    Returns:
        Boolean array indicating which points are on the Pareto frontier
    """
    n_points = len(x_values)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if minimize_x and minimize_y:
                    # Both objectives should be minimized
                    if x_values[j] <= x_values[i] and y_values[j] <= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] < y_values[i]:
                            is_pareto[i] = False
                            break
                elif minimize_x and not minimize_y:
                    # Minimize x, maximize y
                    if x_values[j] <= x_values[i] and y_values[j] >= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] > y_values[i]:
                            is_pareto[i] = False
                            break
    
    return is_pareto


def plot_pareto_curves_per_layer(data: Dict[str, List[Dict]], layers: List[str], 
                                 output_dir: Path = Path("plots"),
                                 max_mse: float = 0.05, max_l0: float = 250):
    """
    Create Pareto curve plots for each layer separately.
    
    Args:
        data: Dictionary with SAE type data
        layers: List of layer names
        output_dir: Output directory for plots
        max_mse: Maximum MSE threshold (default: 0.05)
        max_l0: Maximum L0 threshold (default: 250)
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nFiltering: Removing runs with MSE > {max_mse} or L0 > {max_l0}")
    
    # Color scheme
    colors = {
        'relu': '#2E86AB',  # Blue
        'gated': '#A23B72'  # Purple
    }
    
    # Marker styles
    markers = {
        'relu': 'o',
        'gated': 's'
    }
    
    # Track filtered statistics
    total_filtered = 0
    filtered_by_type = {'relu': 0, 'gated': 0}
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing layer: {layer_name}")
        layer_filtered = 0
        
        # Create figure with 3 subplots for this layer
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: MSE vs L0 (minimize both)
        ax1 = axes[0]
        for sae_type in ['relu', 'gated']:
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                mse_values = []
                run_names = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            mse_values.append(mse)
                            run_names.append(run_data['run_name'])
                        else:
                            layer_filtered += 1
                            filtered_by_type[sae_type] += 1
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                mse_values = np.array(mse_values)
                
                # Find Pareto frontier (minimize both MSE and L0)
                is_pareto = find_pareto_frontier(l0_values, mse_values, 
                                                minimize_x=True, minimize_y=True)
                
                # Plot all points
                ax1.scatter(l0_values, mse_values, 
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.6, s=40, label=f'{sae_type.upper()} runs')
                
                # Highlight and connect Pareto frontier points
                if np.any(is_pareto):
                    pareto_l0 = l0_values[is_pareto]
                    pareto_mse = mse_values[is_pareto]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_mse = pareto_mse[sort_idx]
                    
                    ax1.plot(pareto_l0, pareto_mse, 
                            color=colors[sae_type], linewidth=2, alpha=0.8,
                            label=f'{sae_type.upper()} Pareto')
                    ax1.scatter(pareto_l0, pareto_mse,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=80, edgecolors='black', linewidth=1.5, zorder=5)
        
        ax1.set_xlabel('L0 Sparsity', fontsize=11)
        ax1.set_ylabel('MSE', fontsize=11)
        ax1.set_title('MSE vs L0 Sparsity\n(Lower is better for both)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Explained Variance vs L0 (minimize L0, maximize explained variance)
        ax2 = axes[1]
        for sae_type in ['relu', 'gated']:
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                ev_values = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        ev = run_data['layers'][layer_name]['explained_variance']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            ev_values.append(ev)
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                ev_values = np.array(ev_values)
                
                # Find Pareto frontier (minimize L0, maximize explained variance)
                is_pareto = find_pareto_frontier(l0_values, -ev_values,  # Negate EV to find max
                                                minimize_x=True, minimize_y=True)
                
                # Plot all points
                ax2.scatter(l0_values, ev_values,
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.6, s=40, label=f'{sae_type.upper()} runs')
                
                # Highlight and connect Pareto frontier points
                if np.any(is_pareto):
                    pareto_l0 = l0_values[is_pareto]
                    pareto_ev = ev_values[is_pareto]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_ev = pareto_ev[sort_idx]
                    
                    ax2.plot(pareto_l0, pareto_ev,
                            color=colors[sae_type], linewidth=2, alpha=0.8,
                            label=f'{sae_type.upper()} Pareto')
                    ax2.scatter(pareto_l0, pareto_ev,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=80, edgecolors='black', linewidth=1.5, zorder=5)
        
        ax2.set_xlabel('L0 Sparsity', fontsize=11)
        ax2.set_ylabel('Explained Variance', fontsize=11)
        ax2.set_title('Explained Variance vs L0 Sparsity\n(Lower L0 better, Higher EV better)', fontsize=12)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Alive Dictionary Elements vs L0
        ax3 = axes[2]
        for sae_type in ['relu', 'gated']:
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                alive_values = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        alive = run_data['layers'][layer_name]['alive_dict_components']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            alive_values.append(alive)
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                alive_values = np.array(alive_values)
                
                # Find Pareto frontier (minimize L0, maximize alive components)
                is_pareto = find_pareto_frontier(l0_values, -alive_values,  # Negate to find max
                                                minimize_x=True, minimize_y=True)
                
                # Plot all points
                ax3.scatter(l0_values, alive_values,
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.6, s=40, label=f'{sae_type.upper()} runs')
                
                # Highlight and connect Pareto frontier points
                if np.any(is_pareto):
                    pareto_l0 = l0_values[is_pareto]
                    pareto_alive = alive_values[is_pareto]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_alive = pareto_alive[sort_idx]
                    
                    ax3.plot(pareto_l0, pareto_alive,
                            color=colors[sae_type], linewidth=2, alpha=0.8,
                            label=f'{sae_type.upper()} Pareto')
                    ax3.scatter(pareto_l0, pareto_alive,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=80, edgecolors='black', linewidth=1.5, zorder=5)
        
        ax3.set_xlabel('L0 Sparsity', fontsize=11)
        ax3.set_ylabel('Alive Dictionary Components', fontsize=11)
        ax3.set_title('Alive Dictionary Components vs L0\n(Lower L0 better, Higher alive better)', fontsize=12)
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        plt.suptitle(f'Pareto Curves: ReLU vs Gated SAEs - Layer {layer_name}\n(Filtered: MSE ≤ {max_mse}, L0 ≤ {max_l0})', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"relu_gated_pareto_{layer_display_name}_filtered.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved plot to: {output_path}")
        print(f"  Filtered {layer_filtered} points from this layer")
        
        # Also save as SVG for vector graphics
        output_path_svg = output_dir / f"relu_gated_pareto_{layer_display_name}_filtered.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()
        
        total_filtered += layer_filtered
    
    # Create a combined figure with all layers for comparison
    print(f"\nCreating combined multi-layer comparison...")
    print(f"Total points filtered: {total_filtered} (ReLU: {filtered_by_type['relu']}, Gated: {filtered_by_type['gated']})")
    
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 3, figsize=(18, 5 * n_layers))
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, layer_name in enumerate(layers):
        # MSE vs L0
        ax1 = axes[layer_idx, 0]
        # Explained Variance vs L0
        ax2 = axes[layer_idx, 1]
        # Alive Dictionary vs L0
        ax3 = axes[layer_idx, 2]
        
        for sae_type in ['relu', 'gated']:
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                mse_values = []
                ev_values = []
                alive_values = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        ev = run_data['layers'][layer_name]['explained_variance']
                        alive = run_data['layers'][layer_name]['alive_dict_components']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            mse_values.append(mse)
                            ev_values.append(ev)
                            alive_values.append(alive)
                
                if not l0_values:
                    continue
                
                l0_values = np.array(l0_values)
                mse_values = np.array(mse_values)
                ev_values = np.array(ev_values)
                alive_values = np.array(alive_values)
                
                # Plot MSE vs L0
                is_pareto_mse = find_pareto_frontier(l0_values, mse_values, 
                                                    minimize_x=True, minimize_y=True)
                ax1.scatter(l0_values, mse_values, 
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.6, s=30, label=f'{sae_type.upper()}')
                if np.any(is_pareto_mse):
                    pareto_l0 = l0_values[is_pareto_mse]
                    pareto_mse = mse_values[is_pareto_mse]
                    sort_idx = np.argsort(pareto_l0)
                    ax1.plot(pareto_l0[sort_idx], pareto_mse[sort_idx], 
                            color=colors[sae_type], linewidth=1.5, alpha=0.8)
                
                # Plot EV vs L0
                is_pareto_ev = find_pareto_frontier(l0_values, -ev_values,
                                                   minimize_x=True, minimize_y=True)
                ax2.scatter(l0_values, ev_values,
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.6, s=30, label=f'{sae_type.upper()}')
                if np.any(is_pareto_ev):
                    pareto_l0 = l0_values[is_pareto_ev]
                    pareto_ev = ev_values[is_pareto_ev]
                    sort_idx = np.argsort(pareto_l0)
                    ax2.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                            color=colors[sae_type], linewidth=1.5, alpha=0.8)
                
                # Plot Alive vs L0
                is_pareto_alive = find_pareto_frontier(l0_values, -alive_values,
                                                      minimize_x=True, minimize_y=True)
                ax3.scatter(l0_values, alive_values,
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.6, s=30, label=f'{sae_type.upper()}')
                if np.any(is_pareto_alive):
                    pareto_l0 = l0_values[is_pareto_alive]
                    pareto_alive = alive_values[is_pareto_alive]
                    sort_idx = np.argsort(pareto_l0)
                    ax3.plot(pareto_l0[sort_idx], pareto_alive[sort_idx],
                            color=colors[sae_type], linewidth=1.5, alpha=0.8)
        
        # Labels and formatting
        ax1.set_ylabel(f'Layer {layer_idx+1}\nMSE', fontsize=10)
        ax2.set_ylabel(f'Layer {layer_idx+1}\nExplained Variance', fontsize=10)
        ax3.set_ylabel(f'Layer {layer_idx+1}\nAlive Components', fontsize=10)
        
        if layer_idx == 0:
            ax1.set_title('MSE vs L0 Sparsity', fontsize=11)
            ax2.set_title('Explained Variance vs L0 Sparsity', fontsize=11)
            ax3.set_title('Alive Dictionary vs L0 Sparsity', fontsize=11)
            ax1.legend(loc='upper right', fontsize=8)
            ax2.legend(loc='lower right', fontsize=8)
            ax3.legend(loc='lower right', fontsize=8)
        
        if layer_idx == n_layers - 1:
            ax1.set_xlabel('L0 Sparsity', fontsize=10)
            ax2.set_xlabel('L0 Sparsity', fontsize=10)
            ax3.set_xlabel('L0 Sparsity', fontsize=10)
        
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
    
    plt.suptitle(f'Pareto Curves: ReLU vs Gated SAEs - All Layers Comparison\n(Filtered: MSE ≤ {max_mse}, L0 ≤ {max_l0})', 
                fontsize=14, y=1.01)
    plt.tight_layout()
    
    # Save combined figure
    output_path = output_dir / "relu_gated_pareto_all_layers_filtered.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved combined plot to: {output_path}")
    
    output_path_svg = output_dir / "relu_gated_pareto_all_layers_filtered.svg"
    plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
    print(f"Saved combined SVG to: {output_path_svg}")
    
    plt.show()


def print_pareto_summary_per_layer(data: Dict[str, List[Dict]], layers: List[str],
                                   max_mse: float = 0.05, max_l0: float = 250):
    """Print a summary of the Pareto-optimal points for each layer."""
    
    print("\n" + "=" * 70)
    print(f"PARETO FRONTIER SUMMARY - PER LAYER (Filtered: MSE ≤ {max_mse}, L0 ≤ {max_l0})")
    print("=" * 70)
    
    for layer_name in layers:
        print(f"\n{'='*70}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*70}")
        
        for sae_type in ['relu', 'gated']:
            if not data[sae_type]:
                continue
            
            print(f"\n{sae_type.upper()} SAE:")
            print("-" * 35)
            
            # Extract layer-specific data WITH FILTERING
            runs_data = []
            filtered_count = 0
            for run_data in data[sae_type]:
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    mse = run_data['layers'][layer_name]['mse']
                    
                    if l0 <= max_l0 and mse <= max_mse:
                        runs_data.append({
                            'name': run_data['run_name'],
                            'l0': l0,
                            'mse': mse,
                            'ev': run_data['layers'][layer_name]['explained_variance'],
                            'alive': run_data['layers'][layer_name]['alive_dict_components']
                        })
                    else:
                        filtered_count += 1
            
            if not runs_data:
                print(f"  No data available after filtering ({filtered_count} runs filtered)")
                continue
            elif filtered_count > 0:
                print(f"  ({filtered_count} runs filtered out)")
            
            l0_values = np.array([r['l0'] for r in runs_data])
            mse_values = np.array([r['mse'] for r in runs_data])
            ev_values = np.array([r['ev'] for r in runs_data])
            
            # Find Pareto points
            is_pareto_mse = find_pareto_frontier(l0_values, mse_values, 
                                                minimize_x=True, minimize_y=True)
            is_pareto_ev = find_pareto_frontier(l0_values, -ev_values,
                                               minimize_x=True, minimize_y=True)
            
            # Get runs on either Pareto frontier
            pareto_runs = []
            for i, run in enumerate(runs_data):
                if is_pareto_mse[i] or is_pareto_ev[i]:
                    pareto_info = {
                        'name': run['name'],
                        'l0': run['l0'],
                        'mse': run['mse'],
                        'ev': run['ev'],
                        'alive': run['alive'],
                        'on_mse_frontier': is_pareto_mse[i],
                        'on_ev_frontier': is_pareto_ev[i]
                    }
                    pareto_runs.append(pareto_info)
            
            # Sort by L0
            pareto_runs = sorted(pareto_runs, key=lambda x: x['l0'])
            
            if pareto_runs:
                print(f"  Pareto-optimal runs ({len(pareto_runs)} total):")
                for run in pareto_runs[:5]:  # Show top 5 to save space
                    frontiers = []
                    if run['on_mse_frontier']:
                        frontiers.append('MSE')
                    if run['on_ev_frontier']:
                        frontiers.append('EV')
                    frontier_str = '+'.join(frontiers)
                    
                    print(f"    • {run['name']:28} L0={run['l0']:6.2f}, MSE={run['mse']:.4f}, "
                          f"EV={run['ev']:.4f} [{frontier_str}]")
                
                if len(pareto_runs) > 5:
                    print(f"    ... and {len(pareto_runs) - 5} more")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves for ReLU and Gated SAEs per layer"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="raymondl/tinystories-1m",
        help="Wandb project"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=0.05,
        help="Maximum MSE threshold for filtering (default: 0.05)"
    )
    parser.add_argument(
        "--max-l0",
        type=float,
        default=250,
        help="Maximum L0 threshold for filtering (default: 250)"
    )
    
    args = parser.parse_args()
    
    # Collect data
    data, layers = collect_metrics_data(args.project)
    
    # Create plots with filtering
    plot_pareto_curves_per_layer(data, layers, Path(args.output_dir), 
                                 max_mse=args.max_mse, max_l0=args.max_l0)
    
    # Print summary with filtering
    print_pareto_summary_per_layer(data, layers, max_mse=args.max_mse, max_l0=args.max_l0)


if __name__ == "__main__":
    main() 