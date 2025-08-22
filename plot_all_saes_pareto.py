#!/usr/bin/env python3
"""
Plot Pareto curves for all SAE types: ReLU, Gated, and HardConcrete (input-dependent/independent).
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

from settings import settings
from utils.io import load_metrics_from_wandb
from models import SAETransformer

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9


def collect_all_metrics_data(project: str = "raymondl/tinystories-1m") -> Dict[str, List[Dict]]:
    """
    Collect metrics data for all SAE types including HardConcrete and Top-K.
    
    Returns:
        Dictionary with SAE type keys, each containing list of run data
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
        'gated': [],
        'hardconcrete': [],  # HardConcrete type
        'topk': []  # Top-K type
    }
    
    # Track layer names
    all_layers = set()
    
    for run in runs:
        name_lower = run.name.lower()
        
        # Determine SAE type
        sae_type = None
        if 'topk' in name_lower or 'top-k' in name_lower or 'top_k' in name_lower:
            sae_type = 'topk'
        elif 'relu' in name_lower and 'apply_relu_to_magnitude_true' not in name_lower:
            sae_type = 'relu'
        elif 'gated' in name_lower and 'apply_relu_to_magnitude_true' not in name_lower:
            sae_type = 'gated'
        elif 'apply_relu_to_magnitude_true' in name_lower:
            # This is the new HardConcrete type
            sae_type = 'hardconcrete'
        else:
            continue  # Skip other types
        
        if sae_type is None:
            continue
            
        # Load metrics for this run
        print(f"  Loading metrics for {run.name} ({run.id}) - Type: {sae_type}")
        metrics = load_metrics_from_wandb(run.id, project)
        
        if metrics:
            # Extract sparsity coefficient from run name or config
            sparsity_coeff = None
            k_value = None
            
            # Try to extract sparsity coefficient for any SAE type
            if 'sparsity_coeff_' in run.name:
                coeff_str = run.name.split('sparsity_coeff_')[1].split('_')[0]
                try:
                    # Handle scientific notation
                    coeff_str = coeff_str.replace('_', '.')  # In case underscore is used for decimal
                    sparsity_coeff = float(coeff_str)
                except:
                    sparsity_coeff = None
            
            # For Top-K, also extract k value
            if sae_type == 'topk':
                # Try to extract k from run name first
                if '_k_' in name_lower:
                    try:
                        k_str = name_lower.split('_k_')[1].split('_')[0]
                        k_value = int(k_str)
                    except:
                        pass
                
                # If not found in name, try to load from model
                if k_value is None:
                    try:
                        print(f"    Loading Top-K model to extract k value...")
                        model = SAETransformer.from_wandb(f"{project}/{run.id}")
                        if hasattr(model.sae_config, 'k'):
                            k_value = model.sae_config.k
                        # Clean up model
                        del model
                        import torch
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"    Could not extract k value: {e}")
            
            # Store per-layer metrics
            run_data = {
                'run_name': run.name,
                'run_id': run.id,
                'sparsity_coeff': sparsity_coeff,
                'k_value': k_value,
                'layers': {}
            }
            
            # Process each layer
            for layer_name, layer_metrics in metrics.items():
                all_layers.add(layer_name)
                run_data['layers'][layer_name] = {
                    'l0': layer_metrics['sparsity_l0'],
                    'mse': layer_metrics['mse'],
                    'explained_variance': layer_metrics['explained_variance'],
                    'alive_dict_components': layer_metrics.get('alive_dict_components', 0),
                    'alive_dict_proportion': layer_metrics.get('alive_dict_components_proportion', 0)
                }
            
            data[sae_type].append(run_data)
    
    # Sort by average L0 sparsity for consistency
    for sae_type in data:
        if data[sae_type]:
            data[sae_type] = sorted(data[sae_type], 
                                    key=lambda x: np.mean([m['l0'] for m in x['layers'].values()]))
    
    print(f"\nCollected data summary:")
    for sae_type, runs in data.items():
        print(f"  {sae_type}: {len(runs)} runs")
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


def plot_all_pareto_curves(data: Dict[str, List[Dict]], layers: List[str], 
                           output_dir: Path = Path("plots"),
                           max_mse: float = float('inf'), max_l0: float = float('inf'),
                           min_mse: float = 0.0, min_l0: float = 0.0,
                           use_log_scale: bool = False):
    """
    Create Pareto curve plots for all SAE types.
    
    Args:
        data: Dictionary with SAE type data
        layers: List of layer names
        output_dir: Output directory for plots
        max_mse: Maximum MSE threshold (default: inf - no filtering)
        max_l0: Maximum L0 threshold (default: inf - no filtering)
        use_log_scale: Whether to use log scale for axes (default: False)
    """
    output_dir.mkdir(exist_ok=True)
    
    if max_mse == float('inf') and max_l0 == float('inf'):
        print(f"\nPlotting all data points (no filtering)")
    else:
        print(f"\nFiltering: MSE in [{min_mse}, {max_mse}], L0 in [{min_l0}, {max_l0}]")
    
    # Color scheme for different SAE types - more distinct colors
    colors = {
        'relu': '#1f77b4',           # Blue
        'gated': '#ff7f0e',          # Orange  
        'hardconcrete': '#2ca02c',   # Green
        'topk': '#d62728'            # Red
    }
    
    # Marker styles - more distinct shapes
    markers = {
        'relu': 'o',        # Circle
        'gated': 's',       # Square
        'hardconcrete': '^', # Triangle up
        'topk': 'D'         # Diamond
    }
    
    # Labels for legend
    labels = {
        'relu': 'ReLU',
        'gated': 'Gated',
        'hardconcrete': 'HardConcrete',
        'topk': 'Top-K'
    }
    
    # Track filtered statistics
    total_filtered = 0
    filtered_by_type = {k: 0 for k in data.keys()}
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing layer: {layer_name}")
        layer_filtered = 0
        
        # Create figure with 3 subplots for this layer - larger size
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Plot 1: MSE vs L0 (minimize both)
        ax1 = axes[0]
        for sae_type in ['relu', 'gated', 'hardconcrete', 'topk']:
            if data.get(sae_type):
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                mse_values = []
                run_names = []
                param_labels = []  # Will store sparsity_coeff for HC or k for Top-K
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        
                        # Apply filters
                        if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                            l0_values.append(l0)
                            mse_values.append(mse)
                            run_names.append(run_data['run_name'])
                            
                            # Get appropriate label based on SAE type
                            if sae_type == 'topk':
                                param_labels.append(run_data.get('k_value', None))
                            else:
                                # For all other SAE types, use sparsity coefficient
                                param_labels.append(run_data.get('sparsity_coeff', None))
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
                           alpha=0.7, s=60, label=f'{labels[sae_type]} runs')
                
                # Add parameter labels for all points
                for i, (x, y, param) in enumerate(zip(l0_values, mse_values, param_labels)):
                    if param is not None:
                        # Format label for display
                        if sae_type == 'topk':
                            label = f'k={param}'
                        else:
                            # For sparsity coefficients
                            if param >= 0.01:
                                label = f'{param:.2f}'
                            elif param >= 0.001:
                                label = f'{param:.3f}'
                            else:
                                label = f'{param:.0e}'
                        ax1.annotate(label, (x, y), 
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=6, alpha=0.6, color=colors[sae_type])
                
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
                            label=f'{labels[sae_type]} Pareto')
                    ax1.scatter(pareto_l0, pareto_mse,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=120, edgecolors='black', linewidth=2, zorder=5)
        
        ax1.set_xlabel('L0 Sparsity', fontsize=11)
        ax1.set_ylabel('MSE', fontsize=11)
        ax1.set_title('MSE vs L0 Sparsity\n(Lower is better for both)', fontsize=12)
        if use_log_scale:
            ax1.set_xscale('log')
            ax1.set_yscale('log')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Explained Variance vs L0 (minimize L0, maximize explained variance)
        ax2 = axes[1]
        for sae_type in ['relu', 'gated', 'hardconcrete', 'topk']:
            if data.get(sae_type):
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                ev_values = []
                param_labels = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        ev = run_data['layers'][layer_name]['explained_variance']
                        
                        # Apply filters
                        if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                            l0_values.append(l0)
                            ev_values.append(ev)
                            
                            # Get appropriate label based on SAE type
                            if sae_type == 'topk':
                                param_labels.append(run_data.get('k_value', None))
                            else:
                                # For all other SAE types, use sparsity coefficient
                                param_labels.append(run_data.get('sparsity_coeff', None))
                
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
                           alpha=0.7, s=60, label=f'{labels[sae_type]} runs')
                
                # Add parameter labels for all points
                for i, (x, y, param) in enumerate(zip(l0_values, ev_values, param_labels)):
                    if param is not None:
                        # Format label for display
                        if sae_type == 'topk':
                            label = f'k={param}'
                        else:
                            # For sparsity coefficients
                            if param >= 0.01:
                                label = f'{param:.2f}'
                            elif param >= 0.001:
                                label = f'{param:.3f}'
                            else:
                                label = f'{param:.0e}'
                        ax2.annotate(label, (x, y), 
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=6, alpha=0.6, color=colors[sae_type])
                
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
                            label=f'{labels[sae_type]} Pareto')
                    ax2.scatter(pareto_l0, pareto_ev,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=120, edgecolors='black', linewidth=2, zorder=5)
        
        ax2.set_xlabel('L0 Sparsity', fontsize=11)
        ax2.set_ylabel('Explained Variance', fontsize=11)
        ax2.set_title('Explained Variance vs L0 Sparsity\n(Lower L0 better, Higher EV better)', fontsize=12)
        if use_log_scale:
            ax2.set_xscale('log')
            # Don't use log scale for explained variance as it goes from 0 to 1
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Alive Dictionary Elements vs L0
        ax3 = axes[2]
        for sae_type in ['relu', 'gated', 'hardconcrete', 'topk']:
            if data.get(sae_type):
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                alive_values = []
                param_labels = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        alive = run_data['layers'][layer_name]['alive_dict_components']
                        
                        # Apply filters
                        if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                            l0_values.append(l0)
                            alive_values.append(alive)
                            
                            # Get appropriate label based on SAE type
                            if sae_type == 'topk':
                                param_labels.append(run_data.get('k_value', None))
                            else:
                                # For all other SAE types, use sparsity coefficient
                                param_labels.append(run_data.get('sparsity_coeff', None))
                
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
                           alpha=0.7, s=60, label=f'{labels[sae_type]} runs')
                
                # Add parameter labels for all points
                for i, (x, y, param) in enumerate(zip(l0_values, alive_values, param_labels)):
                    if param is not None:
                        # Format label for display
                        if sae_type == 'topk':
                            label = f'k={param}'
                        else:
                            # For sparsity coefficients
                            if param >= 0.01:
                                label = f'{param:.2f}'
                            elif param >= 0.001:
                                label = f'{param:.3f}'
                            else:
                                label = f'{param:.0e}'
                        ax3.annotate(label, (x, y), 
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=6, alpha=0.6, color=colors[sae_type])
                
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
                            label=f'{labels[sae_type]} Pareto')
                    ax3.scatter(pareto_l0, pareto_alive,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=120, edgecolors='black', linewidth=2, zorder=5)
        
        ax3.set_xlabel('L0 Sparsity', fontsize=11)
        ax3.set_ylabel('Alive Dictionary Components', fontsize=11)
        ax3.set_title('Alive Dictionary Components vs L0\n(Lower L0 better, Higher alive better)', fontsize=12)
        if use_log_scale:
            ax3.set_xscale('log')
            ax3.set_yscale('log')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3, which='both')
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        if max_mse == float('inf') and max_l0 == float('inf'):
            filter_str = 'No filtering'
        else:
            filter_str = f'MSE ∈ [{min_mse}, {max_mse}], L0 ∈ [{min_l0}, {max_l0}]'
        scale_str = ' (log scale)' if use_log_scale else ''
        plt.suptitle(f'Pareto Curves: All SAE Types - Layer {layer_name}{scale_str}\n({filter_str})', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"all_saes_pareto_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved plot to: {output_path}")
        if layer_filtered > 0:
            print(f"  Filtered {layer_filtered} points from this layer")
        
        # Also save as SVG for vector graphics
        output_path_svg = output_dir / f"all_saes_pareto_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()
        
        total_filtered += layer_filtered
    
    if total_filtered > 0:
        print(f"\nTotal points filtered: {total_filtered}")
        for sae_type, count in filtered_by_type.items():
            if count > 0:
                print(f"  {sae_type}: {count} points")


def print_pareto_summary(data: Dict[str, List[Dict]], layers: List[str],
                        max_mse: float = float('inf'), max_l0: float = float('inf'),
                        min_mse: float = 0.0, min_l0: float = 0.0):
    """Print a summary of the Pareto-optimal points for each layer and SAE type."""
    
    print("\n" + "=" * 80)
    if max_mse == float('inf') and max_l0 == float('inf'):
        filter_str = 'No filtering'
    else:
        filter_str = f'MSE ∈ [{min_mse}, {max_mse}], L0 ∈ [{min_l0}, {max_l0}]'
    print(f"PARETO FRONTIER SUMMARY - ALL SAE TYPES ({filter_str})")
    print("=" * 80)
    
    # Define display names
    display_names = {
        'relu': 'ReLU',
        'gated': 'Gated',
        'hardconcrete': 'HardConcrete',
        'topk': 'Top-K'
    }
    
    for layer_name in layers:
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*80}")
        
        for sae_type in ['relu', 'gated', 'hardconcrete', 'topk']:
            if not data.get(sae_type):
                continue
            
            print(f"\n{display_names[sae_type]} SAE:")
            print("-" * 40)
            
            # Extract layer-specific data WITH FILTERING
            runs_data = []
            filtered_count = 0
            for run_data in data[sae_type]:
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    mse = run_data['layers'][layer_name]['mse']
                    
                    if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
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
                if max_mse != float('inf') or max_l0 != float('inf'):
                    print(f"  No data available after filtering ({filtered_count} runs filtered)")
                else:
                    print(f"  No data available")
                continue
            elif filtered_count > 0 and (max_mse != float('inf') or max_l0 != float('inf')):
                print(f"  ({filtered_count} runs filtered out)")
            
            # Calculate statistics
            l0_values = np.array([r['l0'] for r in runs_data])
            mse_values = np.array([r['mse'] for r in runs_data])
            ev_values = np.array([r['ev'] for r in runs_data])
            
            print(f"  Statistics ({len(runs_data)} runs):")
            print(f"    L0: {l0_values.mean():.2f} ± {l0_values.std():.2f}")
            print(f"    MSE: {mse_values.mean():.6f} ± {mse_values.std():.6f}")
            print(f"    EV: {ev_values.mean():.4f} ± {ev_values.std():.4f}")
            
            # Find Pareto points
            is_pareto_mse = find_pareto_frontier(l0_values, mse_values, 
                                                minimize_x=True, minimize_y=True)
            is_pareto_ev = find_pareto_frontier(l0_values, -ev_values,
                                               minimize_x=True, minimize_y=True)
            
            # Count Pareto optimal runs
            n_pareto = np.sum(is_pareto_mse | is_pareto_ev)
            if n_pareto > 0:
                print(f"  Pareto-optimal runs: {n_pareto}/{len(runs_data)}")
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves for all SAE types including HardConcrete variants"
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
        default=0.001,
        help="Maximum MSE threshold for filtering (default: 0.001)"
    )
    parser.add_argument(
        "--max-l0",
        type=float,
        default=100,
        help="Maximum L0 threshold for filtering (default: 100)"
    )
    parser.add_argument(
        "--min-mse",
        type=float,
        default=0.0,
        help="Minimum MSE threshold for filtering (default: 0.0)"
    )
    parser.add_argument(
        "--min-l0",
        type=float,
        default=0.0,
        help="Minimum L0 threshold for filtering (default: 0.0)"
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for axes (default: False)"
    )
    
    args = parser.parse_args()
    
    # Handle log scale flag - simple now
    use_log_scale = args.log_scale
    
    # Collect data
    print("=" * 80)
    print("Collecting data for all SAE types...")
    print("=" * 80)
    data, layers = collect_all_metrics_data(args.project)
    
    # Create plots with filtering
    print("\n" + "=" * 80)
    print("Creating Pareto curve plots...")
    print("=" * 80)
    plot_all_pareto_curves(data, layers, Path(args.output_dir), 
                          max_mse=args.max_mse, max_l0=args.max_l0,
                          min_mse=args.min_mse, min_l0=args.min_l0,
                          use_log_scale=use_log_scale)
    
    # Print summary with filtering
    print_pareto_summary(data, layers, max_mse=args.max_mse, max_l0=args.max_l0,
                        min_mse=args.min_mse, min_l0=args.min_l0)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 