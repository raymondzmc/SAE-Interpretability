#!/usr/bin/env python3
"""
Plot Pareto curves with flexible filtering by SAE methods.
Supports filtering to show only specific SAE types (e.g., only HardConcrete runs).
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Set
import json
import argparse
import tempfile
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from settings import settings
from utils.io import load_metrics_from_wandb
from models import SAETransformer

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9


def _detect_hc_gate_type_from_checkpoint(run: wandb.apis.public.Run) -> Optional[str]:
    """Detect HardConcrete gate type by inspecting the checkpoint state dict.
    Returns 'dependent' or 'independent', or None if undetectable.
    """
    try:
        # Find checkpoint files
        checkpoints = [f for f in run.files() if f.name.endswith('.pt')]
        if not checkpoints:
            return None
        # Pick the latest by numeric suffix if present, else last
        try:
            latest = sorted(checkpoints, key=lambda x: int(x.name.split('.pt')[0].split('_')[-1]))[-1]
        except Exception:
            latest = checkpoints[-1]
        # Download to a temporary directory
        with tempfile.TemporaryDirectory() as td:
            local_path = latest.download(exist_ok=True, replace=True, root=td).name
            state = torch.load(local_path, map_location='cpu')
        # Heuristic: presence of 'gate_logits' params indicates input-independent gates
        has_gate_logits = any('gate_logits' in k for k in state.keys())
        return 'independent' if has_gate_logits else 'dependent'
    except Exception:
        return None


def collect_metrics_data(project: str = "raymondl/tinystories-1m", 
                         method_filter: Optional[Set[str]] = None) -> Tuple[Dict[str, List[Dict]], List[str]]:
    """
    Collect metrics data for specified SAE types.
    
    Args:
        project: Wandb project name
        method_filter: Set of SAE types to include. If None, includes all.
                      Options: 'relu', 'gated', 'hardconcrete', 'topk'
    
    Returns:
        Dictionary with SAE type data and list of layer names
    """
    print("Collecting metrics data from Wandb...")
    if method_filter:
        print(f"Filtering for methods: {', '.join(method_filter)}")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs
    runs = list(api.runs(project))
    
    # Collect data by SAE type
    data = {
        'relu': [],
        'gated': [],
        'hardconcrete': [],
        'topk': []
    }
    
    # Track layer names
    all_layers = set()
    
    for run in runs:
        name_lower = run.name.lower()
        
        # Determine SAE type
        sae_type = None
        if 'relu' in name_lower:
            sae_type = 'relu'
        elif 'gated' in name_lower and 'hard' not in name_lower:
            sae_type = 'gated'
        elif 'hard' in name_lower or 'hardconcrete' in name_lower:
            sae_type = 'hardconcrete'
        elif 'topk' in name_lower or 'top-k' in name_lower:
            sae_type = 'topk'
        else:
            continue  # Skip unknown types
        
        # Apply method filter if specified
        if method_filter and sae_type not in method_filter:
            continue
            
        # Load metrics for this run
        print(f"  Loading metrics for {run.name} ({run.id}) - Type: {sae_type}")
        metrics = load_metrics_from_wandb(run.id, project)
        
        if metrics:
            # Extract sparsity coefficient from run name
            sparsity_coeff = None
            if 'sparsity_coeff_' in run.name:
                coeff_str = run.name.split('sparsity_coeff_')[1].split('_')[0]
                try:
                    sparsity_coeff = float(coeff_str.replace('e-', 'e-'))
                except:
                    sparsity_coeff = None
            elif 'hard_concrete_' in run.name:
                # Extract from hard_concrete_X pattern
                parts = run.name.split('hard_concrete_')[1].split('_')
                if parts[0].replace('.', '').replace('e-', '').replace('e+', '').replace('-', '').isdigit():
                    try:
                        sparsity_coeff = float(parts[0])
                    except:
                        sparsity_coeff = None
            
            # Determine gate type accurately for HardConcrete
            gate_type = None
            if sae_type == 'hardconcrete':
                # Try config first
                cfg = run.config if hasattr(run, 'config') else None
                if cfg and isinstance(cfg, dict):
                    saes_cfg = cfg.get('saes') or cfg.get('saes', None)
                    if saes_cfg and isinstance(saes_cfg, dict) and 'input_dependent_gates' in saes_cfg:
                        gate_type = 'dependent' if saes_cfg['input_dependent_gates'] else 'independent'
                # Fallback to checkpoint inspection
                if gate_type is None:
                    gate_type = _detect_hc_gate_type_from_checkpoint(run)
                # Final fallback to name heuristic
                if gate_type is None:
                    gate_type = 'independent' if ('no_gates' in name_lower or 'indep' in name_lower) else 'dependent'
            
            # Store per-layer metrics
            run_data = {
                'run_name': run.name,
                'run_id': run.id,
                'sparsity_coeff': sparsity_coeff,
                'gate_type': gate_type,
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
    
    # Remove empty SAE types
    data = {k: v for k, v in data.items() if v}
    
    # Sort by average L0 sparsity for consistency
    for sae_type in data:
        data[sae_type] = sorted(data[sae_type], 
                                key=lambda x: np.mean([m['l0'] for m in x['layers'].values()]))
    
    print(f"\nCollected data summary:")
    for sae_type, runs in data.items():
        if sae_type == 'hardconcrete':
            dep_count = sum(1 for r in runs if r['gate_type'] == 'dependent')
            indep_count = sum(1 for r in runs if r['gate_type'] == 'independent')
            print(f"  {sae_type}: {len(runs)} runs (dependent: {dep_count}, independent: {indep_count})")
        else:
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


def plot_pareto_curves_per_layer(data: Dict[str, List[Dict]], layers: List[str], 
                                 output_dir: Path = Path("plots"),
                                 max_mse: float = 0.05, max_l0: float = 250,
                                 show_labels: bool = True,
                                 output_suffix: str = ""):
    """
    Create Pareto curve plots for each layer separately.
    
    Args:
        data: Dictionary with SAE type data
        layers: List of layer names
        output_dir: Output directory for plots
        max_mse: Maximum MSE threshold
        max_l0: Maximum L0 threshold
        show_labels: Whether to show sparsity coefficient labels
        output_suffix: Suffix to add to output filenames
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nFiltering: Removing runs with MSE > {max_mse} or L0 > {max_l0}")
    
    # Color scheme - expanded for more SAE types
    colors = {
        'relu': '#2E86AB',  # Blue
        'gated': '#A23B72',  # Purple
        'hardconcrete': '#FF6B35',  # Orange
        'topk': '#4ECDC4'  # Teal
    }
    
    # Marker styles
    markers = {
        'relu': 'o',
        'gated': 's',
        'hardconcrete': '^',
        'topk': 'D'
    }
    
    # Labels for display
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
        
        # Create figure with 3 subplots for this layer
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: MSE vs L0 (minimize both)
        ax1 = axes[0]
        for sae_type in data.keys():
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                mse_values = []
                run_names = []
                gate_types = []
                coeff_values = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            mse_values.append(mse)
                            run_names.append(run_data['run_name'])
                            gate_types.append(run_data.get('gate_type'))
                            coeff_values.append(run_data.get('sparsity_coeff'))
                        else:
                            layer_filtered += 1
                            filtered_by_type[sae_type] += 1
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                mse_values = np.array(mse_values)
                
                # For HardConcrete, separate by gate type and compute Pareto per subset
                if sae_type == 'hardconcrete' and any(gt is not None for gt in gate_types):
                    dep_mask = np.array([g == 'dependent' for g in gate_types])
                    indep_mask = np.array([g == 'independent' for g in gate_types])
                    
                    if np.any(dep_mask):
                        dep_l0 = l0_values[dep_mask]
                        dep_mse = mse_values[dep_mask]
                        dep_coeffs = np.array(coeff_values, dtype=object)[dep_mask]
                        ax1.scatter(dep_l0, dep_mse, 
                                   color=colors[sae_type], marker=markers[sae_type],
                                   alpha=0.6, s=40, label=f'{labels[sae_type]} (dep)')
                        # Labels
                        if show_labels:
                            for x, y, coeff in zip(dep_l0, dep_mse, dep_coeffs):
                                if coeff is None:
                                    continue
                                label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                                ax1.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                             fontsize=7, alpha=0.75, color=colors[sae_type])
                        is_pareto_dep = find_pareto_frontier(dep_l0, dep_mse, minimize_x=True, minimize_y=True)
                        if np.any(is_pareto_dep):
                            pareto_l0 = dep_l0[is_pareto_dep]
                            pareto_mse = dep_mse[is_pareto_dep]
                            sort_idx = np.argsort(pareto_l0)
                            ax1.plot(pareto_l0[sort_idx], pareto_mse[sort_idx], 
                                    color=colors[sae_type], linewidth=2, alpha=0.8,
                                    label=f'{labels[sae_type]} (dep) Pareto')
                            ax1.scatter(pareto_l0[sort_idx], pareto_mse[sort_idx],
                                       color=colors[sae_type], marker=markers[sae_type],
                                       s=80, edgecolors='black', linewidth=1.5, zorder=5)
                    if np.any(indep_mask):
                        indep_l0 = l0_values[indep_mask]
                        indep_mse = mse_values[indep_mask]
                        indep_coeffs = np.array(coeff_values, dtype=object)[indep_mask]
                        ax1.scatter(indep_l0, indep_mse, 
                                   color=colors[sae_type], marker=markers[sae_type],
                                   alpha=0.6, s=40, facecolors='none', 
                                   edgecolors=colors[sae_type], linewidth=1.5,
                                   label=f'{labels[sae_type]} (indep)')
                        # Labels
                        if show_labels:
                            for x, y, coeff in zip(indep_l0, indep_mse, indep_coeffs):
                                if coeff is None:
                                    continue
                                label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                                ax1.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                             fontsize=7, alpha=0.75, color=colors[sae_type])
                        is_pareto_indep = find_pareto_frontier(indep_l0, indep_mse, minimize_x=True, minimize_y=True)
                        if np.any(is_pareto_indep):
                            pareto_l0 = indep_l0[is_pareto_indep]
                            pareto_mse = indep_mse[is_pareto_indep]
                            sort_idx = np.argsort(pareto_l0)
                            ax1.plot(pareto_l0[sort_idx], pareto_mse[sort_idx], 
                                    color=colors[sae_type], linewidth=2, alpha=0.8,
                                    linestyle='--',
                                    label=f'{labels[sae_type]} (indep) Pareto')
                            ax1.scatter(pareto_l0[sort_idx], pareto_mse[sort_idx],
                                       color=colors[sae_type], marker=markers[sae_type],
                                       s=80, edgecolors='black', linewidth=1.5, zorder=5, facecolors='none')
                else:
                    ax1.scatter(l0_values, mse_values, 
                               color=colors[sae_type], marker=markers[sae_type],
                               alpha=0.6, s=40, label=f'{labels[sae_type]} runs')
                    # Labels
                    if show_labels:
                        for x, y, coeff in zip(l0_values, mse_values, coeff_values):
                            if coeff is None:
                                continue
                            label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                            ax1.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                         fontsize=7, alpha=0.75, color=colors[sae_type])
                    # Find and plot Pareto frontier (single set)
                    is_pareto = find_pareto_frontier(l0_values, mse_values, minimize_x=True, minimize_y=True)
                    if np.any(is_pareto):
                        pareto_l0 = l0_values[is_pareto]
                        pareto_mse = mse_values[is_pareto]
                        sort_idx = np.argsort(pareto_l0)
                        ax1.plot(pareto_l0[sort_idx], pareto_mse[sort_idx], 
                                color=colors[sae_type], linewidth=2, alpha=0.8,
                                label=f'{labels[sae_type]} Pareto')
                        ax1.scatter(pareto_l0[sort_idx], pareto_mse[sort_idx],
                                   color=colors[sae_type], marker=markers[sae_type],
                                   s=80, edgecolors='black', linewidth=1.5, zorder=5)
        
        ax1.set_xlabel('L0 Sparsity', fontsize=11)
        ax1.set_ylabel('MSE', fontsize=11)
        ax1.set_title('MSE vs L0 Sparsity\n(Lower is better for both)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Explained Variance vs L0
        ax2 = axes[1]
        for sae_type in data.keys():
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                ev_values = []
                gate_types = []
                coeff_values = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        ev = run_data['layers'][layer_name]['explained_variance']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            ev_values.append(ev)
                            gate_types.append(run_data.get('gate_type'))
                            coeff_values.append(run_data.get('sparsity_coeff'))
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                ev_values = np.array(ev_values)
                
                # For HardConcrete, separate by gate type and compute Pareto per subset
                if sae_type == 'hardconcrete' and any(gt is not None for gt in gate_types):
                    dep_mask = np.array([g == 'dependent' for g in gate_types])
                    indep_mask = np.array([g == 'independent' for g in gate_types])
                    
                    if np.any(dep_mask):
                        dep_l0 = l0_values[dep_mask]
                        dep_ev = ev_values[dep_mask]
                        dep_coeffs = np.array(coeff_values, dtype=object)[dep_mask]
                        ax2.scatter(dep_l0, dep_ev,
                                   color=colors[sae_type], marker=markers[sae_type],
                                   alpha=0.6, s=40, label=f'{labels[sae_type]} (dep)')
                        # Labels
                        if show_labels:
                            for x, y, coeff in zip(dep_l0, dep_ev, dep_coeffs):
                                if coeff is None:
                                    continue
                                label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                                ax2.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                             fontsize=7, alpha=0.75, color=colors[sae_type])
                        is_pareto_dep = find_pareto_frontier(dep_l0, -dep_ev, minimize_x=True, minimize_y=True)
                        if np.any(is_pareto_dep):
                            pareto_l0 = dep_l0[is_pareto_dep]
                            pareto_ev = dep_ev[is_pareto_dep]
                            sort_idx = np.argsort(pareto_l0)
                            ax2.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                    color=colors[sae_type], linewidth=2, alpha=0.8,
                                    label=f'{labels[sae_type]} (dep) Pareto')
                            ax2.scatter(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                       color=colors[sae_type], marker=markers[sae_type],
                                       s=80, edgecolors='black', linewidth=1.5, zorder=5)
                    if np.any(indep_mask):
                        indep_l0 = l0_values[indep_mask]
                        indep_ev = ev_values[indep_mask]
                        indep_coeffs = np.array(coeff_values, dtype=object)[indep_mask]
                        ax2.scatter(indep_l0, indep_ev,
                                   color=colors[sae_type], marker=markers[sae_type],
                                   alpha=0.6, s=40, facecolors='none',
                                   edgecolors=colors[sae_type], linewidth=1.5,
                                   label=f'{labels[sae_type]} (indep)')
                        # Labels
                        if show_labels:
                            for x, y, coeff in zip(indep_l0, indep_ev, indep_coeffs):
                                if coeff is None:
                                    continue
                                label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                                ax2.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                             fontsize=7, alpha=0.75, color=colors[sae_type])
                        is_pareto_indep = find_pareto_frontier(indep_l0, -indep_ev, minimize_x=True, minimize_y=True)
                        if np.any(is_pareto_indep):
                            pareto_l0 = indep_l0[is_pareto_indep]
                            pareto_ev = indep_ev[is_pareto_indep]
                            sort_idx = np.argsort(pareto_l0)
                            ax2.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                    color=colors[sae_type], linewidth=2, alpha=0.8,
                                    linestyle='--',
                                    label=f'{labels[sae_type]} (indep) Pareto')
                            ax2.scatter(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                       color=colors[sae_type], marker=markers[sae_type],
                                       s=80, edgecolors='black', linewidth=1.5, zorder=5, facecolors='none')
                else:
                    ax2.scatter(l0_values, ev_values,
                               color=colors[sae_type], marker=markers[sae_type],
                               alpha=0.6, s=40, label=f'{labels[sae_type]} runs')
                    # Labels
                    if show_labels:
                        for x, y, coeff in zip(l0_values, ev_values, coeff_values):
                            if coeff is None:
                                continue
                            label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                            ax2.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                         fontsize=7, alpha=0.75, color=colors[sae_type])
                    # Find and plot Pareto frontier (single set)
                    is_pareto = find_pareto_frontier(l0_values, -ev_values, minimize_x=True, minimize_y=True)
                    if np.any(is_pareto):
                        pareto_l0 = l0_values[is_pareto]
                        pareto_ev = ev_values[is_pareto]
                        sort_idx = np.argsort(pareto_l0)
                        ax2.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                color=colors[sae_type], linewidth=2, alpha=0.8,
                                label=f'{labels[sae_type]} Pareto')
                        ax2.scatter(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                   color=colors[sae_type], marker=markers[sae_type],
                                   s=80, edgecolors='black', linewidth=1.5, zorder=5)
        
        ax2.set_xlabel('L0 Sparsity', fontsize=11)
        ax2.set_ylabel('Explained Variance', fontsize=11)
        ax2.set_title('Explained Variance vs L0 Sparsity\n(Lower L0 better, Higher EV better)', fontsize=12)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Alive Dictionary Elements vs L0
        ax3 = axes[2]
        for sae_type in data.keys():
            if data[sae_type]:
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                alive_values = []
                gate_types = []
                coeff_values = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        alive = run_data['layers'][layer_name]['alive_dict_components']
                        
                        # Apply filters
                        if l0 <= max_l0 and mse <= max_mse:
                            l0_values.append(l0)
                            alive_values.append(alive)
                            gate_types.append(run_data.get('gate_type'))
                            coeff_values.append(run_data.get('sparsity_coeff'))
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                alive_values = np.array(alive_values)
                
                # For HardConcrete, separate by gate type and compute Pareto per subset
                if sae_type == 'hardconcrete' and any(gt is not None for gt in gate_types):
                    dep_mask = np.array([g == 'dependent' for g in gate_types])
                    indep_mask = np.array([g == 'independent' for g in gate_types])
                    
                    if np.any(dep_mask):
                        dep_l0 = l0_values[dep_mask]
                        dep_alive = alive_values[dep_mask]
                        dep_coeffs = np.array(coeff_values, dtype=object)[dep_mask]
                        ax3.scatter(dep_l0, dep_alive,
                                   color=colors[sae_type], marker=markers[sae_type],
                                   alpha=0.6, s=40, label=f'{labels[sae_type]} (dep)')
                        # Labels
                        if show_labels:
                            for x, y, coeff in zip(dep_l0, dep_alive, dep_coeffs):
                                if coeff is None:
                                    continue
                                label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                                ax3.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                             fontsize=7, alpha=0.75, color=colors[sae_type])
                        is_pareto_dep = find_pareto_frontier(dep_l0, -dep_alive, minimize_x=True, minimize_y=True)
                        if np.any(is_pareto_dep):
                            pareto_l0 = dep_l0[is_pareto_dep]
                            pareto_alive = dep_alive[is_pareto_dep]
                            sort_idx = np.argsort(pareto_l0)
                            ax3.plot(pareto_l0[sort_idx], pareto_alive[sort_idx],
                                    color=colors[sae_type], linewidth=2, alpha=0.8,
                                    label=f'{labels[sae_type]} (dep) Pareto')
                            ax3.scatter(pareto_l0[sort_idx], pareto_alive[sort_idx],
                                       color=colors[sae_type], marker=markers[sae_type],
                                       s=80, edgecolors='black', linewidth=1.5, zorder=5)
                    if np.any(indep_mask):
                        indep_l0 = l0_values[indep_mask]
                        indep_alive = alive_values[indep_mask]
                        indep_coeffs = np.array(coeff_values, dtype=object)[indep_mask]
                        ax3.scatter(indep_l0, indep_alive,
                                   color=colors[sae_type], marker=markers[sae_type],
                                   alpha=0.6, s=40, facecolors='none',
                                   edgecolors=colors[sae_type], linewidth=1.5,
                                   label=f'{labels[sae_type]} (indep)')
                        # Labels
                        if show_labels:
                            for x, y, coeff in zip(indep_l0, indep_alive, indep_coeffs):
                                if coeff is None:
                                    continue
                                label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                                ax3.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                             fontsize=7, alpha=0.75, color=colors[sae_type])
                        is_pareto_indep = find_pareto_frontier(indep_l0, -indep_alive, minimize_x=True, minimize_y=True)
                        if np.any(is_pareto_indep):
                            pareto_l0 = indep_l0[is_pareto_indep]
                            pareto_alive = indep_alive[is_pareto_indep]
                            sort_idx = np.argsort(pareto_l0)
                            ax3.plot(pareto_l0[sort_idx], pareto_alive[sort_idx],
                                    color=colors[sae_type], linewidth=2, alpha=0.8,
                                    linestyle='--',
                                    label=f'{labels[sae_type]} (indep) Pareto')
                            ax3.scatter(pareto_l0[sort_idx], pareto_alive[sort_idx],
                                       color=colors[sae_type], marker=markers[sae_type],
                                       s=80, edgecolors='black', linewidth=1.5, zorder=5, facecolors='none')
                else:
                    ax3.scatter(l0_values, alive_values,
                               color=colors[sae_type], marker=markers[sae_type],
                               alpha=0.6, s=40, label=f'{labels[sae_type]} runs')
                    # Labels
                    if show_labels:
                        for x, y, coeff in zip(l0_values, alive_values, coeff_values):
                            if coeff is None:
                                continue
                            label = f'{coeff:.2f}' if coeff >= 0.01 else f'{coeff:.0e}'
                            ax3.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                                         fontsize=7, alpha=0.75, color=colors[sae_type])
                    # Find and plot Pareto frontier (single set)
                    is_pareto = find_pareto_frontier(l0_values, -alive_values, minimize_x=True, minimize_y=True)
                    if np.any(is_pareto):
                        pareto_l0 = l0_values[is_pareto]
                        pareto_alive = alive_values[is_pareto]
                        sort_idx = np.argsort(pareto_l0)
                        ax3.plot(pareto_l0[sort_idx], pareto_alive[sort_idx],
                                color=colors[sae_type], linewidth=2, alpha=0.8,
                                label=f'{labels[sae_type]} Pareto')
                        ax3.scatter(pareto_l0[sort_idx], pareto_alive[sort_idx],
                                   color=colors[sae_type], marker=markers[sae_type],
                                   s=80, edgecolors='black', linewidth=1.5, zorder=5)
        
        ax3.set_xlabel('L0 Sparsity', fontsize=11)
        ax3.set_ylabel('Alive Dictionary Components', fontsize=11)
        ax3.set_title('Alive Dictionary Components vs L0\n(Lower L0 better, Higher alive better)', fontsize=12)
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        
        # Create title based on included methods
        method_names = [labels[k] for k in data.keys()]
        if len(method_names) == 1:
            method_str = method_names[0]
        else:
            method_str = ', '.join(method_names)
        
        plt.suptitle(f'Pareto Curves: {method_str} - Layer {layer_name}\n(Filtered: MSE ≤ {max_mse}, L0 ≤ {max_l0})', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"pareto_{layer_display_name}{output_suffix}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved plot to: {output_path}")
        print(f"  Filtered {layer_filtered} points from this layer")
        
        # Also save as SVG for vector graphics
        output_path_svg = output_dir / f"pareto_{layer_display_name}{output_suffix}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()
        
        total_filtered += layer_filtered
    
    print(f"\nTotal points filtered: {total_filtered}")
    for sae_type, count in filtered_by_type.items():
        if count > 0:
            print(f"  {sae_type}: {count} points")


def print_pareto_summary_per_layer(data: Dict[str, List[Dict]], layers: List[str],
                                   max_mse: float = 0.05, max_l0: float = 250):
    """Print a summary of the Pareto-optimal points for each layer."""
    
    print("\n" + "=" * 70)
    print(f"PARETO FRONTIER SUMMARY - PER LAYER (Filtered: MSE ≤ {max_mse}, L0 ≤ {max_l0})")
    print("=" * 70)
    
    labels = {
        'relu': 'ReLU',
        'gated': 'Gated',
        'hardconcrete': 'HardConcrete',
        'topk': 'Top-K'
    }
    
    for layer_name in layers:
        print(f"\n{'='*70}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*70}")
        
        for sae_type in data.keys():
            if not data[sae_type]:
                continue
            
            print(f"\n{labels[sae_type]} SAE:")
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
                            'alive': run_data['layers'][layer_name]['alive_dict_components'],
                            'gate_type': run_data.get('gate_type')
                        })
                    else:
                        filtered_count += 1
            
            if not runs_data:
                print(f"  No data available after filtering ({filtered_count} runs filtered)")
                continue
            elif filtered_count > 0:
                print(f"  ({filtered_count} runs filtered out)")
            
            # For HardConcrete, show gate type breakdown
            if sae_type == 'hardconcrete':
                dep_count = sum(1 for r in runs_data if r['gate_type'] == 'dependent')
                indep_count = sum(1 for r in runs_data if r['gate_type'] == 'independent')
                print(f"  Gate types: dependent={dep_count}, independent={indep_count}")
            
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
                        'on_ev_frontier': is_pareto_ev[i],
                        'gate_type': run.get('gate_type')
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
                    
                    gate_str = f" ({run['gate_type'][:3]})" if run['gate_type'] else ""
                    print(f"    • {run['name']:28} L0={run['l0']:6.2f}, MSE={run['mse']:.4f}, "
                          f"EV={run['ev']:.4f} [{frontier_str}]{gate_str}")
                
                if len(pareto_runs) > 5:
                    print(f"    ... and {len(pareto_runs) - 5} more")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves with flexible filtering by SAE methods"
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
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        choices=['relu', 'gated', 'hardconcrete', 'topk', 'all'],
        default=['all'],
        help="SAE methods to include in plots (default: all)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide sparsity coefficient labels on plots"
    )
    
    args = parser.parse_args()
    
    # Process method filter
    if 'all' in args.methods:
        method_filter = None
    else:
        method_filter = set(args.methods)
    
    # Collect data
    data, layers = collect_metrics_data(args.project, method_filter)
    
    if not data:
        print("No data found for specified methods!")
        return
    
    # Create output suffix based on methods
    if method_filter:
        output_suffix = f"_{'_'.join(sorted(method_filter))}"
    else:
        output_suffix = "_all"
    
    # Create plots with filtering
    plot_pareto_curves_per_layer(data, layers, Path(args.output_dir), 
                                 max_mse=args.max_mse, max_l0=args.max_l0,
                                 show_labels=not args.no_labels,
                                 output_suffix=output_suffix)
    
    # Print summary with filtering
    print_pareto_summary_per_layer(data, layers, max_mse=args.max_mse, max_l0=args.max_l0)


if __name__ == "__main__":
    main() 