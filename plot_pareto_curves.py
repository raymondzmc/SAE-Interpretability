import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import re


EXCLUDE_RUN_IDS = [
    'twoagsgt',
    'f4nvldll',
    'gjjl6n95',
    # 'zu4cqbsw',
    # '0jennxfn',
    # 'x5q9xinm',
    # # 'm159z1bk',
    # 'cqr1x92u',
    # # '3sokvaif',
    # 'g68gpv2r',
    # '953pems7',
    # 'g68gpv2r',
    # 'j3o82ivu',
    # '9n6f7ydq',
    # 'qoufqca1',
    # '51bgc9zd',
    # '5gq797mj',
    # 'z869pdb4',
    # 'gjjl6n95',
    # '15x8bwpp',
]

explained_variance_range = {
    2: (0.98, 1.002),
    4: (0.97, 1.002),
    6: (0.80, 1.002),
}

def format_sparsity_coeff(value: Any) -> str:
    """Format sparsity coefficient with appropriate precision."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    elif value >= 1.0:
        return f'{value:.1f}'  # e.g., "50.0", "3.0"
    elif value >= 0.01:
        return f'{value:.2f}'  # e.g., "0.05", "0.10"
    elif value >= 0.001:
        return f'{value:.3f}'  # e.g., "0.005", "0.001"
    else:
        return f'{value:.0e}'  # e.g., "5e-05", "1e-06"

def extract_multi_layer_metrics(run: Any) -> Dict[str, Any]:
    """Extract metrics for runs with SAEs at multiple layers."""
    run_data = {
        'name': run.name,
        'id': run.id,
        'state': run.state
    }
    
    # Only process finished runs
    if run.state != "finished":
        return run_data
    
    # Determine run type based on SAE type in config and other config parameters
    try:
        sae_type = run.config['saes']['sae_type']
        # Check for hard concrete without learned gates
        input_dependent_gates = run.config['saes'].get('input_dependent_gates', True)
        
        # Map SAE types to display names, distinguishing hard concrete variants
        if sae_type == 'hard_concrete' and not input_dependent_gates:
            run_data['method'] = 'HardConcrete (IIG)'
        else:
            sae_type_mapping = {
                'relu': 'ReLU',
                'hard_concrete': 'HardConcrete',
                'gated': 'Gated',
                'gated_hard_concrete': 'GatedHardConcrete'
            }
            run_data['method'] = sae_type_mapping.get(sae_type, sae_type.title())
    except (KeyError, TypeError):
        # Try to infer from run name or tags
        if 'relu' in run.name.lower() or (run.config.get('wandb_tags') and 'relu' in run.config['wandb_tags']):
            run_data['method'] = 'ReLU'
        elif 'no_learned_gates' in run.name.lower() or (run.config.get('wandb_tags') and 'no_learned_gates' in run.config['wandb_tags']):
            run_data['method'] = 'HardConcrete (IIG)'
        elif 'hard' in run.name.lower() and 'concrete' in run.name.lower():
            run_data['method'] = 'HardConcrete'
        elif 'gated' in run.name.lower():
            if 'hardconcrete' in run.name.lower() or 'hard_concrete' in run.name.lower():
                run_data['method'] = 'GatedHardConcrete'
            else:
                run_data['method'] = 'Gated'
        else:
            run_data['method'] = 'Unknown'
    
    # Extract sparsity coefficient from config
    try:
        sparsity_coeff = run.config['saes']['sparsity_coeff']
        run_data['sparsity_coeff'] = sparsity_coeff
    except (KeyError, TypeError):
        run_data['sparsity_coeff'] = None
    
    # Extract layers from metrics - new metric naming pattern
    layers = set()
    sparsity_metrics = {}
    recon_loss_metrics = {}
    explained_var_metrics = {}
    explained_var_ln_metrics = {}
    
    # All debugging complete - ready for production use
    
    for key, value in run.summary_metrics.items():
        # Look for sparsity metrics: both eval and train
        sparsity_match = re.search(r'(eval|train)/L_0/blocks\.(\d+)\.hook_resid_pre', key)
        if sparsity_match:
            layer = int(sparsity_match.group(2))
            layers.add(layer)
            # Prefer eval metrics, but include train if eval not available
            if layer not in sparsity_metrics or 'eval' in key:
                sparsity_metrics[layer] = value
        
        # Look for reconstruction loss: both eval and train
        recon_match = re.search(r'(eval|train)/loss/blocks\.(\d+)\.hook_resid_pre/mse_loss', key)
        if recon_match:
            layer = int(recon_match.group(2))
            layers.add(layer)
            # Prefer eval metrics, but include train if eval not available
            if layer not in recon_loss_metrics or 'eval' in key:
                recon_loss_metrics[layer] = value
        
        # Look for explained variance: both eval and train
        explained_var_match = re.search(r'(eval|train)/explained_variance/blocks\.(\d+)\.hook_resid_pre', key)
        if explained_var_match:
            layer = int(explained_var_match.group(2))
            layers.add(layer)
            # Prefer eval metrics, but include train if eval not available
            if layer not in explained_var_metrics or 'eval' in key:
                explained_var_metrics[layer] = value
        
        # Look for layer-normalized explained variance: both eval and train
        explained_var_ln_match = re.search(r'(eval|train)/explained_variance_ln/blocks\.(\d+)\.hook_resid_pre', key)
        if explained_var_ln_match:
            layer = int(explained_var_ln_match.group(2))
            layers.add(layer)
            # Prefer eval metrics, but include train if eval not available
            if layer not in explained_var_ln_metrics or 'eval' in key:
                explained_var_ln_metrics[layer] = value
    
    # Store per-layer metrics
    for layer in sorted(layers):
        run_data[f'sparsity_layer_{layer}'] = sparsity_metrics.get(layer, np.nan)
        run_data[f'recon_loss_layer_{layer}'] = recon_loss_metrics.get(layer, np.nan)
        run_data[f'explained_var_layer_{layer}'] = explained_var_metrics.get(layer, np.nan)
        run_data[f'explained_var_ln_layer_{layer}'] = explained_var_ln_metrics.get(layer, np.nan)
    
    run_data['layers'] = sorted(layers)
    
    return run_data

def create_pareto_plot(df: pd.DataFrame, layers: List[int], save_path: str = None, metric_type: str = 'mse'):
    """Create a pareto frontier plot for sparsity vs reconstruction loss or explained variance."""
    
    # Set up the plot
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    # Updated colors and markers for all 5 SAE types
    colors = {
        'ReLU': '#d62728',
        'HardConcrete': '#2ca02c', 
        'HardConcrete (IIG)': '#8c564b',
        'Gated': '#ff7f0e',
        'GatedHardConcrete': '#9467bd'
    }
    markers = {
        'ReLU': 'o',
        'HardConcrete': 's', 
        'HardConcrete (IIG)': 'v',
        'Gated': '^',
        'GatedHardConcrete': 'D'
    }
    
    for i, layer in enumerate(layers):
        ax = axes[i]
        
        # Get data for this layer
        sparsity_col = f'sparsity_layer_{layer}'
        if metric_type == 'mse':
            y_col = f'recon_loss_layer_{layer}'
            y_label = 'Reconstruction Loss (MSE)'
        elif metric_type == 'explained_var':
            y_col = f'explained_var_layer_{layer}'
            y_label = 'Explained Variance'
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
        # Filter out runs that don't have this layer
        layer_df = df.dropna(subset=[sparsity_col, y_col])
        
        if layer_df.empty:
            ax.set_title(f'Layer {layer} (No data)')
            continue
        
        # Plot each method
        for method in ['ReLU', 'HardConcrete', 'HardConcrete (IIG)', 'Gated', 'GatedHardConcrete']:
            method_df = layer_df[layer_df['method'] == method]
            if not method_df.empty:
                # Filter points that are within the plot range for explained variance
                if metric_type == 'explained_var':
                    y_min, y_max = explained_variance_range.get(layer, (0.98, 1.002))
                    visible_points = method_df[
                        (method_df[y_col] >= y_min) & (method_df[y_col] <= 1.0)
                    ]
                else:
                    visible_points = method_df
                
                # Sort by sparsity for line connection (only visible points)
                if not visible_points.empty:
                    visible_points_sorted = visible_points.sort_values(sparsity_col)
                    
                    # Plot line first (so it's behind the points) - only for visible points
                    ax.plot(
                        visible_points_sorted[sparsity_col], 
                        visible_points_sorted[y_col],
                        c=colors[method],
                        alpha=0.6,
                        linewidth=1.5,
                        zorder=1
                    )
                
                # Plot scatter points on top (all points)
                ax.scatter(
                    method_df[sparsity_col], 
                    method_df[y_col],
                    c=colors[method], 
                    marker=markers[method],
                    s=60,
                    alpha=0.7,
                    label=method,
                    edgecolors='black',
                    linewidth=0.5,
                    zorder=2
                )
                
                # Add sparsity coefficient annotations
                for idx, row in method_df.iterrows():
                    if pd.notna(row['sparsity_coeff']):
                        ax.annotate(
                            f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                            (row[sparsity_col], row[y_col]),
                            xytext=(5, 5),  # offset from point
                            textcoords='offset points',
                            fontsize=7,
                            color=colors[method],
                            alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                            zorder=3
                        )
        
        ax.set_xlabel('Sparsity (L0)')
        ax.set_ylabel(y_label)
        ax.set_title(f'Layer {layer}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set log scale for MSE but not for explained variance
        if metric_type == 'mse':
            ax.set_yscale('log')
        elif metric_type == 'explained_var':
            y_min, y_max = explained_variance_range.get(layer, (0.98, 1.002))
            ax.set_ylim(y_min, y_max)
            # Create exactly 5 ticks from y_min to 1.0
            ticks = np.linspace(y_min, 1.0, 5)
            ax.set_yticks(ticks)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig

def main():
    # Try to import settings from the codebase
    try:
        from settings import settings
        
        # Setup wandb API using codebase settings
        if not settings.has_wandb_config():
            print("Error: Wandb configuration not found in settings.")
            print("Please ensure WANDB_API_KEY and WANDB_ENTITY are set in your .env file or environment.")
            return
        
        # Login to wandb using settings
        wandb.login(key=settings.wandb_api_key)
        api = wandb.Api()
        
        # Use tinystories-1m project as specified
        sweep_project = f"{settings.wandb_entity}/tinystories-1m"
        
    except ImportError as e:
        print(f"Could not import settings: {e}")
        print("Falling back to manual configuration...")
        
        # Fallback: use wandb API directly (requires manual login)
        api = wandb.Api()  # This will use your wandb login from command line
        
        # Manual project specification - you may need to update this
        entity = input("Please enter your wandb entity: ").strip()
        if not entity:
            print("No entity provided, exiting.")
            return
        sweep_project = f"{entity}/tinystories-1m"
    
    try:
        runs = api.runs(sweep_project)
        print(f"Total runs: {len(runs)}")
    except Exception as e:
        print(f"Error accessing wandb project {sweep_project}: {e}")
        print("Please check that the project exists and you have access to it.")
        return

    # Extract data from all runs
    print("Extracting metrics from runs...")
    run_data_list = []
    for run in runs:
        run_data = extract_multi_layer_metrics(run)
        run_data_list.append(run_data)

    # Create DataFrame
    df = pd.DataFrame(run_data_list)

    # Exclude specific runs
    if EXCLUDE_RUN_IDS:
        df_before_exclude = len(df)
        df = df[~df['id'].isin(EXCLUDE_RUN_IDS)]
        excluded_count = df_before_exclude - len(df)
        print(f"Excluded {excluded_count} runs with IDs: {EXCLUDE_RUN_IDS}")

    # Filter to only finished runs
    df = df[df['state'] == 'finished']
    print(f"Finished runs: {len(df)}")
    
    # Filter to only include the 5 requested SAE types
    requested_methods = ['ReLU', 'HardConcrete', 'HardConcrete (IIG)', 'Gated', 'GatedHardConcrete']
    df = df[df['method'].isin(requested_methods)]
    print(f"Runs with requested SAE types (ReLU, HardConcrete, HardConcrete (IIG), Gated, GatedHardConcrete): {len(df)}")

    if df.empty:
        print("No finished runs found!")
        return

    # Print summary
    print("\n=== METHOD SUMMARY ===")
    print(df['method'].value_counts())

    # Find available layers
    all_layers = set()
    for _, row in df.iterrows():
        if 'layers' in row and isinstance(row['layers'], list):
            all_layers.update(row['layers'])

    available_layers = sorted(all_layers)
    print(f"\n=== AVAILABLE LAYERS ===")
    print(f"Layers: {available_layers}")

    # Print sample data for inspection
    print("\n=== SAMPLE DATA ===")
    if not df.empty:
        sample_run = df.iloc[0]
        print(f"Sample run: {sample_run['name']}")
        print(f"Method: {sample_run['method']}")
        print(f"Sparsity coeff: {sample_run.get('sparsity_coeff', 'N/A')}")
        for layer in available_layers:
            sparsity = sample_run.get(f'sparsity_layer_{layer}', 'N/A')
            recon = sample_run.get(f'recon_loss_layer_{layer}', 'N/A')
            print(f"  Layer {layer}: Sparsity={sparsity}, Recon Loss={recon}")
        
        print(f"\nSuccessfully extracted sparsity coefficients for {len(df)} runs")
        if df['sparsity_coeff'].notna().any():
            print(f"Sparsity coefficient range: {df['sparsity_coeff'].min():.0e} to {df['sparsity_coeff'].max():.1f}")
        
        # Print all runs with their IDs and sparsity coefficients
        print("\n=== ALL RUNS WITH IDs AND SPARSITY COEFFICIENTS ===")
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('sparsity_coeff')
            print(f"\n{method} runs ({len(method_df)} total):")
            for idx, row in method_df.iterrows():
                sparsity_coeff_str = format_sparsity_coeff(row['sparsity_coeff'])
                print(f"  {row['id']} - λ={sparsity_coeff_str}")

    # Create the pareto plot
    if available_layers:
        # Focus on layers where SAEs are actually placed (have sparsity data)
        sae_layers = []
        for layer in available_layers:
            sparsity_col = f'sparsity_layer_{layer}'
            if sparsity_col in df.columns and not df[sparsity_col].isna().all():
                sae_layers.append(layer)
        
        print(f"\n=== SAE LAYERS WITH DATA ===")
        print(f"SAE layers: {sae_layers}")
        
        if sae_layers:
            print(f"\n=== CREATING PARETO PLOTS ===")
            print("Creating MSE vs L0 plots...")
            fig_mse = create_pareto_plot(df, sae_layers, 'sparsity_vs_reconstruction_loss_pareto.png', metric_type='mse')
            
            print("Creating Explained Variance vs L0 plots...")
            fig_var = create_pareto_plot(df, sae_layers, 'sparsity_vs_explained_variance_pareto.png', metric_type='explained_var')
            
            # Also create combined plots for SAE layers only
            print("\n=== CREATING COMBINED PLOTS ===")
            
            colors = {
                'ReLU': '#d62728',
                'HardConcrete': '#2ca02c', 
                'HardConcrete (IIG)': '#8c564b',
                'Gated': '#ff7f0e',
                'GatedHardConcrete': '#9467bd'
            }
            markers = {
                'ReLU': 'o',
                'HardConcrete': 's', 
                'HardConcrete (IIG)': 'v',
                'Gated': '^',
                'GatedHardConcrete': 'D'
            }
            sizes = {2: 40, 4: 50, 6: 60, 8: 70, 10: 80}  # Different sizes for different layers
            
            # Create MSE vs L0 combined plot
            print("Creating combined MSE vs L0 plot...")
            plt.figure(figsize=(12, 8))
            
            legend_added = {method: False for method in colors.keys()}
            
            for layer in sae_layers:
                sparsity_col = f'sparsity_layer_{layer}'
                recon_col = f'recon_loss_layer_{layer}'
                
                layer_df = df.dropna(subset=[sparsity_col, recon_col])
                
                for method in colors.keys():
                    method_df = layer_df[layer_df['method'] == method]
                    if not method_df.empty:
                        # Add label only once per method for legend
                        label = method if not legend_added[method] else ""
                        if label:
                            legend_added[method] = True
                        
                        # Sort by sparsity for line connection (no filtering for MSE plots - use all points)
                        method_df_sorted = method_df.sort_values(sparsity_col)
                        
                        # Plot line first (so it's behind the points)
                        plt.plot(
                            method_df_sorted[sparsity_col], 
                            method_df_sorted[recon_col],
                            c=colors[method],
                            alpha=0.6,
                            linewidth=1.5,
                            zorder=1
                        )
                        
                        # Plot scatter points on top
                        plt.scatter(
                            method_df[sparsity_col], 
                            method_df[recon_col],
                            c=colors[method], 
                            marker=markers[method],
                            s=sizes.get(layer, 60),
                            alpha=0.7,
                            label=label,
                            edgecolors='black',
                            linewidth=0.5,
                            zorder=2
                        )
                        
                        # Add sparsity coefficient annotations
                        for idx, row in method_df.iterrows():
                            if pd.notna(row['sparsity_coeff']):
                                plt.annotate(
                                    f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                                    (row[sparsity_col], row[recon_col]),
                                    xytext=(5, 5),  # offset from point
                                    textcoords='offset points',
                                    fontsize=8,
                                    color=colors[method],
                                    alpha=0.8,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                                    zorder=3
                                )
            
            plt.xlabel('Sparsity (L0)', fontsize=12)
            plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
            plt.title('Sparsity vs Reconstruction Loss Trade-off\n(5 SAE Architectures: ReLU, HardConcrete, HardConcrete (IIG), Gated, GatedHardConcrete)', fontsize=14)
            plt.yscale('log')
            
            # Only add legend if we have labeled artists
            if any(legend_added.values()):
                plt.legend(fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('combined_sparsity_vs_reconstruction_loss.png', dpi=300, bbox_inches='tight')
            plt.savefig('combined_sparsity_vs_reconstruction_loss.svg', bbox_inches='tight')
            plt.show()
            
            print("Combined MSE plot saved to combined_sparsity_vs_reconstruction_loss.png")
            
            # Create Explained Variance vs L0 combined plot
            print("Creating combined Explained Variance vs L0 plot...")
            plt.figure(figsize=(12, 8))
            
            legend_added = {method: False for method in colors.keys()}
            
            for layer in sae_layers:
                sparsity_col = f'sparsity_layer_{layer}'
                exp_var_col = f'explained_var_layer_{layer}'
                
                layer_df = df.dropna(subset=[sparsity_col, exp_var_col])
                
                for method in colors.keys():
                    method_df = layer_df[layer_df['method'] == method]
                    if not method_df.empty:
                        # Add label only once per method for legend
                        label = method if not legend_added[method] else ""
                        if label:
                            legend_added[method] = True
                        
                        # Filter points that are within the plot range
                        y_min = min(explained_variance_range[layer][0] for layer in explained_variance_range.keys())
                        visible_points = method_df[
                            (method_df[exp_var_col] >= y_min) & (method_df[exp_var_col] <= 1.0)
                        ]
                        
                        # Sort by sparsity for line connection (only visible points)
                        if not visible_points.empty:
                            visible_points_sorted = visible_points.sort_values(sparsity_col)
                            
                            # Plot line first (so it's behind the points) - only for visible points
                            plt.plot(
                                visible_points_sorted[sparsity_col], 
                                visible_points_sorted[exp_var_col],
                                c=colors[method],
                                alpha=0.6,
                                linewidth=1.5,
                                zorder=1
                            )
                        
                        # Plot scatter points on top
                        plt.scatter(
                            method_df[sparsity_col], 
                            method_df[exp_var_col],
                            c=colors[method], 
                            marker=markers[method],
                            s=sizes.get(layer, 60),
                            alpha=0.7,
                            label=label,
                            edgecolors='black',
                            linewidth=0.5,
                            zorder=2
                        )
                        
                        # Add sparsity coefficient annotations
                        for idx, row in method_df.iterrows():
                            if pd.notna(row['sparsity_coeff']):
                                plt.annotate(
                                    f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                                    (row[sparsity_col], row[exp_var_col]),
                                    xytext=(5, 5),  # offset from point
                                    textcoords='offset points',
                                    fontsize=8,
                                    color=colors[method],
                                    alpha=0.8,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                                    zorder=3
                                )
            
            plt.xlabel('Sparsity (L0)', fontsize=12)
            plt.ylabel('Explained Variance', fontsize=12)
            plt.title('Sparsity vs Explained Variance Trade-off\n(5 SAE Architectures: ReLU, HardConcrete, HardConcrete (IIG), Gated, GatedHardConcrete)', fontsize=14)
            # Use the broadest range to accommodate all layers
            y_min = min(explained_variance_range[layer][0] for layer in explained_variance_range.keys())
            y_max = max(explained_variance_range[layer][1] for layer in explained_variance_range.keys())
            plt.ylim(y_min, y_max)
            # Create exactly 5 ticks from y_min to 1.0
            ticks = np.linspace(y_min, 1.0, 5)
            plt.yticks(ticks)
            
            # Only add legend if we have labeled artists
            if any(legend_added.values()):
                plt.legend(fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('combined_sparsity_vs_explained_variance.png', dpi=300, bbox_inches='tight')
            plt.savefig('combined_sparsity_vs_explained_variance.svg', bbox_inches='tight')
            plt.show()
            
            print("Combined Explained Variance plot saved to combined_sparsity_vs_explained_variance.png")
            
            # Create detailed per-layer comparisons
            print("\n=== CREATING PER-LAYER COMPARISONS ===")
            
            n_sae_layers = len(sae_layers)
            
            # MSE per-layer comparison
            print("Creating per-layer MSE vs L0 plots...")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, layer in enumerate(sae_layers):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                sparsity_col = f'sparsity_layer_{layer}'
                recon_col = f'recon_loss_layer_{layer}'
                
                layer_df = df.dropna(subset=[sparsity_col, recon_col])
                
                for method in colors.keys():
                    method_df = layer_df[layer_df['method'] == method]
                    if not method_df.empty:
                        # Sort by sparsity for line connection (no filtering for MSE plots - use all points)
                        method_df_sorted = method_df.sort_values(sparsity_col)
                        
                        # Plot line first (so it's behind the points)
                        ax.plot(
                            method_df_sorted[sparsity_col], 
                            method_df_sorted[recon_col],
                            c=colors[method],
                            alpha=0.6,
                            linewidth=1.5,
                            zorder=1
                        )
                        
                        # Plot scatter points on top
                        ax.scatter(
                            method_df[sparsity_col], 
                            method_df[recon_col],
                            c=colors[method], 
                            marker=markers[method],
                            s=60,
                            alpha=0.7,
                            label=method,
                            edgecolors='black',
                            linewidth=0.5,
                            zorder=2
                        )
                        
                        # Add sparsity coefficient annotations
                        for idx, row in method_df.iterrows():
                            if pd.notna(row['sparsity_coeff']):
                                ax.annotate(
                                    f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                                    (row[sparsity_col], row[recon_col]),
                                    xytext=(5, 5),  # offset from point
                                    textcoords='offset points',
                                    fontsize=7,
                                    color=colors[method],
                                    alpha=0.8,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                                    zorder=3
                                )
                
                ax.set_xlabel('Sparsity (L0)')
                ax.set_ylabel('Reconstruction Loss (MSE)')
                ax.set_title(f'Layer {layer}')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(sae_layers), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('per_layer_sparsity_vs_reconstruction_loss.png', dpi=300, bbox_inches='tight')
            plt.savefig('per_layer_sparsity_vs_reconstruction_loss.svg', bbox_inches='tight')
            plt.show()
            
            print("Per-layer MSE plot saved to per_layer_sparsity_vs_reconstruction_loss.png")
            
            # Explained Variance per-layer comparison
            print("Creating per-layer Explained Variance vs L0 plots...")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, layer in enumerate(sae_layers):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                sparsity_col = f'sparsity_layer_{layer}'
                exp_var_col = f'explained_var_layer_{layer}'
                
                layer_df = df.dropna(subset=[sparsity_col, exp_var_col])
                
                for method in colors.keys():
                    method_df = layer_df[layer_df['method'] == method]
                    if not method_df.empty:
                        # Filter points that are within the plot range
                        y_min, y_max = explained_variance_range.get(layer, (0.98, 1.002))
                        visible_points = method_df[
                            (method_df[exp_var_col] >= y_min) & (method_df[exp_var_col] <= 1.0)
                        ]
                        
                        # Sort by sparsity for line connection (only visible points)
                        if not visible_points.empty:
                            visible_points_sorted = visible_points.sort_values(sparsity_col)
                            
                            # Plot line first (so it's behind the points) - only for visible points
                            ax.plot(
                                visible_points_sorted[sparsity_col], 
                                visible_points_sorted[exp_var_col],
                                c=colors[method],
                                alpha=0.6,
                                linewidth=1.5,
                                zorder=1
                            )
                        
                        # Plot scatter points on top
                        ax.scatter(
                            method_df[sparsity_col], 
                            method_df[exp_var_col],
                            c=colors[method], 
                            marker=markers[method],
                            s=60,
                            alpha=0.7,
                            label=method,
                            edgecolors='black',
                            linewidth=0.5,
                            zorder=2
                        )
                        
                        # Add sparsity coefficient annotations
                        for idx, row in method_df.iterrows():
                            if pd.notna(row['sparsity_coeff']):
                                ax.annotate(
                                    f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                                    (row[sparsity_col], row[exp_var_col]),
                                    xytext=(5, 5),  # offset from point
                                    textcoords='offset points',
                                    fontsize=7,
                                    color=colors[method],
                                    alpha=0.8,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                                    zorder=3
                                )
                
                ax.set_xlabel('Sparsity (L0)')
                ax.set_ylabel('Explained Variance')
                ax.set_title(f'Layer {layer}')
                y_min, y_max = explained_variance_range.get(layer, (0.98, 1.002))
                ax.set_ylim(y_min, y_max)
                # Create exactly 5 ticks from y_min to 1.0
                ticks = np.linspace(y_min, 1.0, 5)
                ax.set_yticks(ticks)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(sae_layers), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('per_layer_sparsity_vs_explained_variance.png', dpi=300, bbox_inches='tight')
            plt.savefig('per_layer_sparsity_vs_explained_variance.svg', bbox_inches='tight')
            plt.show()
            
            print("Per-layer Explained Variance plot saved to per_layer_sparsity_vs_explained_variance.png")

        else:
            print("No layers found with data!")

        # Create and save comprehensive data table
        print("\n=== CREATING DATA TABLE ===")
        
        # Prepare data for table
        table_data = []
        for _, row in df.iterrows():
            table_row = {
                'run_id': row['id'],
                'run_name': row['name'], 
                'method': row['method'],
                'sparsity_coeff': row['sparsity_coeff'],
                'state': row['state']
            }
            
            # Add per-layer metrics for this row
            for layer in sae_layers:
                sparsity_col = f'sparsity_layer_{layer}'
                recon_col = f'recon_loss_layer_{layer}'
                explained_var_col = f'explained_var_layer_{layer}'
                explained_var_ln_col = f'explained_var_ln_layer_{layer}'
                table_row[f'L0_layer_{layer}'] = row.get(sparsity_col, None)
                table_row[f'mse_layer_{layer}'] = row.get(recon_col, None)
                table_row[f'explained_var_layer_{layer}'] = row.get(explained_var_col, None)
                table_row[f'explained_var_ln_layer_{layer}'] = row.get(explained_var_ln_col, None)
            
            table_data.append(table_row)
        
        # Create table DataFrame  
        table_df = pd.DataFrame(table_data)
        
        # Sort by method and sparsity coefficient
        table_df = table_df.sort_values(['method', 'sparsity_coeff'])
        
        # Save to CSV
        table_df.to_csv('sae_pareto_data_table.csv', index=False)
        print("Data table saved to sae_pareto_data_table.csv")
        
        # Print summary table to console
        print("\n=== SUMMARY TABLE (First 10 rows) ===")
        display_cols = (['run_id', 'method', 'sparsity_coeff'] + 
                       [f'L0_layer_{layer}' for layer in sae_layers] + 
                       [f'mse_layer_{layer}' for layer in sae_layers] +
                       [f'explained_var_layer_{layer}' for layer in sae_layers] +
                       [f'explained_var_ln_layer_{layer}' for layer in sae_layers])
        summary_table = table_df[display_cols].head(10)
        
        # Format the table for better readability
        formatted_table = summary_table.copy()
        for col in formatted_table.columns:
            if ('L0_layer' in col or 'mse_layer' in col or 'explained_var' in col or col == 'sparsity_coeff'):
                formatted_table[col] = formatted_table[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) and x != 0 else str(x))
        
        print(formatted_table.to_string(index=False))
        
        # Also create a compact view focusing on layer 2 
        print("\n=== COMPACT VIEW (Layer 2 focus) ===")
        compact_cols = ['method', 'sparsity_coeff', f'L0_layer_2', f'mse_layer_2', f'explained_var_layer_2']
        compact_table = table_df[compact_cols].copy()
        compact_table.columns = ['Method', 'λ', 'L0', 'MSE', 'ExplVar']
        
        # Format for readability
        compact_table['λ'] = compact_table['λ'].apply(lambda x: f'{x:.3f}' if x < 0.01 else f'{x:.1f}')
        compact_table['L0'] = compact_table['L0'].apply(lambda x: f'{x:.1f}' if pd.notna(x) and x > 1 else f'{x:.2e}' if pd.notna(x) else 'N/A')
        compact_table['MSE'] = compact_table['MSE'].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
        compact_table['ExplVar'] = compact_table['ExplVar'].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
        
        print(compact_table.to_string(index=False))
        
        print(f"\nFull table with {len(table_df)} rows saved to CSV file.")
        
        # Print per-method statistics
        print("\n=== PER-METHOD SUMMARY ===")
        for method in table_df['method'].unique():
            method_data = table_df[table_df['method'] == method]
            print(f"\n{method} ({len(method_data)} runs):")
            print(f"  Sparsity coeff range: {method_data['sparsity_coeff'].min():.3f} - {method_data['sparsity_coeff'].max():.1f}")
            
            for layer in sae_layers:
                l0_col = f'L0_layer_{layer}'
                mse_col = f'mse_layer_{layer}'
                exp_var_col = f'explained_var_layer_{layer}'
                
                l0_data = method_data[l0_col].dropna()
                mse_data = method_data[mse_col].dropna()
                exp_var_data = method_data[exp_var_col].dropna()
                
                if not l0_data.empty and not mse_data.empty:
                    stats_str = f"    Layer {layer}: L0 range {l0_data.min():.2f}-{l0_data.max():.2f}, MSE range {mse_data.min():.2e}-{mse_data.max():.2e}"
                    if not exp_var_data.empty:
                        stats_str += f", ExplVar range {exp_var_data.min():.2e}-{exp_var_data.max():.2e}"
                    print(stats_str)

        # Print detailed statistics for SAE layers only
        print("\n=== DETAILED STATISTICS (SAE LAYERS ONLY) ===")
        sae_layers = []
        for layer in available_layers:
            sparsity_col = f'sparsity_layer_{layer}'
            if sparsity_col in df.columns and not df[sparsity_col].isna().all():
                sae_layers.append(layer)

        for layer in sae_layers:
            sparsity_col = f'sparsity_layer_{layer}'
            recon_col = f'recon_loss_layer_{layer}'
            
            layer_df = df.dropna(subset=[sparsity_col, recon_col])
            
            print(f"\nLayer {layer}:")
            for method in df['method'].unique():
                method_df = layer_df[layer_df['method'] == method]
                if not method_df.empty:
                    print(f"  {method}: {len(method_df)} runs")
                    print(f"    Sparsity range: {method_df[sparsity_col].min():.2f} - {method_df[sparsity_col].max():.2f}")
                    print(f"    Recon loss range: {method_df[recon_col].min():.2e} - {method_df[recon_col].max():.2e}")
                    
                    # Find best trade-offs (lowest sparsity for given reconstruction loss ranges)
                    print(f"    Performance summary:")
                    for recon_threshold in [0.001, 0.01, 0.1, 1.0]:
                        viable_runs = method_df[method_df[recon_col] <= recon_threshold]
                        if not viable_runs.empty:
                            min_sparsity = viable_runs[sparsity_col].min()
                            best_run = viable_runs[viable_runs[sparsity_col] == min_sparsity].iloc[0]
                            sparsity_coeff = format_sparsity_coeff(best_run['sparsity_coeff'])
                            print(f"      Best sparsity at recon ≤ {recon_threshold}: {min_sparsity:.2f} (run {best_run['id']}, λ={sparsity_coeff})")

if __name__ == "__main__":
    main() 