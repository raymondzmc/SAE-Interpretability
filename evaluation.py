import asyncio  # For running async explainer
import json
import os
from collections.abc import Sequence
from typing import Any
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from settings import settings
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import (  # ContextSize if needed
    PromptFormat,
    TokenActivationPairExplainer,
)
from neuron_explainer.explanations.explanations import ScoredSimulation
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import LogprobFreeExplanationTokenSimulator
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import AutoTokenizer


from config import Config
from data import create_dataloaders
from models import SAETransformer, SAETransformerOutput
from utils.io import load_config
from utils.enums import SAEType
# from e2e_sae import SAETransformer
# from e2e_sae.data import DatasetConfig, create_data_loader
# from e2e_sae.losses import calc_explained_variance

WINDOW_SIZE = 64
NUM_NEURONS = 300
TOP_K = 100
NUM_FEATURES_FOR_EXPLANATION = 20
MIN_COUNT = 20
N_EVAL_SAMPLES = 50000
OPENAI_API_KEY = settings.openai_api_key
EPSILON = 1e-6
OPENAI_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini-07-18",
    "gpt-4o": "gpt-4o-2024-11-20",
}
PROJECT = "raymondl/tinystories-1m"


def create_pareto_plots(all_run_metrics: list[dict[str, Any]]) -> None:
    """
    Create pareto plots showing trade-offs between sparsity and other metrics, with separate curves per layer.
    
    Args:
        all_run_metrics: List of dictionaries containing metrics for each run
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    
    # Prepare data for plotting, grouped by layer
    layer_data = defaultdict(list)
    
    for run_metric in all_run_metrics:
        run_name = run_metric['run_name']
        metrics = run_metric['metrics']
        config = run_metric['config']
        
        # Extract sparsity coefficient from config
        try:
            sparsity_coeff = config['loss']['sparsity']['coeff']
        except (KeyError, TypeError):
            sparsity_coeff = 'N/A'
        
        # Extract metrics for each SAE position for this run
        for sae_pos, pos_metrics in metrics.items():
            # Skip if metrics weren't computed (loading existing data case)
            if isinstance(pos_metrics.get('sparsity_l0'), str):
                continue
                
            layer_data[sae_pos].append({
                'run_name': run_name,
                'sae_position': sae_pos,
                'sparsity_l0': pos_metrics['sparsity_l0'],
                'mse': pos_metrics['mse'],
                'explained_variance': pos_metrics['explained_variance'],
                'alive_dict_proportion': pos_metrics['alive_dict_components_proportion'],
                'sparsity_coeff': sparsity_coeff
            })
    
    if not layer_data:
        print("No valid metrics found for plotting.")
        return
    
    # Get unique layers and sort them
    unique_layers = sorted(layer_data.keys())
    n_layers = len(unique_layers)
    
    print(f"Creating pareto plots for {n_layers} layers: {unique_layers}")
    
    # Create figure with subplots (3 metric types × number of layers)
    fig, axes = plt.subplots(3, n_layers, figsize=(6*n_layers, 18))
    
    # Handle case where there's only one layer
    if n_layers == 1:
        axes = axes.reshape(3, 1)
    
    output_dir = Path("pareto_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Plot for each layer
    for layer_idx, layer_name in enumerate(unique_layers):
        plot_data = layer_data[layer_name]
        
        # Convert to arrays for easier plotting
        sparsity = np.array([d['sparsity_l0'] for d in plot_data])
        mse = np.array([d['mse'] for d in plot_data])
        explained_variance = np.array([d['explained_variance'] for d in plot_data])
        alive_dict_proportion = np.array([d['alive_dict_proportion'] for d in plot_data])
        run_names = [d['run_name'] for d in plot_data]
        sparsity_coeffs = [d['sparsity_coeff'] for d in plot_data]
        
        # Color by run type (differentiate ReLU vs Bayesian if applicable)
        colors = []
        for name in run_names:
            if 'bayesian' in name.lower() or 'variational' in name.lower():
                colors.append('red')
            elif 'relu' in name.lower():
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Plot 1: Sparsity vs MSE
        axes[0, layer_idx].scatter(sparsity, mse, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, mse, sparsity_coeffs)):
            if coeff != 'N/A':
                axes[0, layer_idx].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
        axes[0, layer_idx].set_xlabel('L0 Sparsity')
        axes[0, layer_idx].set_ylabel('MSE (Reconstruction Loss)')
        axes[0, layer_idx].set_title(f'{layer_name}: Sparsity vs MSE')
        axes[0, layer_idx].grid(True, alpha=0.3)
        
        # Plot 2: Sparsity vs Explained Variance
        axes[1, layer_idx].scatter(sparsity, explained_variance, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, explained_variance, sparsity_coeffs)):
            if coeff != 'N/A':
                axes[1, layer_idx].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
        axes[1, layer_idx].set_xlabel('L0 Sparsity')
        axes[1, layer_idx].set_ylabel('Explained Variance')
        axes[1, layer_idx].set_title(f'{layer_name}: Sparsity vs Explained Variance')
        axes[1, layer_idx].grid(True, alpha=0.3)
        
        # Plot 3: Sparsity vs Alive Dictionary Proportion
        axes[2, layer_idx].scatter(sparsity, alive_dict_proportion, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, alive_dict_proportion, sparsity_coeffs)):
            if coeff != 'N/A':
                axes[2, layer_idx].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
        axes[2, layer_idx].set_xlabel('L0 Sparsity')
        axes[2, layer_idx].set_ylabel('Alive Dictionary Elements Proportion')
        axes[2, layer_idx].set_title(f'{layer_name}: Sparsity vs Alive Dict Elements')
        axes[2, layer_idx].grid(True, alpha=0.3)
        
        # Add legend to the first plot of each row
        if layer_idx == 0:
            unique_colors = list(set(colors))
            if len(unique_colors) > 1:
                legend_elements = []
                if 'red' in unique_colors:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Bayesian SAE'))
                if 'blue' in unique_colors:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='ReLU SAE'))
                if 'gray' in unique_colors:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other'))
                
                if legend_elements:
                    axes[1, layer_idx].legend(handles=legend_elements, loc='upper right')
        
        # Print layer-specific statistics
        valid_coeffs = [c for c in sparsity_coeffs if c != 'N/A']
        print(f"\n{layer_name} statistics ({len(plot_data)} data points):")
        print(f"  Sparsity range: {sparsity.min():.4f} - {sparsity.max():.4f}")
        print(f"  MSE range: {mse.min():.4f} - {mse.max():.4f}")
        print(f"  Explained Variance range: {explained_variance.min():.4f} - {explained_variance.max():.4f}")
        print(f"  Alive Dict Proportion range: {alive_dict_proportion.min():.4f} - {alive_dict_proportion.max():.4f}")
        if valid_coeffs:
            print(f"  Sparsity coefficient range: {min(valid_coeffs):.2e} - {max(valid_coeffs):.2e}")
        else:
            print(f"  Sparsity coefficient: N/A")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / "sparsity_pareto_plots_by_layer.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "sparsity_pareto_plots_by_layer.svg", bbox_inches='tight')
    
    # Also create individual plots for each layer for better readability
    for layer_idx, layer_name in enumerate(unique_layers):
        plot_data = layer_data[layer_name]
        
        # Convert to arrays for easier plotting
        sparsity = np.array([d['sparsity_l0'] for d in plot_data])
        mse = np.array([d['mse'] for d in plot_data])
        explained_variance = np.array([d['explained_variance'] for d in plot_data])
        alive_dict_proportion = np.array([d['alive_dict_proportion'] for d in plot_data])
        run_names = [d['run_name'] for d in plot_data]
        sparsity_coeffs = [d['sparsity_coeff'] for d in plot_data]
        
        # Color by run type
        colors = []
        for name in run_names:
            if 'bayesian' in name.lower() or 'variational' in name.lower():
                colors.append('red')
            elif 'relu' in name.lower():
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Create individual figure for this layer
        fig_individual, axes_individual = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Sparsity vs MSE
        axes_individual[0].scatter(sparsity, mse, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, mse, sparsity_coeffs)):
            if coeff != 'N/A':
                axes_individual[0].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=10, alpha=0.8)
        axes_individual[0].set_xlabel('L0 Sparsity')
        axes_individual[0].set_ylabel('MSE (Reconstruction Loss)')
        axes_individual[0].set_title(f'{layer_name}: Sparsity vs MSE')
        axes_individual[0].grid(True, alpha=0.3)
        
        # Plot 2: Sparsity vs Explained Variance
        axes_individual[1].scatter(sparsity, explained_variance, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, explained_variance, sparsity_coeffs)):
            if coeff != 'N/A':
                axes_individual[1].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=10, alpha=0.8)
        axes_individual[1].set_xlabel('L0 Sparsity')
        axes_individual[1].set_ylabel('Explained Variance')
        axes_individual[1].set_title(f'{layer_name}: Sparsity vs Explained Variance')
        axes_individual[1].grid(True, alpha=0.3)
        
        # Plot 3: Sparsity vs Alive Dictionary Proportion
        axes_individual[2].scatter(sparsity, alive_dict_proportion, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, alive_dict_proportion, sparsity_coeffs)):
            if coeff != 'N/A':
                axes_individual[2].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=10, alpha=0.8)
        axes_individual[2].set_xlabel('L0 Sparsity')
        axes_individual[2].set_ylabel('Alive Dictionary Elements Proportion')
        axes_individual[2].set_title(f'{layer_name}: Sparsity vs Alive Dict Elements')
        axes_individual[2].grid(True, alpha=0.3)
        
        # Add legend
        unique_colors = list(set(colors))
        if len(unique_colors) > 1:
            legend_elements = []
            if 'red' in unique_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Bayesian SAE'))
            if 'blue' in unique_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='ReLU SAE'))
            if 'gray' in unique_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other'))
            
            if legend_elements:
                axes_individual[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save individual layer plot
        safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
        plt.savefig(output_dir / f"pareto_{safe_layer_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f"pareto_{safe_layer_name}.svg", bbox_inches='tight')
        plt.close(fig_individual)
    
    total_points = sum(len(layer_data[layer]) for layer in unique_layers)
    unique_runs = set()
    for layer_points in layer_data.values():
        for point in layer_points:
            unique_runs.add(point['run_name'])
    
    print(f"\nSaved pareto plots to {output_dir}/")
    print(f"  - Combined: sparsity_pareto_plots_by_layer.png|svg")
    print(f"  - Individual: pareto_{{layer_name}}.png|svg for each layer")
    print(f"Plotted {total_points} total data points from {len(unique_runs)} unique runs across {n_layers} layers")
    
    plt.show()

def save_activation_data(accumulated_data: dict[str, dict[str, torch.Tensor]], run_dir: str) -> None:
    """Save accumulated activation/probability data with a directory for the run and a file for each layer.
    
    Note: For feature extraction, Bayesian SAEs use Hard Concrete gate probabilities while ReLU SAEs use activations.
    However, metrics (L0 sparsity, MSE, explained variance) are computed using actual activations for both types.
    """
    activation_data_dir = os.path.join(run_dir, "activation_data")
    os.makedirs(activation_data_dir, exist_ok=True)
    
    for sae_pos, data in accumulated_data.items():
        # Replace dots and other characters that might be problematic in filenames
        safe_layer_name = sae_pos.replace(".", "--").replace("/", "--")
        file_path = os.path.join(activation_data_dir, f"{safe_layer_name}.pt")
        torch.save(data, file_path)
        print(f"Saved activation data for {sae_pos} to {file_path}")


def load_activation_data(run_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    """Load accumulated activation data for a specific run."""
    activation_data_dir = os.path.join(run_dir, "activation_data")
    
    if not os.path.exists(activation_data_dir):
        raise FileNotFoundError(f"Activation data directory not found: {activation_data_dir}")
    
    accumulated_data = {}
    # Find all .pt files in the run directory
    for filename in os.listdir(activation_data_dir):
        if filename.endswith(".pt"):
            # Convert filename back to original sae_pos format
            safe_layer_name = filename[:-3]  # Remove .pt extension
            sae_pos = safe_layer_name.replace("--", ".")  # Convert back from safe filename

            file_path = os.path.join(activation_data_dir, filename)
            data = torch.load(file_path)
            accumulated_data[sae_pos] = data
            
            print(f"Loaded activation data for {sae_pos} from {file_path}")
    
    return accumulated_data


async def _run_simulation_and_scoring(
    explanation_text: str,
    records_for_simulation: Sequence[ActivationRecord],
    model_name_for_simulator: str,
    few_shot_example_set: FewShotExampleSet,
    prompt_format: PromptFormat,
    num_retries: int = 5,
) -> tuple[float | None, Any | None]:
    """Helper to run simulation and scoring with retries."""
    attempts = 0
    score = None
    scored_simulation = None
    while attempts < num_retries:  # Retry loop
        try:
            simulator = UncalibratedNeuronSimulator(
                LogprobFreeExplanationTokenSimulator(
                    model_name_for_simulator,
                    explanation_text,
                    json_mode=True,
                    max_concurrent=10,
                    few_shot_example_set=few_shot_example_set,
                    prompt_format=prompt_format,
                )
            )
            scored_simulation: ScoredSimulation = await simulate_and_score(simulator, records_for_simulation)
            score = scored_simulation.get_preferred_score() if scored_simulation else None

        except Exception as e:
            print(f"Error in attempt {attempts + 1}: {e}")
            attempts += 1

        if score is not None and not np.isnan(score):
            break

    return score, scored_simulation


    
    

device = torch.device('cpu')
wandb.login(key=settings.wandb_api_key)
api = wandb.Api()
runs = api.runs(PROJECT)

# Collect all metrics for pareto plots
all_run_metrics = []

for run in runs:
    run_id = run.id
    run_config = run.config
    run_config['data']['n_eval_samples'] = N_EVAL_SAMPLES
    config = load_config(run_config, Config)
    run_dir = f"evaluation_results/{config.wandb_run_name}-{run_id}"

    _, eval_loader = create_dataloaders(data_config=config.data, global_seed=config.seed)
    metrics = {}
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    model = SAETransformer.from_wandb(f"{PROJECT}/{run_id}").to(device)
    model.saes.eval()

    if os.path.exists(run_dir):
        accumulated_data = load_activation_data(run_dir)
        all_token_ids = torch.load(os.path.join(run_dir, "all_token_ids.pt"))
        
        # Initialize metrics when loading existing data
        metrics_file = os.path.join(run_dir, "evaluation_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            print(f"Loaded existing evaluation metrics from {metrics_file}")
        else:
            # Initialize with placeholder values - these will only be updated with alive dict components
            for sae_pos in model.raw_sae_positions:
                metrics[sae_pos] = {
                    'alive_dict_components': 0,
                    'alive_dict_components_proportion': 0.0,
                    'sparsity_l0': 'not_computed_from_existing_data',
                    'mse': 'not_computed_from_existing_data', 
                    'explained_variance': 'not_computed_from_existing_data',
                }
            print("Note: L0 sparsity, MSE, and explained variance not computed when loading existing data")
    else:
        print(f"Obtaining features for {run_id}")
        total_tokens = 0
        all_token_ids: list[list[int]] = []

        # Create placeholder tensors for efficient batch accumulation
        # Note: For feature extraction, 'nonzero_activations' stores probabilities for Bayesian SAEs, activations for ReLU SAEs
        accumulated_data = {}
        for sae_pos in model.raw_sae_positions:
            accumulated_data[sae_pos] = {
                'nonzero_activations': torch.empty(0, WINDOW_SIZE, dtype=torch.float16),
                'data_indices': torch.empty(0, dtype=torch.long),
                'neuron_indices': torch.empty(0, dtype=torch.long),
            }
            metrics[sae_pos] = {
                'alive_dict_components': set(),
                'sparsity_l0': 0.0,
                'mse': 0.0,
            }
        
        # Compute max activations and metrics for each neuron
        max_activations = {pos: None for pos in model.raw_sae_positions}
        total_tokens = 0
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Computing max activations and metrics")):
            token_ids = batch[config.data.column_name].to(device)
            batch_size_, seq_len = token_ids.shape
            num_chunks = seq_len // WINDOW_SIZE
            token_ids_chunked = token_ids.view(batch_size_, num_chunks, WINDOW_SIZE)
            token_ids_chunked = token_ids_chunked.reshape(-1, WINDOW_SIZE)
            n_tokens = token_ids_chunked.shape[0] * token_ids_chunked.shape[1]
            total_tokens += n_tokens
            with torch.no_grad():
                output: SAETransformerOutput = model.forward(
                    tokens=token_ids_chunked,
                    sae_positions=model.raw_sae_positions,
                    compute_loss=False,
                )
            import pdb; pdb.set_trace()

            for pos in model.raw_sae_positions:
                metrics[pos]['mse'] += mse_loss(
                    output.sae_outputs[pos].output,
                    output.sae_outputs[pos].input,
                    reduction='mean'
                ) * n_tokens
                activations = output.sae_outputs[pos].c
                metrics[pos]['sparsity_l0'] += torch.norm(activations, p=0, dim=-1).mean().item() * n_tokens
                metrics[pos]['alive_dict_components'].update(activations.sum(0).sum(0).nonzero().unique().cpu().tolist())

                # For feature extraction: Use probabilities for Bayesian SAEs, activations for ReLU SAEs
                if config.saes.sae_type == SAEType.HardConcrete:
                    acts = sae_acts.z
                elif config.saes.sae_type == SAEType.RELU:
                    acts = sae_acts.c
                elif config.saes.sae_type == SAEType.GATED:
                    acts = sae_acts.mask
                else:
                    raise ValueError(f"Invalid SAE type: {config.saes.sae_type}")
                
                max_acts = acts.view(-1, acts.size(-1)).max(dim=0).values
                if max_activations[pos] is None:
                    max_activations[pos] = max_acts
                else:
                    max_activations[pos] = torch.maximum(max_activations[pos], max_acts)


        # Store activations for each neuron
        for batch_idx, batch in tqdm(enumerate(eval_loader), desc="Eval Steps"):
            token_ids = batch[config.data.column_name].to(device)
            n_tokens = token_ids.shape[0] * token_ids.shape[1]
            total_tokens += n_tokens

            # Reshape token_ids to break into chunks of WINDOW_SIZE
            batch_size, seq_len = token_ids.shape
            if seq_len % WINDOW_SIZE != 0:
                raise ValueError(f"Sequence length {seq_len} is not divisible by WINDOW_SIZE {WINDOW_SIZE}")

            num_chunks = seq_len // WINDOW_SIZE
            chunked_batch_size = batch_size * num_chunks
            token_ids_chunked = token_ids.view(batch_size, num_chunks, WINDOW_SIZE)
            token_ids_chunked = token_ids_chunked.reshape(chunked_batch_size, WINDOW_SIZE)

            # Run through the SAE-augmented model
            with torch.no_grad():
                new_logits, new_acts = model.forward(
                    tokens=token_ids_chunked,
                    sae_positions=model.raw_sae_positions,
                    cache_positions=None,
                )

            for sae_pos in model.raw_sae_positions:
                sae_acts = new_acts[sae_pos]
                
                if config.saes.sae_type == SAEType.HardConcrete:
                    acts = sae_acts.z
                elif config.saes.sae_type == SAEType.RELU:
                    acts = sae_acts.c
                elif config.saes.sae_type == SAEType.GATED:
                    acts = sae_acts.mask
                else:
                    raise ValueError(f"Invalid SAE type: {config.saes.sae_type}")

                # # For feature extraction: Use probabilities for Bayesian SAEs, activations for ReLU SAEs
                # if hasattr(sae_acts, 'logits') and sae_acts.logits is not None:
                #     # Compute probabilities for Bayesian SAE feature extraction
                #     # P(z > 0) = sigmoid(logits - beta * log(-l/r))
                #     probs = compute_hard_concrete_probabilities(
                #         sae_acts.logits, sae_acts.beta, sae_acts.l, sae_acts.r
                #     )
                #     if probs is not None:
                #         # Use probabilities for feature extraction
                #         acts = probs  # chunked_batch_size x seq_len x num_dictionary_elements
                #         max_acts = max_activations[sae_pos]
                #         # For probabilities, we don't need to normalize by max_acts since they're already in [0,1]
                #         # Instead, we'll discretize the probabilities directly
                #         discretized_acts = torch.round(acts * 10).to(torch.int8)
                #     else:
                #         print(f"Warning: Invalid Hard Concrete parameters for {sae_pos}, falling back to activations")
                #         acts = sae_acts.c
                #         max_acts = max_activations[sae_pos]
                #         safe_max = torch.where(max_acts > 0, max_acts, torch.ones_like(max_acts))
                #         discretized_acts = torch.round((acts / safe_max.unsqueeze(0).unsqueeze(0)) * 10).to(torch.int8)
                # else:
                #     # Regular SAE - use activations
                #     acts = sae_acts.c # chunked_batch_size x seq_len x num_dictionary_elements
                #     max_acts = max_activations[sae_pos]
                #     safe_max = torch.where(max_acts > 0, max_acts, torch.ones_like(max_acts))
                #     discretized_acts = torch.round((acts / safe_max.unsqueeze(0).unsqueeze(0)) * 10).to(torch.int8)
                
                data_indices, neuron_indices = acts.sum(1).nonzero(as_tuple=True)
                if data_indices.numel() > 0:
                    # Extract all relevant activations/probabilities at once (N, seq_len)
                    nonzero_activations = acts[data_indices, :, neuron_indices]

                    # Add the offset to the data indices for global indexing
                    global_data_indices = data_indices + len(all_token_ids)

                    # Accumulate tensors for this SAE position
                    accumulated_data[sae_pos]['nonzero_activations'] = torch.cat([
                        accumulated_data[sae_pos]['nonzero_activations'], 
                        nonzero_activations.cpu()
                    ], dim=0)
                    accumulated_data[sae_pos]['data_indices'] = torch.cat([
                        accumulated_data[sae_pos]['data_indices'], 
                        global_data_indices.cpu()
                    ], dim=0)
                    accumulated_data[sae_pos]['neuron_indices'] = torch.cat([
                        accumulated_data[sae_pos]['neuron_indices'], 
                        neuron_indices.cpu()
                    ], dim=0)

            chunked_tokens = [tokenizer.convert_ids_to_tokens(token_ids_chunked[i]) for i in range(chunked_batch_size)]
            all_token_ids.extend(chunked_tokens)

        # Average metrics over all batches
        for sae_pos in model.raw_sae_positions:
            metrics[sae_pos]['sparsity_l0'] /= total_tokens
            metrics[sae_pos]['mse'] /= total_tokens
            # explained_variance is already averaged per batch, so we keep the last value
            
        # After all batches are processed, save and analyze the accumulated data
        print("Saving accumulated activation data...")
        save_activation_data(accumulated_data, run_dir)
        torch.save(all_token_ids, os.path.join(run_dir, "all_token_ids.pt"))
    print("Processing accumulated activation data...")
    for sae_pos in model.raw_sae_positions:
        data = accumulated_data[sae_pos]

        # Find all unique neurons with at least one non-zero activation
        unique_neurons = torch.unique(data['neuron_indices'])
        n_dict_components = model.saes[sae_pos.replace('.', '-')].n_dict_components
        alive_dict_components = len(unique_neurons)
        alive_dict_components_proportion = alive_dict_components / n_dict_components
        
        # Update metrics with alive dictionary components
        metrics[sae_pos]['alive_dict_components'] = alive_dict_components
        metrics[sae_pos]['alive_dict_components_proportion'] = alive_dict_components_proportion
        
        print(
            f"SAE position {sae_pos}: "
            f"{alive_dict_components}/{n_dict_components} "
            f"({alive_dict_components_proportion * 100:.2f}%) "
            "neurons with non-zero activations (Alive Dictionary Components)"
        )
        print(f"Total activation records for {sae_pos}: {data['nonzero_activations'].shape[0]}")
        
        # Print all computed metrics
        print(f"Metrics for {sae_pos}:")
        
        # Handle metrics that might not be computed when loading existing data
        if isinstance(metrics[sae_pos]['sparsity_l0'], str):
            print(f"  L0 Sparsity: {metrics[sae_pos]['sparsity_l0']}")
        else:
            print(f"  L0 Sparsity: {metrics[sae_pos]['sparsity_l0']:.6f}")
            
        if isinstance(metrics[sae_pos]['mse'], str):
            print(f"  MSE (Reconstruction Loss): {metrics[sae_pos]['mse']}")
        else:
            print(f"  MSE (Reconstruction Loss): {metrics[sae_pos]['mse']:.6f}")
            
        if isinstance(metrics[sae_pos]['explained_variance'], str):
            print(f"  Explained Variance: {metrics[sae_pos]['explained_variance']}")
        else:
            print(f"  Explained Variance: {metrics[sae_pos]['explained_variance']:.6f}")
            
        print(f"  Alive Dictionary Components: {metrics[sae_pos]['alive_dict_components']}")
        print(f"  Alive Dict Components Proportion: {metrics[sae_pos]['alive_dict_components_proportion']:.4f}")
        print()
    
    # Save metrics to JSON file (only if we computed new metrics or updated alive dict components)
    metrics_file = os.path.join(run_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_file}")
    
    # Collect metrics for pareto plot
    run_metrics = {
        'run_id': run_id,
        'run_name': run.name,
        'config': config,
        'metrics': metrics
    }
    all_run_metrics.append(run_metrics)
    
    # Initialize TokenActivationPairExplainer
    explainer = TokenActivationPairExplainer(
        model_name=OPENAI_MODELS.get("gpt-4o", "gpt-4o"), 
        prompt_format=PromptFormat.HARMONY_V4
    )
    
    # Filter neurons with at least MIN_COUNT examples and construct activation records
    explanations_for_run = {}
    
    import pdb; pdb.set_trace()
    for sae_pos in model.raw_sae_positions:
        data = accumulated_data[sae_pos]
        
        # Count occurrences of each neuron and calculate total activation
        unique_neurons, counts = torch.unique(data['neuron_indices'], return_counts=True)
        
        # Calculate total activation/probability for each unique neuron
        # Note: For Bayesian SAEs, this sums probabilities; for ReLU SAEs, this sums activations
        neuron_total_activations = []
        for neuron_idx in unique_neurons:
            neuron_mask = data['neuron_indices'] == neuron_idx
            neuron_activations = data['nonzero_activations'][neuron_mask].float()
            total_activation = neuron_activations.sum().item()
            neuron_total_activations.append(total_activation)
        
        neuron_total_activations = torch.tensor(neuron_total_activations)
        
        # Sort neurons by total activation (descending) and take top NUM_NEURONS
        sorted_indices = torch.argsort(neuron_total_activations, descending=True)
        top_neuron_indices = sorted_indices[:NUM_NEURONS]
        top_neurons = unique_neurons[top_neuron_indices]
        top_counts = counts[top_neuron_indices]
        
        # Filter top neurons that also have at least MIN_COUNT examples
        neurons_with_min_count = top_neurons[top_counts >= MIN_COUNT]
        
        print(f"SAE position {sae_pos}: {len(unique_neurons)} total neurons, taking top {NUM_NEURONS}")
        print(f"  {len(neurons_with_min_count)} of top {NUM_NEURONS} neurons have at least {MIN_COUNT} examples")
        print(f"  Processing {len(neurons_with_min_count)} neurons for explanation...")
        
        # Process each neuron that meets the criteria (top activation + minimum count)
        for neuron_idx in neurons_with_min_count:
            neuron_idx_item = neuron_idx.item()
            
            # Get all data for this specific neuron
            neuron_mask = data['neuron_indices'] == neuron_idx
            neuron_data_indices = data['data_indices'][neuron_mask]
            neuron_activations = data['nonzero_activations'][neuron_mask]  # (n_examples, seq_len)
            
                         # Get top 10 activation records with highest activation values
            max_activations_per_example = neuron_activations.float().max(dim=1).values  # Max across sequence
            
            # Get indices sorted by activation value (descending)
            sorted_indices = torch.argsort(max_activations_per_example, descending=True)
            
            # Take top 10 (or fewer if less than 10 examples)
            top_k = min(10, len(sorted_indices))
            top_indices = sorted_indices[:top_k]
            
            # Convert to activation records for top examples only
            activation_records = []
            for idx in top_indices:
                i = idx.item()
                data_idx = neuron_data_indices[i].item()
                activations = neuron_activations[i].float().tolist()  # Convert to list of floats
                max_activation_value = max_activations_per_example[i].item()
                
                # Get the corresponding tokens
                if data_idx < len(all_token_ids):
                    tokens = [token.replace("Ġ", "") for token in all_token_ids[data_idx]]
                    
                    # Create activation record
                    activation_record = ActivationRecord(
                        tokens=tokens,
                        activations=activations
                    )
                    activation_records.append(activation_record)
            
            if not activation_records:
                print(f"  Skipping neuron {neuron_idx_item} - no valid activation records")
                continue
                
            print(f"  Processing neuron {neuron_idx_item} with {len(activation_records)} examples...")
            
            # Calculate max activation for this neuron
            max_activation = calculate_max_activation(activation_records)
            if max_activation == 0:
                print(f"  Skipping neuron {neuron_idx_item} - max activation is zero")
                continue
            
            try:
                # Generate explanation for this neuron
                generated_explanations = asyncio.run(explainer.generate_explanations(
                    all_activation_records=activation_records,
                    max_activation=max_activation,
                    num_samples=1,
                    max_tokens=100,
                    temperature=0.0
                ))
                if generated_explanations:
                    explanation = generated_explanations[0].strip()
                    print(f"    Neuron {neuron_idx_item}: {explanation}")
                    
                    # Prepare records for scoring (clean up tokens)
                    temp_activation_records = [
                        ActivationRecord(
                            tokens=[
                                token.replace("<|endoftext|>", "<|not_endoftext|>")
                                .replace(" 55", "_55")
                                .replace("Ġ", "")
                                .encode("ascii", errors="backslashreplace")
                                .decode("ascii")
                                for token in record.tokens
                            ],
                            activations=record.activations,
                        )
                        for record in activation_records
                    ]
                    
                    # Score the explanation
                    score, scored_simulation_details = asyncio.run(
                        _run_simulation_and_scoring(
                            explanation_text=explanation,
                            records_for_simulation=temp_activation_records,
                            model_name_for_simulator='gpt-4o-mini',
                            few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
                            prompt_format=PromptFormat.HARMONY_V4
                        )
                    )
                    # Store the explanation and score
                    key = f"{sae_pos}_neuron_{neuron_idx_item}"
                    explanations_for_run[key] = {
                        "text": explanation,
                        "score": score,
                        "sae_position": sae_pos,
                        "neuron_index": neuron_idx_item,
                        "num_examples": len(activation_records)
                    }
                    
                    print(f"    Neuron {neuron_idx_item} - Score: {score}")
                    
                else:
                    print(f"    No explanation generated for neuron {neuron_idx_item}")
                    
            except Exception as e:
                print(f"    Error processing neuron {neuron_idx_item}: {e}")
    
    # Save all explanations to a JSON file
    output_explanation_file = os.path.join(run_dir, "explanations.json")
    os.makedirs(os.path.dirname(output_explanation_file), exist_ok=True)
    with open(output_explanation_file, "w") as f:
        json.dump(explanations_for_run, f, indent=2)
    print(f"\nSaved explanations for run {run_id} to {output_explanation_file}")

# Create pareto plots after processing all runs
print(f"\nCreating pareto plots from {len(all_run_metrics)} runs...")
create_pareto_plots(all_run_metrics)
