import asyncio  # For running async explainer
import json
import os
from collections.abc import Sequence
from typing import Any
import numpy as np
import torch
import wandb
import argparse
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
from utils.io import (
    load_config, 
    save_activation_data_to_wandb,
    save_metrics_to_wandb,
    save_explanations_to_wandb,
    load_activation_data_from_wandb,
    load_metrics_from_wandb
)
from utils.enums import SAEType
from utils.metrics import explained_variance
from utils.plotting import create_pareto_plots


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


def run_evaluation(args: argparse.Namespace) -> None:
    """
    Run SAE evaluation including activation data collection, neuron explanation generation, and analysis.
    """
    
    # OpenAI model mappings
    OPENAI_MODELS = {
        "gpt-4o-mini": "gpt-4o-mini-07-18",
        "gpt-4o": "gpt-4o-2024-11-20",
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    runs = api.runs(args.wandb_project)

    if args.filter_runs_by_name is not None:
        old_len = len(runs)
        runs = [run for run in runs if args.filter_runs_by_name in run.name]
        print(f"Found {len(runs)}/{old_len} runs matching filter: {args.filter_runs_by_name}")

    # Collect all metrics for pareto plots
    all_run_metrics = []

    for run in runs:
        run_id = run.id
        run_config = run.config
        run_config['data']['n_eval_samples'] = args.n_eval_samples
        
        # Override n_train_samples if specified (to avoid slow data skipping)
        if args.override_n_train_samples is not None:
            print(f"Overriding n_train_samples: {run_config['data']['n_train_samples']} -> {args.override_n_train_samples}")
            run_config['data']['n_train_samples'] = args.override_n_train_samples
            
        config: Config = load_config(run_config, Config)

        # Initialize Wandb run for this specific run (resume original run)
        wandb.init(
            project=args.wandb_project.split("/")[-1],  # Extract project name
            entity=args.wandb_project.split("/")[0],    # Extract entity  
            id=run_id,  # Use the same run ID
            resume="allow",  # Allow resuming existing run
            reinit="finish_previous"
        )

        _, eval_loader = create_dataloaders(data_config=config.data, global_seed=config.seed)
        metrics = {}
        accumulated_data = None
        all_token_ids = None
        loaded_metrics = None
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
        model = SAETransformer.from_wandb(f"{args.wandb_project}/{run_id}").to(device)
        model.saes.eval()

        # Try to load existing data from Wandb
        print(f"Attempting to load activation data from Wandb for run {run_id}...")
        try:
            accumulated_data, all_token_ids = load_activation_data_from_wandb(
                run_id, project=args.wandb_project
            )
            
            print(f"Successfully loaded activation data from Wandb run files")
            if all_token_ids is not None:
                print(f"Loaded token IDs from Wandb run files")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No existing activation data found: {e}")
            print("Will compute activation data and metrics from scratch")
        
        # Try to load existing metrics separately
        try:
            loaded_metrics = load_metrics_from_wandb(run_id, project=args.wandb_project)
            if loaded_metrics is not None and not args.force_recompute:
                metrics = loaded_metrics
                print(f"Loaded existing metrics from Wandb for {len(metrics)} SAE positions")
            elif loaded_metrics is not None and args.force_recompute:
                print(f"Found existing metrics but --force_recompute is set, will recompute")
        except Exception as e:
            print(f"No existing metrics found: {e}")
            print("Will compute metrics from scratch")

        if accumulated_data is None or len(metrics) == 0:
            print(f"Obtaining features for {run_id}")
            total_tokens = 0
            all_token_ids: list[list[str]] = []

            # Create placeholder tensors for efficient batch accumulation
            # Note: For feature extraction, 'nonzero_activations' stores probabilities for Bayesian SAEs, activations for ReLU SAEs
            accumulated_data = {}
            for sae_pos in model.raw_sae_positions:
                accumulated_data[sae_pos] = {
                    'nonzero_activations': torch.empty(0, args.window_size, dtype=torch.float16),
                    'data_indices': torch.empty(0, dtype=torch.long),
                    'neuron_indices': torch.empty(0, dtype=torch.long),
                }
                metrics[sae_pos] = {
                    'alive_dict_components': set(),
                    'sparsity_l0': 0.0,
                    'mse': 0.0,
                    'explained_variance': 0.0,
                }
            
            total_tokens = 0
            for batch in tqdm(eval_loader, desc="Processing batches"):
                token_ids = batch[config.data.column_name].to(device)
                
                # Reshape token_ids to break into chunks of window_size
                batch_size, seq_len = token_ids.shape
                if seq_len % args.window_size != 0:
                    raise ValueError(f"Sequence length {seq_len} is not divisible by window_size {args.window_size}")

                num_chunks = seq_len // args.window_size
                chunked_batch_size = batch_size * num_chunks
                token_ids_chunked = token_ids.view(batch_size, num_chunks, args.window_size)
                token_ids_chunked = token_ids_chunked.reshape(chunked_batch_size, args.window_size)
                
                n_tokens = token_ids_chunked.shape[0] * token_ids_chunked.shape[1]
                total_tokens += n_tokens

                # Run through the SAE-augmented model
                with torch.no_grad():
                    output: SAETransformerOutput = model.forward(
                        tokens=token_ids_chunked,
                        sae_positions=model.raw_sae_positions,
                        compute_loss=True,
                    )

                for sae_pos in model.raw_sae_positions:
                    sae_output = output.sae_outputs[sae_pos]
                    
                    # Compute metrics
                    metrics[sae_pos]['mse'] += mse_loss(
                        sae_output.output,
                        sae_output.input,
                        reduction='mean'
                    ).item() * n_tokens
                    
                    # Compute explained variance using the proper function from utils.metrics
                    exp_var = explained_variance(
                        sae_output.output,
                        sae_output.input,
                        layer_norm_flag=False
                    ).mean().item()
                    metrics[sae_pos]['explained_variance'] += exp_var * n_tokens
                    
                    # Get activations based on SAE type
                    if config.saes.sae_type == SAEType.HARD_CONCRETE:
                        if args.hard_concrete_method == "c":
                            acts = sae_output.c
                        elif args.hard_concrete_method == "z":
                            acts = sae_output.z
                        else:
                            raise ValueError(f"Invalid hard concrete method: {args.hard_concrete_method}")
                    elif config.saes.sae_type == SAEType.RELU:
                        acts = sae_output.c
                    elif config.saes.sae_type == SAEType.GATED:
                        acts = sae_output.mask if hasattr(sae_output, 'mask') else sae_output.c
                    elif config.saes.sae_type == SAEType.TOPK:
                        acts = sae_output.code
                    else:
                        acts = sae_output.c  # Default to main activations
                    
                    # Update sparsity and alive components
                    metrics[sae_pos]['sparsity_l0'] += torch.norm(acts, p=0, dim=-1).mean().item() * n_tokens
                    
                    # Get alive components, ensuring we always have a list
                    nonzero_indices = acts.sum(0).sum(0).nonzero().squeeze().cpu()
                    if nonzero_indices.numel() == 0:
                        alive_indices = []
                    elif nonzero_indices.numel() == 1:
                        alive_indices = [nonzero_indices.item()]
                    else:
                        alive_indices = nonzero_indices.tolist()
                    
                    metrics[sae_pos]['alive_dict_components'].update(alive_indices)

                    if args.save_activation_data:
                        # Collect non-zero activations for explanation generation
                        data_indices, neuron_indices = acts.sum(1).nonzero(as_tuple=True)
                        if data_indices.numel() > 0:
                            # Extract all relevant activations at once (N, seq_len)
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

                # Store tokenized sequences for explanation generation
                chunked_tokens = [tokenizer.convert_ids_to_tokens(token_ids_chunked[i]) for i in range(chunked_batch_size)]
                all_token_ids.extend(chunked_tokens)

            # Average metrics over all batches and convert alive components from set to count
            for sae_pos in model.raw_sae_positions:
                metrics[sae_pos]['sparsity_l0'] /= total_tokens
                metrics[sae_pos]['mse'] /= total_tokens
                metrics[sae_pos]['explained_variance'] /= total_tokens
                
                # Convert alive components set to count and proportion
                alive_components = metrics[sae_pos]['alive_dict_components']
                num_alive = len(alive_components)
                total_dict_size = accumulated_data[sae_pos]['neuron_indices'].max().item() + 1 if accumulated_data[sae_pos]['neuron_indices'].numel() > 0 else 1
                
                metrics[sae_pos]['alive_dict_components'] = num_alive
                metrics[sae_pos]['alive_dict_components_proportion'] = num_alive / total_dict_size
            
            # Always save metrics
            print("Saving metrics to Wandb...")
            try:
                save_metrics_to_wandb(metrics=metrics)
            except Exception as e:
                print(f"Warning: Failed to upload metrics to Wandb: {e}")

            # Save activation data to Wandb
            if args.save_activation_data:
                print("Saving accumulated activation data to Wandb...")
                try:
                    save_activation_data_to_wandb(
                        accumulated_data=accumulated_data,
                        all_token_ids=all_token_ids
                    )
                except Exception as e:
                    print(f"Warning: Failed to upload activation data to Wandb: {e}")
            
            

        # Collect metrics for pareto plot
        run_metrics = {
            'run_id': run_id,
            'run_name': run.name,
            'config': config,
            'metrics': metrics
        }
        
        all_run_metrics.append(run_metrics)
        
        if args.generate_explanations:
            # Initialize TokenActivationPairExplainer
            explainer = TokenActivationPairExplainer(
                model_name=OPENAI_MODELS.get(args.explanation_model, args.explanation_model), 
                prompt_format=PromptFormat.HARMONY_V4
            )
            
            # Filter neurons with sufficient examples and construct activation records
            explanations_for_run = {}

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
                top_neuron_indices = sorted_indices[:args.num_neurons]
                top_neurons = unique_neurons[top_neuron_indices]
                top_counts = counts[top_neuron_indices]
                
                # Take top neurons for explanation (we'll use top examples regardless of count)
                neurons_to_explain = top_neurons
                
                print(f"SAE position {sae_pos}: {len(unique_neurons)} total neurons, taking top {args.num_neurons}")
                print(f"  Processing {len(neurons_to_explain)} neurons for explanation...")
                
                # Process each neuron for explanation
                for neuron_idx in neurons_to_explain:
                    neuron_idx_item = neuron_idx.item()
                    
                    # Get all data for this specific neuron
                    neuron_mask = data['neuron_indices'] == neuron_idx
                    neuron_data_indices = data['data_indices'][neuron_mask]
                    neuron_activations = data['nonzero_activations'][neuron_mask]  # (n_examples, seq_len)
                    
                    # Get top 10 activation records with highest activation values
                    max_activations_per_example = neuron_activations.float().max(dim=1).values  # Max across sequence
                    
                    # Get indices sorted by activation value (descending)
                    sorted_indices = torch.argsort(max_activations_per_example, descending=True)
                    
                    # Take top num_features_to_explain examples (or fewer if not enough examples)
                    top_k = min(args.num_features_to_explain, len(sorted_indices))
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
                            
                            explanation_score = None
                            if args.evaluate_explanations:
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
                                explanation_score, scored_simulation_details = asyncio.run(
                                    _run_simulation_and_scoring(
                                        explanation_text=explanation,
                                        records_for_simulation=temp_activation_records,
                                        model_name_for_simulator=OPENAI_MODELS.get(args.simulator_model, args.simulator_model),
                                        few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
                                        prompt_format=PromptFormat.HARMONY_V4
                                    )
                                )
                                print(f"    Neuron {neuron_idx_item} - Score: {explanation_score}")
                            
                            # Store the explanation and score
                            key = f"{sae_pos}_neuron_{neuron_idx_item}"
                            explanations_for_run[key] = {
                                "text": explanation,
                                "score": explanation_score,
                                "sae_position": sae_pos,
                                "neuron_index": neuron_idx_item,
                                "num_examples": len(activation_records)
                            }
                            
                        else:
                            print(f"    No explanation generated for neuron {neuron_idx_item}")
                            
                    except Exception as e:
                        print(f"    Error processing neuron {neuron_idx_item}: {e}")
            
            # Save explanations to Wandb
            if explanations_for_run:
                try:
                    print(f"Saving explanations to Wandb for run {run_id}")
                    save_explanations_to_wandb(explanations=explanations_for_run)
                    print(f"Successfully uploaded explanations to Wandb for run {run_id}")
                    
                except Exception as e:
                    print(f"Warning: Failed to upload explanations to Wandb: {e}")
        
        # Finish the current wandb run before moving to the next one
        wandb.finish()

    # Create pareto plots after processing all runs
    # print(f"\nCreating pareto plots from {len(all_run_metrics)} runs...")
    create_pareto_plots(all_run_metrics)


def main():
    """Main function with argparse configuration."""
    parser = argparse.ArgumentParser(description="Run SAE evaluation with configurable parameters")
    
    # Window and processing parameters
    parser.add_argument("--window_size", type=int, default=64, 
                       help="Size of token windows for processing (default: 64)")
    parser.add_argument("--num_neurons", type=int, default=300,
                       help="Number of top neurons to process per layer (default: 300)")
    parser.add_argument("--min_activated_features", type=int, default=200,
                       help="Minimum number of activated features to use for explanation (default: 200)")
    parser.add_argument("--num_features_to_explain", type=int, default=10,
                       help="Number of top activation examples to use for explanation (default: 10)")
    parser.add_argument("--n_eval_samples", type=int, default=50000,
                       help="Number of evaluation samples to process (default: 50000)")
    
    # Model parameters
    parser.add_argument("--explanation_model", type=str, default="gpt-4o",
                       choices=["gpt-4o", "gpt-4o-mini"],
                       help="Model to use for generating explanations (default: gpt-4o)")
    parser.add_argument("--simulator_model", type=str, default="gpt-4o-mini",
                       choices=["gpt-4o", "gpt-4o-mini"],
                       help="Model to use for simulation and scoring (default: gpt-4o-mini)")
    
    # Wandb and storage parameters
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project in format 'entity/project' (default: raymondl/tinystories-1m)")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by a specific string in their name (default: None)")
    
    # Execution flags
    parser.add_argument("--save_activation_data", action="store_true", default=False,
                       help="Save activation data (default: False)")
    
    parser.add_argument("--generate_explanations", action="store_true", default=False,
                       help="Generate neuron explanations (default: False)")
    
    parser.add_argument("--evaluate_explanations", action="store_true",
                       help="Evaluate explanation quality (default: False)")
    
    parser.add_argument("--force_recompute", action="store_true", default=False,
                       help="Force recomputation of metrics even if existing ones are found (default: False)")

    # For debugging
    parser.add_argument("--override_n_train_samples", type=int, default=None,
                       help="Override n_train_samples to avoid slow data skipping (default: None - use config value)")
    parser.add_argument("--hard_concrete_method", type=str, default="c", choices=["c", "z"],
                       help="Method to use for Hard Concrete SAE (default: c)")
    
    args = parser.parse_args()
    
    # Run the evaluation with parsed arguments
    run_evaluation(args)


if __name__ == "__main__":
    main()
