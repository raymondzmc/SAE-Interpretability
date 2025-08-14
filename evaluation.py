import asyncio  # For running async explainer
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
import torch
import wandb
import argparse
from settings import settings
# from neuron_explainer.activations.activation_records import calculate_max_activation
# from neuron_explainer.activations.activations import ActivationRecord
# from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
# from neuron_explainer.explanations.explainer import (  # ContextSize if needed
#     PromptFormat,
#     TokenActivationPairExplainer,
# )
# from neuron_explainer.explanations.explanations import ScoredSimulation
# from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
# from neuron_explainer.explanations.scoring import simulate_and_score
# from neuron_explainer.explanations.simulator import LogprobFreeExplanationTokenSimulator
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
from auto_interp.explainers.features import FeatureRecord, Feature
from auto_interp.explainers.explainer import DefaultExplainer, ExplainerResult
from auto_interp.clients import OpenAIClient, TogetherAIClient

# Import scorers
from auto_interp.scorers.classifier.detection import DetectionScorer
from auto_interp.scorers.classifier.fuzz import FuzzingScorer
try:
    from auto_interp.scorers.embedding.embedding import EmbeddingScorer
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDING_SCORER = True
except ImportError:
    HAS_EMBEDDING_SCORER = False
    print("Warning: EmbeddingScorer not available (missing sentence-transformers)")


class MockTokenizer:
    """Mock tokenizer for when tokens are already strings."""
    is_mock = True
    
    def batch_decode(self, tokens):
        """Just return the tokens as they're already strings."""
        return tokens

# async def _run_simulation_and_scoring(
#     explanation_text: str,
#     records_for_simulation: Sequence[ActivationRecord],
#     model_name_for_simulator: str,
#     few_shot_example_set: FewShotExampleSet,
#     prompt_format: PromptFormat,
#     num_retries: int = 5,
# ) -> tuple[float | None, Any | None]:
#     """Helper to run simulation and scoring with retries."""
#     attempts = 0
#     score = None
#     scored_simulation = None
#     while attempts < num_retries:  # Retry loop
#         try:
#             simulator = UncalibratedNeuronSimulator(
#                 LogprobFreeExplanationTokenSimulator(
#                     model_name_for_simulator,
#                     explanation_text,
#                     json_mode=True,
#                     max_concurrent=10,
#                     few_shot_example_set=few_shot_example_set,
#                     prompt_format=prompt_format,
#                 )
#             )
#             scored_simulation: ScoredSimulation = await simulate_and_score(simulator, records_for_simulation)
#             score = scored_simulation.get_preferred_score() if scored_simulation else None

#         except Exception as e:
#             print(f"Error in attempt {attempts + 1}: {e}")
#             attempts += 1

#         if score is not None and not np.isnan(score):
#             break

#     return score, scored_simulation


def run_evaluation(args: argparse.Namespace) -> None:
    """
    Run SAE evaluation including activation data collection, neuron explanation generation, and analysis.
    """
    
    # Set Wandb cache directories to use output_path instead of home directory
    import os
    from pathlib import Path
    output_path_abs = Path(args.output_path).absolute()
    output_path_abs.mkdir(parents=True, exist_ok=True)
    
    wandb_cache_dir = output_path_abs / "wandb_cache"
    wandb_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables BEFORE any Wandb operations
    os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
    os.environ["WANDB_DATA_DIR"] = str(wandb_cache_dir)
    os.environ["WANDB_DIR"] = str(wandb_cache_dir)
    os.environ["TMPDIR"] = str(output_path_abs)  # Also set general temp directory
    
    print(f"Using Wandb cache directory: {wandb_cache_dir}")
    print(f"Using temp directory: {output_path_abs}")
    
    # OpenAI model mappings
    # OPENAI_MODELS = {
    #     "gpt-4o-mini": "gpt-4o-mini-07-18",
    #     "gpt-4o": "gpt-4o-2024-11-20",
    # }

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
    
    # Initialize list to store all explanation scores across runs
    all_explanation_scores = []

    for run in runs:
        run_id = run.id
        run_config = run.config
        run_config['data']['n_eval_samples'] = args.n_eval_samples
        model = SAETransformer.from_wandb(f"{args.wandb_project}/{run_id}").to(device)
        model.saes.eval()
        
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
            reinit="finish_previous",
            dir=str(wandb_cache_dir)  # Explicitly set Wandb's working directory
        )

        metrics = {}
        accumulated_data = None
        all_token_ids = None
        loaded_metrics = None
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)

        # Try to load existing data from Wandb
        print(f"Attempting to load activation data from Wandb for run {run_id}...")
        try:
            accumulated_data, all_token_ids = load_activation_data_from_wandb(
                run_id, project=args.wandb_project, output_path=args.output_path
            )
            
            print(f"Successfully loaded activation data from Wandb run files")
            if all_token_ids is not None:
                print(f"Loaded token IDs from Wandb run files")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No existing activation data found: {e}")
            print("Will compute activation data and metrics from scratch")
        
        # Try to load existing metrics separately
        try:
            loaded_metrics = load_metrics_from_wandb(run_id, project=args.wandb_project, output_path=args.output_path)
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

            # Load model and dataloader
            _, eval_loader = create_dataloaders(data_config=config.data, global_seed=config.seed)
            total_tokens = 0
            all_token_ids: list[list[str]] = []

            # Create placeholder tensors for efficient batch accumulation
            # Note: For feature extraction, 'nonzero_activations' stores probabilities for Bayesian SAEs, activations for ReLU SAEs
            accumulated_data = {}
            for sae_pos in model.raw_sae_positions:
                accumulated_data[sae_pos] = {
                    'nonzero_activations': [],
                    'data_indices': [],
                    'neuron_indices': [],
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
                        acts = sae_output.c
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
                            accumulated_data[sae_pos]['nonzero_activations'].append(nonzero_activations.to(torch.float16).cpu())
                            accumulated_data[sae_pos]['data_indices'].append(global_data_indices.cpu())
                            accumulated_data[sae_pos]['neuron_indices'].append(neuron_indices.cpu())

                # Store tokenized sequences for explanation generation
                chunked_tokens = [tokenizer.convert_ids_to_tokens(token_ids_chunked[i]) for i in range(chunked_batch_size)]
                all_token_ids.extend(chunked_tokens)

            for sae_pos in model.raw_sae_positions:
                if args.save_activation_data:
                    accumulated_data[sae_pos]['nonzero_activations'] = torch.cat(accumulated_data[sae_pos]['nonzero_activations'], dim=0).contiguous()
                    accumulated_data[sae_pos]['data_indices'] = torch.cat(accumulated_data[sae_pos]['data_indices'], dim=0).contiguous()
                    accumulated_data[sae_pos]['neuron_indices'] = torch.cat(accumulated_data[sae_pos]['neuron_indices'], dim=0).contiguous()

                metrics[sae_pos]['sparsity_l0'] /= total_tokens
                metrics[sae_pos]['mse'] /= total_tokens
                metrics[sae_pos]['explained_variance'] /= total_tokens
                
                # Convert alive components set to count and proportion
                alive_components = metrics[sae_pos]['alive_dict_components']
                num_alive = len(alive_components)
                total_dict_size = model.saes[sae_pos.replace(".", "-")].n_dict_components
                metrics[sae_pos]['alive_dict_components'] = num_alive
                metrics[sae_pos]['alive_dict_components_proportion'] = num_alive / total_dict_size
            
            # Always save metrics
            print("Saving metrics to Wandb...")
            try:
                save_metrics_to_wandb(metrics=metrics, output_path=args.output_path)
            except Exception as e:
                print(f"Warning: Failed to upload metrics to Wandb: {e}")

            # Save activation data to Wandb
            if args.save_activation_data:
                print("Saving accumulated activation data to Wandb...")
                save_activation_data_to_wandb(
                    accumulated_data=accumulated_data,
                    all_token_ids=all_token_ids,
                    output_path=args.output_path,
                    skip_upload=args.skip_upload,
                    chunk_upload=not args.no_chunk_upload  # Default is True unless disabled
                )

        # Collect metrics for pareto plot
        run_metrics = {
            'run_id': run_id,
            'run_name': run.name,
            'config': config,
            'metrics': metrics
        }
        
        all_run_metrics.append(run_metrics)
        
        if args.generate_explanations:
            explainer = DefaultExplainer(
                client=OpenAIClient(
                    api_key=settings.openai_api_key,
                    model=args.explanation_model  # Use the model specified in args
                ),
                tokenizer=tokenizer,
                cot=True,
                threshold=0.6,
            )
            
            # Skip embedding model loading to save time
            # # Load embedding model once if we're evaluating explanations
            # embedding_model = None
            # if args.evaluate_explanations and HAS_EMBEDDING_SCORER:
            #     print("Loading Stella embedding model for evaluation...")
            #     embedding_model = SentenceTransformer('NovaSearch/stella_en_400M_v5', trust_remote_code=True)
            #     print("✓ Loaded NovaSearch/stella_en_400M_v5 model")
            
            # Filter neurons with sufficient examples and construct activation records
            # explanations_for_run = {}

            for sae_pos in model.raw_sae_positions:
                data = accumulated_data[sae_pos]
                
                # Count occurrences of each neuron and calculate total activation
                unique_neurons, counts = torch.unique(data['neuron_indices'], return_counts=True)
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
                neurons_to_explain = unique_neurons[top_neuron_indices]
                
                print(f"SAE position {sae_pos}: {len(unique_neurons)} total neurons, taking top {args.num_neurons}")
                print(f"  Processing {len(neurons_to_explain)} neurons for explanation...")
                
            #     # Process each neuron for explanation
                for neuron_idx in neurons_to_explain:
                    neuron_idx_item = neuron_idx.item()
                    feature = Feature(
                        sae_pos=sae_pos,
                        neuron_idx=neuron_idx_item,
                    )
                    # Use the new from_data class method for cleaner sampling
                    feature_record = FeatureRecord.from_data(
                        data=data,
                        feature=feature,
                        all_token_ids=all_token_ids,
                        neuron_idx=neuron_idx,
                        num_explanation_examples=args.num_features_to_explain,
                        num_positive_examples=100,  # For scoring
                        num_negative_examples=100,  # For scoring
                        stratified_quantiles=args.stratified_quantiles,
                        min_examples_required=args.min_activated_features_per_neuron,
                        seed=42,
                    )
                    
                    # Skip if we couldn't create a valid feature record
                    if feature_record is None:
                        print(f"  Skipping neuron {neuron_idx_item} - not enough examples")
                        continue
                    # Generate explanation using the explanation_examples
                    explanation: ExplainerResult = explainer(feature_record)

                    # Score the explanation if requested
                    if args.evaluate_explanations:
                        print(f"  Neuron {neuron_idx_item}: {explanation.explanation}")
                        
                        # Create mock tokenizer since tokens are already strings
                        mock_tokenizer = MockTokenizer()
                        
                        # Create scoring client for Detection and Fuzz scorers
                        score_client = TogetherAIClient(
                            api_key=settings.together_ai_api_key,  # Use together API key
                            model=args.scoring_model
                        )
                        
                        # 1. Detection Score
                        print(f"    Computing Detection score...")
                        detection_scorer = DetectionScorer(
                            client=score_client,
                            tokenizer=mock_tokenizer,
                            verbose=False,
                            batch_size=10,
                        )
                        detection_result = asyncio.run(detection_scorer(explanation))
                        import pdb; pdb.set_trace()
                        # Calculate accuracy from the classifier outputs
                        detection_outputs = detection_result.score
                        correct_predictions = sum(1 for output in detection_outputs if output.correct)
                        detection_score = correct_predictions / len(detection_outputs) if detection_outputs else 0
                        print(f"    ✓ Detection score: {detection_score:.3f} ({correct_predictions}/{len(detection_outputs)} correct)")
                        
                        # 2. Fuzz Score
                        print(f"    Computing Fuzz score...")
                        fuzz_scorer = FuzzingScorer(
                            client=score_client,
                            tokenizer=mock_tokenizer,
                            verbose=False,
                            batch_size=1,
                            threshold=0.3,
                        )
                        fuzz_result = asyncio.run(fuzz_scorer(explanation))
                        # Calculate accuracy from the classifier outputs
                        fuzz_outputs = fuzz_result.score
                        correct_fuzz = sum(1 for output in fuzz_outputs if output.correct)
                        fuzz_score = correct_fuzz / len(fuzz_outputs) if fuzz_outputs else 0
                        print(f"    ✓ Fuzz score: {fuzz_score:.3f} ({correct_fuzz}/{len(fuzz_outputs)} correct)")
                        
                        # 3. Embedding Score - Skipped to save time
                        embedding_score = None
                        # if HAS_EMBEDDING_SCORER and embedding_model is not None:
                        #     print(f"    Computing Embedding score...")
                        #     
                        #     embedding_scorer = EmbeddingScorer(
                        #         model=embedding_model,
                        #         tokenizer=mock_tokenizer,
                        #         verbose=False,
                        #         batch_size=10,
                        #     )
                        #     embedding_result = asyncio.run(embedding_scorer(explanation))
                        #     
                        #     # Calculate average similarity for positive and negative examples
                        #     pos_similarities = []
                        #     neg_similarities = []
                        #     for sample, output in zip(embedding_result.record.positive_examples + embedding_result.record.negative_examples, 
                        #                             embedding_result.score):
                        #         if sample in embedding_result.record.positive_examples:
                        #             pos_similarities.append(output.similarity)
                        #         else:
                        #             neg_similarities.append(output.similarity)
                        #     
                        #     if pos_similarities and neg_similarities:
                        #         avg_pos_sim = sum(pos_similarities) / len(pos_similarities)
                        #         avg_neg_sim = sum(neg_similarities) / len(neg_similarities)
                        #         # Score is the difference between positive and negative similarities
                        #         embedding_score = avg_pos_sim - avg_neg_sim
                        #         print(f"    ✓ Embedding score: {embedding_score:.4f} (pos: {avg_pos_sim:.4f}, neg: {avg_neg_sim:.4f})")
                        #     else:
                        #         print(f"    ⚠️ Could not compute embedding score")
                        # else:
                        #     print(f"    ⚠️ Embedding scorer not available")
                        
                        # Store scores
                        all_explanation_scores.append({
                            'sae_pos': sae_pos,
                            'neuron_idx': neuron_idx_item,
                            'explanation': explanation.explanation,
                            'detection_score': detection_score,
                            'fuzz_score': fuzz_score,
                            'embedding_score': embedding_score,  # Will be None
                        })
                    else:
                        print(f"  Neuron {neuron_idx_item}: {explanation.explanation}")
            
            # # Save explanations to Wandb
            # if explanations_for_run:
            #     try:
            #         print(f"Saving explanations to Wandb for run {run_id}")
            #         save_explanations_to_wandb(explanations=explanations_for_run)
            #         print(f"Successfully uploaded explanations to Wandb for run {run_id}")
                    
            #     except Exception as e:
            #         print(f"Warning: Failed to upload explanations to Wandb: {e}")
        
        # Finish the current wandb run before moving to the next one
        wandb.finish()

    # Save and display scores if they were computed
    if args.evaluate_explanations and all_explanation_scores:
        print("\n" + "=" * 60)
        print("EXPLANATION SCORES SUMMARY")
        print("=" * 60)
        
        # Save scores to JSON
        import json
        scores_path = Path(args.output_path) / "explanation_scores.json"
        with open(scores_path, 'w') as f:
            json.dump(all_explanation_scores, f, indent=2)
        print(f"\nScores saved to: {scores_path}")
        
        # Display summary statistics
        if all_explanation_scores:
            detection_scores = [s['detection_score'] for s in all_explanation_scores if s['detection_score'] is not None]
            fuzz_scores = [s['fuzz_score'] for s in all_explanation_scores if s['fuzz_score'] is not None]
            # embedding_scores = [s['embedding_score'] for s in all_explanation_scores if s['embedding_score'] is not None]
            
            print("\nScore Statistics:")
            if detection_scores:
                print(f"  Detection: mean={np.mean(detection_scores):.3f}, std={np.std(detection_scores):.3f}, n={len(detection_scores)}")
            if fuzz_scores:
                print(f"  Fuzz:      mean={np.mean(fuzz_scores):.3f}, std={np.std(fuzz_scores):.3f}, n={len(fuzz_scores)}")
            # Embedding score skipped to save time
            # if embedding_scores:
            #     print(f"  Embedding: mean={np.mean(embedding_scores):.3f}, std={np.std(embedding_scores):.3f}, n={len(embedding_scores)}")
            
            # Display top scoring explanations
            print("\nTop 5 explanations by Detection score:")
            sorted_by_detection = sorted([s for s in all_explanation_scores if s['detection_score'] is not None], 
                                        key=lambda x: x['detection_score'], reverse=True)[:5]
            for i, score_entry in enumerate(sorted_by_detection, 1):
                print(f"  {i}. Neuron {score_entry['neuron_idx']} ({score_entry['sae_pos']}): {score_entry['detection_score']:.3f}")
                print(f"     {score_entry['explanation'][:100]}...")

    # Create pareto plots after processing all runs
    # print(f"\nCreating pareto plots from {len(all_run_metrics)} runs...")
    create_pareto_plots(all_run_metrics)


def main():
    """Main function with argparse configuration."""
    parser = argparse.ArgumentParser(description="Run SAE evaluation with configurable parameters")
    
    # Window and processing parameters
    parser.add_argument("--window_size", type=int, default=64, 
                       help="Size of token windows for processing (default: 64)")

    # Explanation parameters
    parser.add_argument("--num_neurons", type=int, default=300,
                       help="Number of top neurons to process per layer (default: 300)")
    parser.add_argument("--min_activated_features_per_neuron", type=int, default=100,
                       help="Minimum number of activated features to use for explanation per neuron (default: 100)")
    parser.add_argument("--max_activated_features_per_neuron", type=int, default=10000,
                       help="Maximum number of activated features to use for explanation per neuron (default: 1000)")
    parser.add_argument("--num_features_to_explain", type=int, default=10,
                       help="Number of top activation examples to use for explanation (default: 10)")
    parser.add_argument("--n_eval_samples", type=int, default=50000,
                       help="Number of evaluation samples to process (default: 50000)")
    parser.add_argument("--stratified_quantiles", type=int, default=20,
                       help="Number of quantiles for stratified sampling of activation examples (default: 4)")
    
    # Model parameters
    parser.add_argument("--explanation_model", type=str, default="gpt-4o",
                       help="Model to use for generating explanations (default: gpt-4o)")
    parser.add_argument("--scoring_model", type=str, default="llama-3.3-70b",
                       help="Model to use for scoring explanations (default: llama-3.3-70b)")
    
    # Wandb and storage parameters
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project in format 'entity/project' (default: raymondl/tinystories-1m)")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by a specific string in their name (default: None)")
    parser.add_argument("--output_path", type=str, default="./artifacts",
                       help="Path for storing temporary files and artifacts (default: ./artifacts)")
    
    # Execution flags
    parser.add_argument("--save_activation_data", action="store_true", default=False,
                       help="Save activation data (default: False)")
    
    parser.add_argument("--skip_upload", action="store_true", default=False,
                       help="Skip uploading to Wandb, only save locally (default: False)")
    
    parser.add_argument("--no_chunk_upload", action="store_true", default=False,
                       help="Disable chunked upload (upload all files at once) (default: False)")
    
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
