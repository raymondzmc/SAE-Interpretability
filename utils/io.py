import yaml
from pathlib import Path
from typing import Any
import os
import tempfile
import json

import torch
import wandb
from torch import nn
from utils.logging import logger
from utils.types import BaseModelType
from utils.constants import CONFIG_FILE


def save_module(
    config_dict: dict[str, Any],
    save_dir: Path,
    module: nn.Module,
    model_filename: str,
    config_filename: str = CONFIG_FILE,
) -> None:
    """Save the pytorch module and config to the save_dir.

    The config will only be saved if the save_dir doesn't exist (i.e. the first time the module is
    saved assuming the save_dir is unique to the module).

    Args:
        config_dict: Dictionary representation of the config to save.
        save_dir: Directory to save the module.
        module: The module to save.
        model_filename: The filename to save the model to.
        config_filename: The filename to save the config to.
    """
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        with open(save_dir / config_filename, "w") as f:
            yaml.dump(config_dict, f)
        logger.info("Saved config to %s", save_dir / config_filename)

    torch.save(module.state_dict(), save_dir / model_filename)
    logger.info("Saved model to %s", save_dir / model_filename)


def load_config(config_path_or_obj: Path | str | BaseModelType | dict[str, Any], config_model: type[BaseModelType]) -> BaseModelType:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)
    
    if isinstance(config_path_or_obj, dict):
        config_dict = config_path_or_obj
    else:
        assert isinstance(
            config_path_or_obj, Path
        ), f"passed config is of invalid type {type(config_path_or_obj)}"
        assert (
            config_path_or_obj.suffix == ".yaml"
        ), f"Config file {config_path_or_obj} must be a YAML file."
        assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
        with open(config_path_or_obj) as f:
            config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


# Activation Data Functions

def save_activation_data_to_wandb(
    accumulated_data: dict[str, dict[str, torch.Tensor]], 
    run_id: str,
    run_name: str,
    all_token_ids: list[list[str]] | None = None,
    metrics: dict[str, dict[str, Any]] | None = None,
    artifact_name: str = "activation_data"
) -> None:
    """Save accumulated activation data and metrics to Wandb as artifacts for a specific run.
    
    Args:
        accumulated_data: Dictionary mapping SAE positions to activation data
        run_id: The Wandb run ID to associate the artifact with
        run_name: Human-readable run name for artifact metadata
        all_token_ids: Optional list of token ID sequences to save alongside activation data
        metrics: Optional dictionary mapping SAE positions to their evaluation metrics
        artifact_name: Name for the Wandb artifact (default: "activation_data")
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save each layer's data to temporary files
        for sae_pos, data in accumulated_data.items():
            # Replace dots and other characters that might be problematic in filenames
            safe_layer_name = sae_pos.replace(".", "--").replace("/", "--")
            file_path = temp_path / f"{safe_layer_name}.pt"
            torch.save(data, file_path)
            print(f"Staged activation data for {sae_pos} at {file_path}")
        
        # Save token IDs if provided
        if all_token_ids is not None:
            token_ids_path = temp_path / "all_token_ids.pt"
            torch.save(all_token_ids, token_ids_path)
            print(f"Staged token IDs at {token_ids_path}")
        
        # Save metrics if provided
        if metrics is not None:
            metrics_path = temp_path / "evaluation_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Staged metrics at {metrics_path}")
        
        # Create Wandb artifact
        artifact = wandb.Artifact(
            name=f"{artifact_name}_{run_id}",
            type="activation_data",
            description=f"SAE activation data for run {run_name} ({run_id})",
            metadata={
                "run_id": run_id,
                "run_name": run_name,
                "num_layers": len(accumulated_data),
                "layer_names": list(accumulated_data.keys()),
                "has_token_ids": all_token_ids is not None,
                "has_metrics": metrics is not None
            }
        )
        
        # Add all files to the artifact
        artifact.add_dir(temp_path, name="activation_data")
        
        # Log the artifact
        try:
            wandb.log_artifact(artifact)
            print(f"Successfully uploaded activation data to Wandb artifact: {artifact.name}")
        except Exception as e:
            print(f"Warning: Failed to upload activation data to Wandb: {e}")
            print("Continuing with local save only...")


def save_activation_data(
    accumulated_data: dict[str, dict[str, torch.Tensor]], 
    run_dir: str,
    metrics: dict[str, dict[str, Any]] | None = None,
    all_token_ids: list[list[str]] | None = None,
    upload_to_wandb: bool = False,
    run_id: str | None = None,
    run_name: str | None = None
) -> None:
    """Save accumulated activation data, metrics, and token IDs for a run.
    
    Note: For feature extraction, Bayesian SAEs use Hard Concrete gate probabilities while ReLU SAEs use activations.
    However, metrics (L0 sparsity, MSE, explained variance) are computed using actual activations for both types.
    
    Args:
        accumulated_data: Dictionary mapping SAE positions to activation data
        run_dir: Local directory to save the data
        metrics: Optional dictionary mapping SAE positions to their evaluation metrics
        all_token_ids: Optional list of token ID sequences to save alongside activation data
        upload_to_wandb: Whether to also upload the data to Wandb as artifacts
        run_id: Required if upload_to_wandb is True - the Wandb run ID
        run_name: Required if upload_to_wandb is True - human-readable run name
    """
    # Create run directory if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)
    
    # Save activation data
    activation_data_dir = os.path.join(run_dir, "activation_data")
    os.makedirs(activation_data_dir, exist_ok=True)
    
    for sae_pos, data in accumulated_data.items():
        # Replace dots and other characters that might be problematic in filenames
        safe_layer_name = sae_pos.replace(".", "--").replace("/", "--")
        file_path = os.path.join(activation_data_dir, f"{safe_layer_name}.pt")
        torch.save(data, file_path)
        print(f"Saved activation data for {sae_pos} to {file_path}")
    
    # Save token IDs if provided
    if all_token_ids is not None:
        token_ids_path = os.path.join(run_dir, "all_token_ids.pt")
        torch.save(all_token_ids, token_ids_path)
        print(f"Saved token IDs to {token_ids_path}")
    
    # Save metrics if provided
    if metrics is not None:
        metrics_file = os.path.join(run_dir, "evaluation_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to {metrics_file}")
    
    # Optionally upload to Wandb
    if upload_to_wandb:
        if run_id is None or run_name is None:
            print("Warning: run_id and run_name are required for Wandb upload. Skipping Wandb upload.")
        else:
            save_activation_data_to_wandb(accumulated_data, run_id, run_name, all_token_ids, metrics)


def load_activation_data(run_dir: str) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]] | None, list[list[str]] | None]:
    """Load accumulated activation data, metrics, and token IDs for a specific run from local files.
    
    Args:
        run_dir: Directory containing the run data
        
    Returns:
        Tuple of (activation_data, metrics, all_token_ids)
        - activation_data: Dictionary mapping SAE positions to activation data
        - metrics: Dictionary mapping SAE positions to metrics, or None if not found
        - all_token_ids: List of token sequences, or None if not found
    """
    activation_data_dir = os.path.join(run_dir, "activation_data")
    
    if not os.path.exists(activation_data_dir):
        raise FileNotFoundError(f"Activation data directory not found: {activation_data_dir}")
    
    # Load activation data
    accumulated_data = {}
    for filename in os.listdir(activation_data_dir):
        if filename.endswith(".pt"):
            # Convert filename back to original sae_pos format
            safe_layer_name = filename[:-3]  # Remove .pt extension
            sae_pos = safe_layer_name.replace("--", ".")  # Convert back from safe filename

            file_path = os.path.join(activation_data_dir, filename)
            data = torch.load(file_path)
            accumulated_data[sae_pos] = data
            
            print(f"Loaded activation data for {sae_pos} from {file_path}")
    
    # Load metrics if available
    metrics = None
    metrics_file = os.path.join(run_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        print(f"Loaded evaluation metrics from {metrics_file}")
    
    # Load token IDs if available
    all_token_ids = None
    token_ids_path = os.path.join(run_dir, "all_token_ids.pt")
    if os.path.exists(token_ids_path):
        all_token_ids = torch.load(token_ids_path)
        print(f"Loaded token IDs from {token_ids_path}")
    
    return accumulated_data, metrics, all_token_ids


def load_activation_data_from_wandb(
    run_id: str,
    artifact_name: str = "activation_data",
    project: str = "raymondl/tinystories-1m"
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]] | None, list[list[str]] | None]:
    """Load accumulated activation data, metrics, and token IDs from Wandb artifacts for a specific run.
    
    Args:
        run_id: The Wandb run ID to load artifacts from
        artifact_name: Name of the Wandb artifact (default: "activation_data")
        project: Wandb project name
    
    Returns:
        Tuple of (activation_data_dict, metrics, all_token_ids)
        
    Raises:
        FileNotFoundError: If the artifact is not found
        RuntimeError: If the artifact exists but doesn't contain expected data
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Try to find the artifact
        artifact_full_name = f"{artifact_name}_{run_id}"
        try:
            artifact = api.artifact(f"{project}/{artifact_full_name}:latest")
        except wandb.errors.CommError:
            raise FileNotFoundError(f"Artifact {artifact_full_name} not found in project {project}")
        
        # Download the artifact to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(root=temp_dir)
            activation_data_dir = os.path.join(artifact_dir, "activation_data")
            
            if not os.path.exists(activation_data_dir):
                raise RuntimeError(f"No activation_data directory found in artifact {artifact_full_name}")
            
            accumulated_data = {}
            all_token_ids = None
            metrics = None
            
            # Load all .pt files from the artifact
            for filename in os.listdir(activation_data_dir):
                if filename.endswith(".pt"):
                    file_path = os.path.join(activation_data_dir, filename)
                    
                    if filename == "all_token_ids.pt":
                        # Load token IDs
                        all_token_ids = torch.load(file_path, map_location='cpu')
                        print(f"Loaded token IDs from Wandb artifact")
                    else:
                        # Convert filename back to original sae_pos format
                        safe_layer_name = filename[:-3]  # Remove .pt extension
                        sae_pos = safe_layer_name.replace("--", ".")  # Convert back from safe filename
                        
                        data = torch.load(file_path, map_location='cpu')  # Load to CPU first
                        accumulated_data[sae_pos] = data
                        
                        print(f"Loaded activation data for {sae_pos} from Wandb artifact")
            
            # Load metrics if available
            metrics_path = os.path.join(artifact_dir, "activation_data", "evaluation_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                print(f"Loaded evaluation metrics from Wandb artifact")
            
            return accumulated_data, metrics, all_token_ids
            
    except (FileNotFoundError, RuntimeError):
        # Re-raise these specific exceptions
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading activation data from Wandb: {e}")


# Evaluation Results Functions

def save_evaluation_results_to_wandb(
    metrics: dict[str, dict[str, Any]],
    explanations: dict[str, dict[str, Any]],
    run_id: str,
    run_name: str,
    artifact_name: str = "evaluation_results"
) -> None:
    """Save evaluation results (metrics + explanations) to Wandb as artifacts.
    
    Args:
        metrics: Dictionary mapping SAE positions to their evaluation metrics
        explanations: Dictionary mapping neuron keys to explanation data
        run_id: The Wandb run ID to associate the artifact with
        run_name: Human-readable run name for artifact metadata
        artifact_name: Name for the Wandb artifact (default: "evaluation_results")
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save metrics as JSON
        metrics_path = temp_path / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Staged evaluation metrics at {metrics_path}")
        
        # Save explanations as JSON
        explanations_path = temp_path / "explanations.json"
        with open(explanations_path, "w") as f:
            json.dump(explanations, f, indent=2)
        print(f"Staged explanations at {explanations_path}")
        
        # Create summary statistics
        summary_stats = {
            "num_layers": len(metrics),
            "layer_names": list(metrics.keys()),
            "num_explanations": len(explanations),
            "explained_neurons_per_layer": {}
        }
        
        # Count explanations per layer
        for key in explanations.keys():
            if "_neuron_" in key:
                layer_name = key.split("_neuron_")[0]
                if layer_name not in summary_stats["explained_neurons_per_layer"]:
                    summary_stats["explained_neurons_per_layer"][layer_name] = 0
                summary_stats["explained_neurons_per_layer"][layer_name] += 1
        
        summary_path = temp_path / "summary_stats.json"
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Staged summary statistics at {summary_path}")
        
        # Create Wandb artifact
        artifact = wandb.Artifact(
            name=f"{artifact_name}_{run_id}",
            type="evaluation_results",
            description=f"SAE evaluation results (metrics + explanations) for run {run_name} ({run_id})",
            metadata={
                "run_id": run_id,
                "run_name": run_name,
                **summary_stats
            }
        )
        
        # Add all files to the artifact
        artifact.add_dir(temp_path, name="evaluation_results")
        
        # Log the artifact
        try:
            wandb.log_artifact(artifact)
            print(f"Successfully uploaded evaluation results to Wandb artifact: {artifact.name}")
        except Exception as e:
            print(f"Warning: Failed to upload evaluation results to Wandb: {e}")
            print("Continuing with local save only...")


def load_evaluation_results_from_wandb(
    run_id: str,
    artifact_name: str = "evaluation_results",
    project: str = "raymondl/tinystories-1m"
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]] | tuple[None, None]:
    """Load evaluation results from Wandb artifacts for a specific run.
    
    Args:
        run_id: The Wandb run ID to load artifacts from
        artifact_name: Name of the Wandb artifact (default: "evaluation_results")
        project: Wandb project name
    
    Returns:
        Tuple of (metrics_dict, explanations_dict) or (None, None) if not found
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Try to find the artifact
        artifact_full_name = f"{artifact_name}_{run_id}"
        try:
            artifact = api.artifact(f"{project}/{artifact_full_name}:latest")
        except wandb.errors.CommError:
            print(f"Artifact {artifact_full_name} not found in project {project}")
            return None, None
        
        # Download the artifact to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(root=temp_dir)
            results_dir = os.path.join(artifact_dir, "evaluation_results")
            
            if not os.path.exists(results_dir):
                print(f"No evaluation_results directory found in artifact {artifact_full_name}")
                return None, None
            
            metrics = None
            explanations = None
            
            # Load metrics
            metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                print(f"Loaded evaluation metrics from Wandb artifact")
            
            # Load explanations
            explanations_path = os.path.join(results_dir, "explanations.json")
            if os.path.exists(explanations_path):
                with open(explanations_path, "r") as f:
                    explanations = json.load(f)
                print(f"Loaded explanations from Wandb artifact")
            
            return metrics, explanations
            
    except Exception as e:
        print(f"Error loading evaluation results from Wandb: {e}")
        return None, None
