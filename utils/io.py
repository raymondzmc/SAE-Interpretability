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


# File Upload Functions

def save_activation_data_to_wandb(
    accumulated_data: dict[str, dict[str, torch.Tensor]], 
    all_token_ids: list[list[str]] | None = None,
) -> None:
    """Save accumulated activation data directly to the current Wandb run's files directory.
    
    Args:
        accumulated_data: Dictionary mapping SAE positions to activation data
        all_token_ids: Optional list of token ID sequences to save alongside activation data
    """
    # Ensure we have an active run
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
    run_dir = Path(wandb.run.dir)

    activation_data_dir = run_dir / "activation_data"
    activation_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each layer's data to run dir
    for sae_pos, data in accumulated_data.items():
        safe_layer_name = sae_pos.replace(".", "--").replace("/", "--")
        file_path = activation_data_dir / f"{safe_layer_name}.pt"
        torch.save(data, file_path)
        print(f"Staged activation data for {sae_pos} at {file_path}")
    
    # Save token IDs if provided
    if all_token_ids is not None:
        token_ids_path = activation_data_dir / "all_token_ids.pt"
        torch.save(all_token_ids, token_ids_path)
        print(f"Staged token IDs at {token_ids_path}")
    
    # Upload activation data directory to current run (preserve activation_data/ prefix only)
    try:
        wandb.save(str(activation_data_dir / "*"), base_path=str(run_dir), policy="now")
        print(f"Successfully uploaded activation data to Wandb run files")
    except Exception as e:
        print(f"Warning: Failed to upload activation data to Wandb: {e}")


def save_metrics_to_wandb(
    metrics: dict[str, dict[str, Any]]
) -> None:
    """Save evaluation metrics directly to the current Wandb run's files directory.
    
    Args:
        metrics: Dictionary mapping SAE positions to their evaluation metrics
    """
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
    run_dir = Path(wandb.run.dir)

    # Save metrics as JSON at run root
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Staged metrics at {metrics_path}")
    
    # Upload metrics file to current run at root (no files/ prefix)
    try:
        wandb.save(str(metrics_path), base_path=str(run_dir), policy="now")
        print(f"Successfully uploaded metrics to Wandb run files")
    except Exception as e:
        print(f"Warning: Failed to upload metrics to Wandb: {e}")


# Local file functions removed - now only using Wandb artifacts


def load_activation_data_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m"
) -> tuple[dict[str, dict[str, torch.Tensor]], list[list[str]] | None]:
    """Load accumulated activation data and token IDs from Wandb run files.
    
    Args:
        run_id: The Wandb run ID to load files from
        project: Wandb project name
    
    Returns:
        Tuple of (activation_data_dict, all_token_ids)
        
    Raises:
        FileNotFoundError: If the run or files are not found
        RuntimeError: If the files exist but can't be loaded
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Get the run
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            raise FileNotFoundError(f"Run {run_id} not found in project {project}")
        
        # Download run files to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            accumulated_data = {}
            all_token_ids = None
            
            # Try to download activation_data directory
            try:
                for file in run.files():
                    if file.name.startswith("activation_data/") and file.name.endswith(".pt"):
                        local_path = temp_path / file.name
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        file.download(root=temp_dir, replace=True)
                        
                        filename = Path(file.name).name
                        
                        if filename == "all_token_ids.pt":
                            # Load token IDs
                            all_token_ids = torch.load(local_path, map_location='cpu')
                            print(f"Loaded token IDs from Wandb run files")
                        else:
                            # Convert filename back to original sae_pos format
                            safe_layer_name = filename[:-3]  # Remove .pt extension
                            sae_pos = safe_layer_name.replace("--", ".")  # Convert back from safe filename
                            
                            data = torch.load(local_path, map_location='cpu')  # Load to CPU first
                            accumulated_data[sae_pos] = data
                            
                            print(f"Loaded activation data for {sae_pos} from Wandb run files")
                            
            except Exception as e:
                print(f"Warning: Could not load activation data files: {e}")
            
            if not accumulated_data:
                raise FileNotFoundError(f"No activation data files found in run {run_id}")
            
            return accumulated_data, all_token_ids
            
    except FileNotFoundError:
        # Re-raise these specific exceptions
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading activation data from Wandb: {e}")


def load_metrics_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m"
) -> dict[str, dict[str, Any]] | None:
    """Load evaluation metrics from Wandb run files.
    
    Args:
        run_id: The Wandb run ID to load files from
        project: Wandb project name
    
    Returns:
        Dictionary mapping SAE positions to their evaluation metrics, or None if not found
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Get the run
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            print(f"Run {run_id} not found in project {project}")
            return None
        
        # Download metrics file to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Try to download metrics.json (root or nested under a folder like 'files/')
            try:
                for file in run.files():
                    if file.name == "metrics.json" or file.name.endswith("/metrics.json"):
                        local_path = temp_path / file.name
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        file.download(root=temp_dir, replace=True)
                        
                        with open(local_path, "r") as f:
                            metrics = json.load(f)
                        print(f"Loaded evaluation metrics from Wandb run files")
                        return metrics
                        
            except Exception as e:
                print(f"Warning: Could not load metrics file: {e}")
            
            print(f"No metrics.json file found in run {run_id}")
            return None
            
    except Exception as e:
        print(f"Error loading metrics from Wandb: {e}")
        return None


# Evaluation Results Functions

def save_explanations_to_wandb(
    explanations: dict[str, dict[str, Any]]
) -> None:
    """Save explanations directly to the current Wandb run's files directory.
    
    Args:
        explanations: Dictionary mapping neuron keys to explanation data
    """
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
    run_dir = Path(wandb.run.dir)

    # Save explanations at run root
    explanations_path = run_dir / "explanations.json"
    with open(explanations_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"Staged explanations at {explanations_path}")

    # Create summary statistics at run root
    summary_stats = {
        "num_explanations": len(explanations),
        "explained_neurons_per_layer": {}
    }
    for key in explanations.keys():
        if "_neuron_" in key:
            layer_name = key.split("_neuron_")[0]
            if layer_name not in summary_stats["explained_neurons_per_layer"]:
                summary_stats["explained_neurons_per_layer"][layer_name] = 0
            summary_stats["explained_neurons_per_layer"][layer_name] += 1

    summary_path = run_dir / "explanation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Staged explanation summary at {summary_path}")
    
    # Upload files to current run at root (no files/ prefix)
    try:
        wandb.save(str(explanations_path), base_path=str(run_dir), policy="now")
        wandb.save(str(summary_path), base_path=str(run_dir), policy="now")
        print(f"Successfully uploaded explanations to Wandb run files")
    except Exception as e:
        print(f"Warning: Failed to upload explanations to Wandb: {e}")


def load_explanations_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m"
) -> dict[str, dict[str, Any]] | None:
    """Load explanations from Wandb run files.
    
    Args:
        run_id: The Wandb run ID to load files from
        project: Wandb project name
    
    Returns:
        Dictionary mapping neuron keys to explanation data, or None if not found
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Get the run
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            print(f"Run {run_id} not found in project {project}")
            return None
        
        # Download explanations file to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Try to download explanations.json (root or nested under a folder like 'files/')
            try:
                for file in run.files():
                    if file.name == "explanations.json" or file.name.endswith("/explanations.json"):
                        local_path = temp_path / file.name
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        file.download(root=temp_dir, replace=True)
                        
                        with open(local_path, "r") as f:
                            explanations = json.load(f)
                        print(f"Loaded explanations from Wandb run files")
                        return explanations
                        
            except Exception as e:
                print(f"Warning: Could not load explanations file: {e}")
            
            print(f"No explanations.json file found in run {run_id}")
            return None
            
    except Exception as e:
        print(f"Error loading explanations from Wandb: {e}")
        return None
