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
    """Save accumulated activation data as Wandb artifacts to the current run.
    
    Args:
        accumulated_data: Dictionary mapping SAE positions to activation data
        all_token_ids: Optional list of token ID sequences to save alongside activation data
    """
    # Ensure we have an active run
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."

    # Create temporary directory for staging files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        activation_data_dir = temp_path / "activation_data"
        activation_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each layer's data to temp files
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
        
        try:
            # Create artifact name with run name and ID for easy identification
            run_name = wandb.run.name or "unnamed"
            run_id = wandb.run.id
            # Clean run name for artifact naming (replace invalid characters)
            clean_run_name = run_name.replace("/", "-").replace(":", "-").replace(" ", "_")
            artifact_name = f"evaluation_activation_data_{clean_run_name}_{run_id}"
            
            # Create artifact for activation data with alias to override existing
            artifact = wandb.Artifact(
                name=artifact_name,
                type="activation_data",
                description=f"SAE activation data for run {run_name} ({run_id})"
            )
            # Add all files in the activation_data directory
            artifact.add_dir(str(activation_data_dir), name="activation_data")
            
            # Log the artifact with "latest" alias to override previous versions
            wandb.log_artifact(artifact, aliases=["latest"])
            print(f"Successfully uploaded activation data as Wandb artifact: {artifact_name}")
            
        except Exception as e:
            print(f"Warning: Failed to upload activation data artifact to Wandb: {e}")


def save_metrics_to_wandb(
    metrics: dict[str, dict[str, Any]]
) -> None:
    """Save evaluation metrics as a Wandb artifact to the current run.
    
    Args:
        metrics: Dictionary mapping SAE positions to their evaluation metrics
    """
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
    
    # Use a temporary file for metrics
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='metrics_') as f:
        json.dump(metrics, f, indent=2)
        temp_metrics_path = f.name
    
    print(f"Staged metrics at {temp_metrics_path}")
    
    try:
        # Create artifact name with run name and ID for easy identification
        run_name = wandb.run.name or "unnamed"
        run_id = wandb.run.id
        # Clean run name for artifact naming (replace invalid characters)
        clean_run_name = run_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        artifact_name = f"evaluation_metrics_{clean_run_name}_{run_id}"
        
        # Create artifact for metrics with alias to override existing
        artifact = wandb.Artifact(
            name=artifact_name,
            type="metrics",
            description=f"SAE evaluation metrics for run {run_name} ({run_id})"
        )
        artifact.add_file(temp_metrics_path, name="metrics.json")
        
        # Log the artifact with "latest" alias to override previous versions
        wandb.log_artifact(artifact, aliases=["latest"])
        print(f"Successfully uploaded metrics as Wandb artifact: {artifact_name}")
        
    except Exception as e:
        print(f"Warning: Failed to upload metrics artifact to Wandb: {e}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_metrics_path)
        except:
            pass


# Local file functions removed - now only using Wandb artifacts


def load_activation_data_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m"
) -> tuple[dict[str, dict[str, torch.Tensor]], list[list[str]] | None]:
    """Load accumulated activation data and token IDs from Wandb artifacts.
    
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
        
        # Load from artifacts
        try:
            # Get all logged artifacts from this run
            artifacts = list(run.logged_artifacts())
            activation_artifacts = [a for a in artifacts if a.type == "activation_data" and "evaluation_activation_data" in a.name]
            
            if activation_artifacts:
                # Use the latest activation data artifact (highest version)
                latest_artifact = max(activation_artifacts, key=lambda x: x.version)
                artifact_dir = latest_artifact.download()
                
                accumulated_data = {}
                all_token_ids = None
                
                activation_data_path = Path(artifact_dir) / "activation_data"
                if activation_data_path.exists():
                    # Load activation data files
                    for file_path in activation_data_path.glob("*.pt"):
                        filename = file_path.name
                        
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
                    
                    if accumulated_data:
                        print(f"Loaded activation data from Wandb artifact: {latest_artifact.name} (v{latest_artifact.version})")
                        return accumulated_data, all_token_ids
                    
        except Exception as e:
            print(f"Could not load activation data from artifacts: {e}")
        
        raise FileNotFoundError(f"No activation data files found in run {run_id}")
            
    except FileNotFoundError:
        # Re-raise these specific exceptions
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading activation data from Wandb: {e}")


def load_metrics_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m"
) -> dict[str, dict[str, Any]] | None:
    """Load evaluation metrics from Wandb artifacts.
    
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
        
        # Load from artifacts
        try:
            # Get all logged artifacts from this run
            artifacts = list(run.logged_artifacts())
            metrics_artifacts = [a for a in artifacts if a.type == "metrics" and "evaluation_metrics" in a.name]
            
            if metrics_artifacts:
                # Use the latest metrics artifact (highest version)
                latest_artifact = max(metrics_artifacts, key=lambda x: x.version)
                artifact_dir = latest_artifact.download()
                
                metrics_path = Path(artifact_dir) / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    print(f"Loaded evaluation metrics from Wandb artifact: {latest_artifact.name} (v{latest_artifact.version})")
                    return metrics
                    
        except Exception as e:
            print(f"Could not load metrics from artifacts: {e}")
        
        print(f"No metrics found in run {run_id}")
        return None
            
    except Exception as e:
        print(f"Error loading metrics from Wandb: {e}")
        return None


# Evaluation Results Functions

def save_explanations_to_wandb(
    explanations: dict[str, dict[str, Any]]
) -> None:
    """Save explanations as Wandb artifacts to the current run.
    
    Args:
        explanations: Dictionary mapping neuron keys to explanation data
    """
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."

    # Use temporary files for explanations and summary
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='explanations_') as f:
        json.dump(explanations, f, indent=2)
        temp_explanations_path = f.name

    # Create summary statistics
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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='explanation_summary_') as f:
        json.dump(summary_stats, f, indent=2)
        temp_summary_path = f.name

    print(f"Staged explanations at {temp_explanations_path}")
    print(f"Staged explanation summary at {temp_summary_path}")
    
    try:
        # Create artifact name with run name and ID for easy identification
        run_name = wandb.run.name or "unnamed"
        run_id = wandb.run.id
        # Clean run name for artifact naming (replace invalid characters)
        clean_run_name = run_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        artifact_name = f"evaluation_explanations_{clean_run_name}_{run_id}"
        
        # Create artifact for explanations with alias to override existing
        artifact = wandb.Artifact(
            name=artifact_name,
            type="explanations",
            description=f"Neuron explanations for run {run_name} ({run_id})"
        )
        artifact.add_file(temp_explanations_path, name="explanations.json")
        artifact.add_file(temp_summary_path, name="explanation_summary.json")
        
        # Log the artifact with "latest" alias to override previous versions
        wandb.log_artifact(artifact, aliases=["latest"])
        print(f"Successfully uploaded explanations as Wandb artifact: {artifact_name}")
        
    except Exception as e:
        print(f"Warning: Failed to upload explanations artifact to Wandb: {e}")
    finally:
        # Clean up temp files
        try:
            os.unlink(temp_explanations_path)
            os.unlink(temp_summary_path)
        except:
            pass


def load_explanations_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m"
) -> dict[str, dict[str, Any]] | None:
    """Load explanations from Wandb artifacts.
    
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
        
        # Load from artifacts
        try:
            # Get all logged artifacts from this run
            artifacts = list(run.logged_artifacts())
            explanations_artifacts = [a for a in artifacts if a.type == "explanations" and "evaluation_explanations" in a.name]
            
            if explanations_artifacts:
                # Use the latest explanations artifact (highest version)
                latest_artifact = max(explanations_artifacts, key=lambda x: x.version)
                artifact_dir = latest_artifact.download()
                
                explanations_path = Path(artifact_dir) / "explanations.json"
                if explanations_path.exists():
                    with open(explanations_path, "r") as f:
                        explanations = json.load(f)
                    print(f"Loaded explanations from Wandb artifact: {latest_artifact.name} (v{latest_artifact.version})")
                    return explanations
                    
        except Exception as e:
            print(f"Could not load explanations from artifacts: {e}")
        
        print(f"No explanations found in run {run_id}")
        return None
            
    except Exception as e:
        print(f"Error loading explanations from Wandb: {e}")
        return None
