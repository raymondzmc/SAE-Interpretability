#!/usr/bin/env python3
"""
Setup Wandb sweeps for SAE hyperparameter optimization.

Usage:
    python scripts/setup_wandb_sweep.py --sae_type relu --project tinystories-sweeps
"""

import argparse
import yaml
import wandb
from pathlib import Path
from typing import Dict, Any
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sweep_config(sae_type: str, base_config_path: Path) -> Dict[str, Any]:
    """Create a Wandb sweep configuration for the given SAE type."""
    
    # Load base config to get fixed parameters
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter search spaces for each SAE type
    param_configs = {
        "relu": {
            "parameters": {
                "lr": {"values": [1e-4, 3e-4, 1e-3]},
                "saes.dict_size_to_input_ratio": {"values": [8.0, 16.0, 32.0, 64.0]},
                "saes.sparsity_coeff": {"values": [0.01, 0.03, 0.1, 0.3]},
            }
        },
        "hard_concrete": {
            "parameters": {
                "lr": {"values": [1e-4, 3e-4, 1e-3]},
                "saes.dict_size_to_input_ratio": {"values": [8.0, 16.0, 32.0, 64.0]},
                "saes.sparsity_coeff": {"values": [0.01, 0.03, 0.1, 0.3]},
                "saes.input_dependent_gates": {"values": [True, False]},
                "saes.initial_beta": {"values": [0.1, 0.5, 1.0]},
            }
        },
        "gated": {
            "parameters": {
                "lr": {"values": [1e-4, 3e-4, 1e-3]},
                "saes.dict_size_to_input_ratio": {"values": [8.0, 16.0, 32.0, 64.0]},
                "saes.sparsity_coeff": {"values": [0.01, 0.03, 0.1, 0.3]},
                "saes.aux_coeff": {"values": [0.01, 0.03125, 0.1]},
            }
        },
        "gated_hard_concrete": {
            "parameters": {
                "lr": {"values": [1e-4, 3e-4, 1e-3]},
                "saes.dict_size_to_input_ratio": {"values": [8.0, 16.0, 32.0, 64.0]},
                "saes.sparsity_coeff": {"values": [0.01, 0.03, 0.1, 0.3]},
                "saes.aux_coeff": {"values": [0.01, 0.03125, 0.1]},
                "saes.initial_beta": {"values": [0.1, 0.5, 1.0]},
            }
        }
    }
    
    if sae_type not in param_configs:
        raise ValueError(f"Unknown SAE type: {sae_type}")
    
    sweep_config = {
        "method": "grid",  # Use grid search for comprehensive coverage
        "name": f"{sae_type}_sae_sweep",
        "description": f"Hyperparameter sweep for {sae_type} SAE",
        "metric": {
            "name": "mse_loss",
            "goal": "minimize"
        },
        "parameters": param_configs[sae_type]["parameters"]
    }
    
    return sweep_config


def create_sweep_runner_script(base_config_path: Path, sae_type: str):
    """Create a script that can be used as a Wandb sweep agent."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Wandb sweep agent script for {sae_type} SAE.
This script is called by the wandb sweep agent.
"""

import wandb
import yaml
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import run
from config import Config


def main():
    # Initialize wandb run
    wandb.init()
    
    # Load base config
    with open("{base_config_path}", 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Update config with sweep parameters
    config_dict = base_config.copy()
    
    # Set SAE type
    config_dict['saes']['sae_type'] = "{sae_type}"
    config_dict['saes']['name'] = f"{sae_type}_sae"
    
    # Update parameters from wandb config
    wandb_config = dict(wandb.config)
    
    for key, value in wandb_config.items():
        if key.startswith('saes.'):
            sae_param = key.split('.', 1)[1]
            config_dict['saes'][sae_param] = value
        else:
            config_dict[key] = value
    
    # Set run name based on parameters
    param_string = "_".join([f"{{k}}_{{v}}" for k, v in wandb_config.items()])
    config_dict['wandb_run_name'] = f"{sae_type}_{{param_string}}"
    config_dict['wandb_tags'] = ["{sae_type}"] + [f"{{k}}_{{v}}" for k, v in wandb_config.items()]
    
    # Create Config object and run
    config = Config.model_validate(config_dict)
    run(config)


if __name__ == "__main__":
    main()
'''
    
    script_path = Path(f"scripts/sweep_agent_{sae_type}.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    script_path.chmod(0o755)
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Setup Wandb sweeps for SAE experiments")
    parser.add_argument("--sae_type", type=str, required=True,
                        choices=["relu", "hard_concrete", "gated", "gated_hard_concrete"],
                        help="SAE type to create sweep for")
    parser.add_argument("--project", type=str, default="tinystories-sweeps",
                        help="Wandb project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="Wandb entity (team/username)")
    parser.add_argument("--base_config", type=str, default="example_configs/tinystories-relu.yaml",
                        help="Base configuration file")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of sweep runs (default: all combinations)")
    parser.add_argument("--create_agent_script", action="store_true",
                        help="Create a sweep agent script")
    
    args = parser.parse_args()
    
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    # Create sweep configuration
    sweep_config = create_sweep_config(args.sae_type, base_config_path)
    
    # Save sweep config to file
    sweep_config_path = Path(f"sweep_configs/wandb_sweep_{args.sae_type}.yaml")
    sweep_config_path.parent.mkdir(exist_ok=True)
    
    with open(sweep_config_path, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    print(f"Sweep config saved to: {sweep_config_path}")
    
    # Create agent script if requested
    if args.create_agent_script:
        agent_script = create_sweep_runner_script(base_config_path, args.sae_type)
        print(f"Agent script created: {agent_script}")
    
    # Initialize wandb and create sweep
    try:
        wandb.login()
        
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project,
            entity=args.entity
        )
        
        print(f"\nCreated Wandb sweep: {sweep_id}")
        print(f"Project: {args.project}")
        print(f"URL: https://wandb.ai/{args.entity or 'your-entity'}/{args.project}/sweeps/{sweep_id}")
        
        print(f"\nTo run the sweep:")
        if args.create_agent_script:
            print(f"  wandb agent {sweep_id} --program scripts/sweep_agent_{args.sae_type}.py")
        else:
            print(f"  # Create an agent script first with --create_agent_script")
            print(f"  wandb agent {sweep_id}")
        
        # Calculate total runs
        total_combinations = 1
        for param_config in sweep_config["parameters"].values():
            if "values" in param_config:
                total_combinations *= len(param_config["values"])
        
        print(f"\nTotal parameter combinations: {total_combinations}")
        if args.count:
            print(f"Will run {min(args.count, total_combinations)} experiments")
        
    except Exception as e:
        print(f"Error creating Wandb sweep: {e}")
        print("You can still use the saved sweep config file to create the sweep manually.")


if __name__ == "__main__":
    main() 