#!/usr/bin/env python3
"""
Generate sweep configurations for different SAE types and hyperparameters.

Usage:
    python scripts/generate_sweep_configs.py --output_dir sweep_configs
"""

import argparse
import yaml
from pathlib import Path
from itertools import product
from typing import Dict, Any, List
import copy


def load_base_config(config_path: Path) -> Dict[str, Any]:
    """Load base configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_config_variations(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    sae_type: str
) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters for a given SAE type."""
    
    configs = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        config = copy.deepcopy(base_config)
        
        # Update SAE type
        config['saes']['sae_type'] = sae_type
        config['saes']['name'] = f"{sae_type}_sae"
        
        # Update hyperparameters
        param_string_parts = []
        for param_name, value in zip(param_names, values):
            if param_name.startswith('saes.'):
                # SAE-specific parameter
                sae_param = param_name.split('.', 1)[1]
                config['saes'][sae_param] = value
                param_string_parts.append(f"{sae_param}_{value}")
            else:
                # Top-level parameter
                config[param_name] = value
                param_string_parts.append(f"{param_name}_{value}")
        
        # Update run name to include hyperparameters
        param_string = "_".join(param_string_parts)
        config['wandb_run_name'] = f"{sae_type}_{param_string}"
        config['wandb_tags'] = [sae_type] + [f"{k}_{v}" for k, v in zip(param_names, values)]
        
        configs.append(config)
    
    return configs


def save_configs(configs: List[Dict[str, Any]], output_dir: Path, sae_type: str):
    """Save generated configs to YAML files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, config in enumerate(configs):
        filename = f"{sae_type}_sweep_{i:03d}.yaml"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate SAE sweep configurations")
    parser.add_argument("--base_config", type=str, default="example_configs/tinystories-relu.yaml",
                        help="Base configuration file")
    parser.add_argument("--output_dir", type=str, default="sweep_configs",
                        help="Output directory for generated configs")
    
    args = parser.parse_args()
    
    base_config_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    base_config = load_base_config(base_config_path)
    
    # Define hyperparameter grids for each SAE type
    sae_configs = {
        "relu": {
            "saes.dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "saes.sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "lr": [1e-4, 3e-4, 1e-3]
        },
        "hard_concrete": {
            "saes.dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "saes.sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "saes.input_dependent_gates": [True, False],
            "saes.initial_beta": [0.1, 0.5, 1.0],
            "lr": [1e-4, 3e-4, 1e-3]
        },
        "gated": {
            "saes.dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "saes.sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "saes.aux_coeff": [0.01, 0.03125, 0.1],
            "lr": [1e-4, 3e-4, 1e-3]
        },
        "gated_hard_concrete": {
            "saes.dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "saes.sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "saes.aux_coeff": [0.01, 0.03125, 0.1],
            "saes.initial_beta": [0.1, 0.5, 1.0],
            "lr": [1e-4, 3e-4, 1e-3]
        }
    }
    
    total_configs = 0
    
    for sae_type, param_grid in sae_configs.items():
        print(f"\nGenerating configs for {sae_type} SAE...")
        configs = generate_config_variations(base_config, param_grid, sae_type)
        save_configs(configs, output_dir, sae_type)
        
        print(f"Generated {len(configs)} configs for {sae_type}")
        total_configs += len(configs)
    
    print(f"\nTotal configs generated: {total_configs}")
    print(f"Configs saved to: {output_dir}")


if __name__ == "__main__":
    main() 