#!/usr/bin/env python3
"""
Run dynamic experiment sweeps by creating configs in memory and calling the run function directly.
This approach is more efficient than subprocess calls and doesn't require pre-generating config files.

Usage:
    python scripts/run_dynamic_sweep.py --sae_types relu,hard_concrete --max_parallel 2
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
import logging
from itertools import product
import copy
import traceback

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import run
from config import Config
from utils.io import load_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_experiment_configs(base_config_path: Path, sae_types: List[str]) -> List[Dict[str, Any]]:
    """Create all experiment configurations dynamically."""
    
    # Load base config as dict
    with open(base_config_path, 'r') as f:
        import yaml
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter grids for each SAE type
    param_grids = {
        "relu": {
            "dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
        },
        "hard_concrete": {
            "dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "input_dependent_gates": [True, False],
            "initial_beta": [0.1, 0.5, 1.0],
        },
        "gated": {
            "dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "aux_coeff": [0.01, 0.03125, 0.1],
        },
        "gated_hard_concrete": {
            "dict_size_to_input_ratio": [8.0, 16.0, 32.0, 64.0],
            "sparsity_coeff": [0.01, 0.03, 0.1, 0.3],
            "aux_coeff": [0.01, 0.03125, 0.1],
            "initial_beta": [0.1, 0.5, 1.0],
        }
    }
    
    # Learning rate grid (common to all)
    lr_values = [1e-4, 3e-4, 1e-3]
    
    all_configs = []
    
    for sae_type in sae_types:
        if sae_type not in param_grids:
            logger.warning(f"Unknown SAE type: {sae_type}, skipping...")
            continue
            
        param_grid = param_grids[sae_type]
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Add learning rate to the grid
        param_names.append('lr')
        param_values.append(lr_values)
        
        for values in product(*param_values):
            config = copy.deepcopy(base_config)
            
            # Update SAE type and name
            config['saes']['sae_type'] = sae_type
            config['saes']['name'] = f"{sae_type}_sae"
            
            # Update hyperparameters
            param_string_parts = []
            for param_name, value in zip(param_names, values):
                if param_name == 'lr':
                    config['lr'] = value
                    param_string_parts.append(f"lr_{value}")
                else:
                    config['saes'][param_name] = value
                    param_string_parts.append(f"{param_name}_{value}")
            
            # Update run name and tags
            param_string = "_".join(param_string_parts)
            config['wandb_run_name'] = f"{sae_type}_{param_string}"
            config['wandb_tags'] = [sae_type] + [f"{k}_{v}" for k, v in zip(param_names, values)]
            
            # Add experiment metadata
            config['_experiment_id'] = f"{sae_type}_{len(all_configs):03d}"
            config['_sae_type'] = sae_type
            
            all_configs.append(config)
    
    return all_configs


def run_single_experiment_direct(config_dict: Dict[str, Any]) -> tuple[str, bool, str]:
    """Run a single experiment directly using the run function."""
    experiment_id = config_dict.get('_experiment_id', 'unknown')
    start_time = time.time()
    
    try:
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Create Config object from dict
        config = Config.model_validate(config_dict)
        
        # Run the experiment
        run(config)
        
        duration = time.time() - start_time
        logger.info(f"✅ Completed {experiment_id} in {duration:.1f}s")
        return experiment_id, True, f"Success in {duration:.1f}s"
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error after {duration:.1f}s: {str(e)}"
        logger.error(f"❌ Failed {experiment_id}: {error_msg}")
        logger.debug(f"Traceback for {experiment_id}:\n{traceback.format_exc()}")
        return experiment_id, False, error_msg


def run_experiments_sequential(configs: List[Dict[str, Any]]) -> List[tuple[str, bool, str]]:
    """Run experiments sequentially."""
    results = []
    
    for i, config in enumerate(configs):
        experiment_id = config.get('_experiment_id', f'exp_{i}')
        logger.info(f"Running experiment {i+1}/{len(configs)}: {experiment_id}")
        result = run_single_experiment_direct(config)
        results.append(result)
    
    return results


def run_experiments_parallel(configs: List[Dict[str, Any]], max_workers: int) -> List[tuple[str, bool, str]]:
    """Run experiments in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_experiment_direct, config): config 
            for config in configs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            experiment_id = config.get('_experiment_id', 'unknown')
            
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed: {result[0]} ({len(results)}/{len(configs)})")
            except Exception as e:
                error_result = (experiment_id, False, f"Process error: {e}")
                results.append(error_result)
                logger.error(f"Process failed: {experiment_id}: {e}")
    
    return results


def print_summary(results: List[tuple[str, bool, str]], by_sae_type: bool = True):
    """Print a summary of all experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if by_sae_type and results:
        # Group by SAE type
        sae_types = set(r[0].split('_')[0] for r in results)
        for sae_type in sorted(sae_types):
            sae_results = [r for r in results if r[0].startswith(sae_type)]
            sae_successful = [r for r in sae_results if r[1]]
            print(f"\n{sae_type.upper()} SAE: {len(sae_successful)}/{len(sae_results)} successful")
    
    if failed:
        print("\nFAILED EXPERIMENTS:")
        for experiment_id, _, message in failed:
            print(f"  ❌ {experiment_id}: {message}")


def main():
    parser = argparse.ArgumentParser(description="Run dynamic SAE experiment sweep")
    parser.add_argument("--base_config", type=str, default="example_configs/tinystories-relu.yaml",
                        help="Base configuration file")
    parser.add_argument("--sae_types", type=str, default="relu,hard_concrete,gated,gated_hard_concrete",
                        help="Comma-separated list of SAE types to test")
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Maximum number of parallel experiments (1 = sequential)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate configs and show what would run without actually running")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of experiments to run (for testing)")
    
    args = parser.parse_args()
    
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    sae_types = [s.strip() for s in args.sae_types.split(',')]
    
    # Generate configs
    logger.info("Generating experiment configurations...")
    configs = create_experiment_configs(base_config_path, sae_types)
    
    if args.limit:
        configs = configs[:args.limit]
    
    print(f"Generated {len(configs)} experiment configurations")
    
    # Group by SAE type for summary
    for sae_type in sae_types:
        count = len([c for c in configs if c.get('_sae_type') == sae_type])
        print(f"  {sae_type}: {count} experiments")
    
    if args.dry_run:
        print("\nDRY RUN - Would run the following experiments:")
        for config in configs:
            exp_id = config.get('_experiment_id')
            run_name = config.get('wandb_run_name', 'unnamed')
            print(f"  {exp_id}: {run_name}")
        return
    
    # Run experiments
    start_time = time.time()
    
    if args.max_parallel == 1:
        logger.info("Running experiments sequentially...")
        results = run_experiments_sequential(configs)
    else:
        logger.info(f"Running experiments with max {args.max_parallel} parallel workers...")
        results = run_experiments_parallel(configs, args.max_parallel)
    
    total_duration = time.time() - start_time
    
    # Print summary
    print_summary(results)
    print(f"\nTotal sweep duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")


if __name__ == "__main__":
    main() 