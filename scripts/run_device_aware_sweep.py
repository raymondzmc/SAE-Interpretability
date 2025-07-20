#!/usr/bin/env python3
"""
Device-aware experiment sweep runner.
Extends the existing sweep approaches with automatic device detection and distribution.

Usage:
    python scripts/run_device_aware_sweep.py --config_dir sweep_configs --devices auto
    python scripts/run_device_aware_sweep.py --sae_types relu --devices cuda:0,cuda:1
"""

import argparse
import subprocess
import time
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import logging
import copy
import sys
from itertools import product

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing scripts
from scripts.generate_sweep_configs import create_experiment_configs as generate_configs
from scripts.run_sweep import get_config_files, print_summary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_available_devices() -> List[str]:
    """Detect all available compute devices."""
    devices = []
    
    # Detect CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_id = f"cuda:{i}"
            try:
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                devices.append(device_id)
                logger.info(f"Found CUDA device: {device_id} ({memory_gb:.1f}GB) - {props.name}")
            except Exception as e:
                logger.warning(f"Error querying CUDA device {i}: {e}")
    
    # Detect MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        devices.append("mps")
        logger.info("Found MPS device (Apple Silicon)")
    
    # Always add CPU as fallback
    devices.append("cpu")
    logger.info("Found CPU device")
    
    return devices


def parse_device_list(device_str: str) -> List[str]:
    """Parse device specification string into list of device IDs."""
    if device_str.lower() == "auto":
        return detect_available_devices()
    else:
        return [d.strip() for d in device_str.split(',')]


def run_single_experiment_on_device(config_path: Path, device_id: str) -> tuple[str, int, str]:
    """Run a single experiment on the specified device."""
    start_time = time.time()
    config_name = config_path.name
    
    try:
        logger.info(f"[{device_id}] Starting experiment: {config_name}")
        
        # Run experiment with device specification
        cmd = ["python", "run.py", str(config_path), "--device", device_id]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"[{device_id}] ✅ Completed {config_name} in {duration:.1f}s")
            return config_name, 0, f"Success on {device_id} in {duration:.1f}s"
        else:
            error_msg = result.stderr.strip()[:200]  # Truncate long errors
            logger.error(f"[{device_id}] ❌ Failed {config_name}: {error_msg}")
            return config_name, result.returncode, f"Failed on {device_id}: {error_msg}"
            
    except subprocess.TimeoutExpired:
        logger.error(f"[{device_id}] ⏰ Timeout {config_name}")
        return config_name, -1, f"Timeout on {device_id} after 2 hours"
    except Exception as e:
        logger.error(f"[{device_id}] 💥 Error {config_name}: {e}")
        return config_name, -2, f"Error on {device_id}: {e}"


def run_experiments_on_devices(config_files: List[Path], device_ids: List[str]) -> List[tuple[str, int, str]]:
    """Run experiments distributed across multiple devices."""
    results = []
    
    # Create device-config pairs by cycling through devices
    device_config_pairs = []
    for i, config_file in enumerate(config_files):
        device_id = device_ids[i % len(device_ids)]
        device_config_pairs.append((config_file, device_id))
    
    # Log distribution
    device_counts = {device_id: 0 for device_id in device_ids}
    for _, device_id in device_config_pairs:
        device_counts[device_id] += 1
    
    logger.info("Experiment distribution across devices:")
    for device_id, count in device_counts.items():
        logger.info(f"  {device_id}: {count} experiments")
    
    with ProcessPoolExecutor(max_workers=len(device_ids)) as executor:
        # Submit all jobs with their assigned devices
        future_to_info = {
            executor.submit(run_single_experiment_on_device, config_file, device_id): (config_file, device_id)
            for config_file, device_id in device_config_pairs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_info):
            config_file, device_id = future_to_info[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed: {result[0]} ({len(results)}/{len(config_files)})")
            except Exception as e:
                error_result = (config_file.name, -3, f"Process error on {device_id}: {e}")
                results.append(error_result)
                logger.error(f"Process failed: {config_file.name} on {device_id}: {e}")
    
    return results


def create_experiment_configs_simplified(base_config_path: Path, sae_types: List[str]) -> List[Dict[str, Any]]:
    """Create experiment configurations with a simplified grid for device testing."""
    
    # Load base config as dict
    with open(base_config_path, 'r') as f:
        import yaml
        base_config = yaml.safe_load(f)
    
    # Simplified parameter grids for device testing
    param_grids = {
        "relu": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
        },
        "hard_concrete": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
            "input_dependent_gates": [True, False],
            "initial_beta": [0.5],
        },
        "gated": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
            "aux_coeff": [0.03125],
        },
        "gated_hard_concrete": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
            "aux_coeff": [0.03125],
            "initial_beta": [0.5],
        }
    }
    
    lr_values = [3e-4]  # Single LR for testing
    
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
            
            all_configs.append(config)
    
    return all_configs


def save_temp_configs(configs: List[Dict[str, Any]], temp_dir: Path) -> List[Path]:
    """Save experiment configs to temporary files and return paths."""
    import yaml
    
    temp_dir.mkdir(exist_ok=True)
    config_paths = []
    
    for i, config in enumerate(configs):
        config_path = temp_dir / f"experiment_{i:03d}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        config_paths.append(config_path)
    
    return config_paths


def main():
    parser = argparse.ArgumentParser(description="Device-aware SAE experiment sweep")
    parser.add_argument("--config_dir", type=str, default=None,
                        help="Directory containing pre-generated config files")
    parser.add_argument("--base_config", type=str, default="example_configs/tinystories-relu.yaml",
                        help="Base configuration file (for dynamic generation)")
    parser.add_argument("--sae_types", type=str, default="relu,hard_concrete",
                        help="Comma-separated list of SAE types (for dynamic generation)")
    parser.add_argument("--devices", type=str, default="auto",
                        help="Devices to use: 'auto' or comma-separated list (e.g., 'cuda:0,cuda:1,cpu')")
    parser.add_argument("--pattern", type=str, default="*.yaml",
                        help="Pattern to match config files (when using --config_dir)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter configs by SAE type")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show device detection and experiment distribution without running")
    parser.add_argument("--show_devices", action="store_true",
                        help="Just show available devices and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of experiments to run")
    
    args = parser.parse_args()
    
    # Show devices if requested
    if args.show_devices:
        devices = detect_available_devices()
        print("Available devices:")
        for device in devices:
            print(f"  {device}")
        return
    
    # Parse device specification
    device_ids = parse_device_list(args.devices)
    logger.info(f"Using devices: {device_ids}")
    
    if not device_ids:
        raise ValueError("No devices available or specified")
    
    # Get config files either from directory or generate dynamically
    config_files = []
    temp_dir = None
    
    if args.config_dir:
        # Use pre-generated config files
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        # Get config files with optional filtering
        pattern = f"{args.filter}_*.yaml" if args.filter else args.pattern
        config_files = get_config_files(config_dir, pattern)
        
    else:
        # Generate configs dynamically
        base_config_path = Path(args.base_config)
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base config file not found: {base_config_path}")
        
        sae_types = [s.strip() for s in args.sae_types.split(',')]
        logger.info("Generating experiment configurations dynamically...")
        
        configs = create_experiment_configs_simplified(base_config_path, sae_types)
        
        # Save to temporary files
        temp_dir = Path("temp_device_sweep_configs")
        config_files = save_temp_configs(configs, temp_dir)
    
    if not config_files:
        logger.warning("No config files found or generated")
        return
    
    if args.limit:
        config_files = config_files[:args.limit]
    
    print(f"Found/generated {len(config_files)} config files")
    
    if args.dry_run:
        print(f"\nWould distribute across {len(device_ids)} devices:")
        for i, device_id in enumerate(device_ids):
            device_configs = [cf for j, cf in enumerate(config_files) if j % len(device_ids) == i]
            print(f"  {device_id}: {len(device_configs)} experiments")
        
        if temp_dir:
            print(f"\nCleaning up temporary configs...")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    # Run experiments
    start_time = time.time()
    
    try:
        results = run_experiments_on_devices(config_files, device_ids)
        
        total_duration = time.time() - start_time
        
        # Print summary
        print_summary(results)
        print(f"\nDevice-specific performance:")
        device_results = {}
        for config_name, return_code, message in results:
            # Extract device from message
            if " on " in message:
                device = message.split(" on ")[1].split(" ")[0]
                if device not in device_results:
                    device_results[device] = {"success": 0, "failed": 0}
                if return_code == 0:
                    device_results[device]["success"] += 1
                else:
                    device_results[device]["failed"] += 1
        
        for device, stats in device_results.items():
            total = stats["success"] + stats["failed"]
            success_rate = (stats["success"] / total * 100) if total > 0 else 0
            print(f"  {device}: {stats['success']}/{total} success ({success_rate:.1f}%)")
        
        print(f"\nTotal sweep duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"Average time per experiment: {total_duration/len(results):.1f}s")
        
        if len(device_ids) > 1:
            speedup = len(device_ids) * (total_duration / len(results)) / total_duration
            print(f"Estimated speedup from multi-device: {speedup:.1f}x")
            
    finally:
        # Clean up temporary files if created
        if temp_dir and temp_dir.exists():
            logger.info("Cleaning up temporary config files...")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main() 