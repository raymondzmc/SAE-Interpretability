#!/usr/bin/env python3
"""
Multi-device experiment scheduler for SAE hyperparameter sweeps.
Automatically detects available devices (GPUs + CPU) and distributes experiments across them.

Usage:
    python scripts/run_multi_device_sweep.py --sae_types relu,hard_concrete --devices auto
    python scripts/run_multi_device_sweep.py --devices cuda:0,cuda:1,cpu --limit 10
"""

import argparse
import subprocess
import time
import torch
import queue
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from itertools import product
import copy
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeviceInfo:
    """Information about a compute device."""
    def __init__(self, device_id: str, device_type: str, memory_gb: float = 0.0):
        self.device_id = device_id  # e.g., "cuda:0", "cpu"
        self.device_type = device_type  # "cuda", "cpu", "mps"
        self.memory_gb = memory_gb
        self.is_available = True
        
    def __str__(self):
        if self.device_type == "cuda":
            return f"{self.device_id} ({self.memory_gb:.1f}GB)"
        else:
            return self.device_id


def detect_available_devices() -> List[DeviceInfo]:
    """Detect all available compute devices."""
    devices = []
    
    # Detect CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_id = f"cuda:{i}"
            try:
                # Get memory info
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                devices.append(DeviceInfo(device_id, "cuda", memory_gb))
                logger.info(f"Found CUDA device: {device_id} ({memory_gb:.1f}GB) - {props.name}")
            except Exception as e:
                logger.warning(f"Error querying CUDA device {i}: {e}")
    
    # Detect MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        devices.append(DeviceInfo("mps", "mps"))
        logger.info("Found MPS device (Apple Silicon)")
    
    # Always add CPU as fallback
    devices.append(DeviceInfo("cpu", "cpu"))
    logger.info("Found CPU device")
    
    return devices


def parse_device_list(device_str: str) -> List[str]:
    """Parse device specification string into list of device IDs."""
    if device_str.lower() == "auto":
        devices = detect_available_devices()
        return [d.device_id for d in devices]
    else:
        return [d.strip() for d in device_str.split(',')]


class ExperimentQueue:
    """Thread-safe queue for managing experiments per device."""
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.queue = queue.Queue()
        self.active_experiment = None
        self.completed_count = 0
        self.failed_count = 0
        
    def add_experiment(self, config_dict: Dict[str, Any]):
        """Add an experiment to this device's queue."""
        self.queue.put(config_dict)
        
    def get_next_experiment(self) -> Optional[Dict[str, Any]]:
        """Get the next experiment from the queue."""
        try:
            experiment = self.queue.get_nowait()
            self.active_experiment = experiment
            return experiment
        except queue.Empty:
            return None
            
    def mark_completed(self, success: bool):
        """Mark the current experiment as completed."""
        if success:
            self.completed_count += 1
        else:
            self.failed_count += 1
        self.active_experiment = None
        
    def is_busy(self) -> bool:
        """Check if this device is currently running an experiment."""
        return self.active_experiment is not None
        
    def get_queue_size(self) -> int:
        """Get the number of experiments waiting in queue."""
        return self.queue.qsize()


class MultiDeviceExperimentRunner:
    """Manages experiment execution across multiple devices."""
    
    def __init__(self, device_ids: List[str]):
        self.device_queues = {device_id: ExperimentQueue(device_id) for device_id in device_ids}
        self.running_threads = {}
        self.results = []
        self.lock = threading.Lock()
        
    def add_experiments(self, experiments: List[Dict[str, Any]]):
        """Distribute experiments across available devices."""
        device_ids = list(self.device_queues.keys())
        
        # Distribute experiments evenly across devices
        for i, experiment in enumerate(experiments):
            device_id = device_ids[i % len(device_ids)]
            self.device_queues[device_id].add_experiment(experiment)
            
        # Log distribution
        logger.info("Experiment distribution:")
        for device_id, device_queue in self.device_queues.items():
            queue_size = device_queue.get_queue_size()
            logger.info(f"  {device_id}: {queue_size} experiments")
    
    def run_single_experiment(self, config_dict: Dict[str, Any], device_id: str) -> tuple[str, bool, str]:
        """Run a single experiment on the specified device."""
        experiment_id = config_dict.get('_experiment_id', 'unknown')
        start_time = time.time()
        
        try:
            logger.info(f"[{device_id}] Starting experiment: {experiment_id}")
            
            # Create temporary config file
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                # Remove internal metadata before saving
                clean_config = {k: v for k, v in config_dict.items() if not k.startswith('_')}
                yaml.dump(clean_config, f)
                temp_config_path = f.name
            
            try:
                # Run experiment with device specification
                cmd = ["python", "run.py", temp_config_path, "--device", device_id]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    logger.info(f"[{device_id}] ✅ Completed {experiment_id} in {duration:.1f}s")
                    return experiment_id, True, f"Success in {duration:.1f}s"
                else:
                    error_msg = result.stderr.strip()[:200]  # Truncate long errors
                    logger.error(f"[{device_id}] ❌ Failed {experiment_id}: {error_msg}")
                    return experiment_id, False, f"Failed: {error_msg}"
                    
            finally:
                # Clean up temporary file
                Path(temp_config_path).unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            logger.error(f"[{device_id}] ⏰ Timeout {experiment_id}")
            return experiment_id, False, "Timeout after 2 hours"
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error after {duration:.1f}s: {str(e)}"
            logger.error(f"[{device_id}] 💥 Error {experiment_id}: {error_msg}")
            return experiment_id, False, error_msg
    
    def device_worker(self, device_id: str):
        """Worker function that processes experiments for a specific device."""
        device_queue = self.device_queues[device_id]
        
        while True:
            experiment = device_queue.get_next_experiment()
            if experiment is None:
                # No more experiments for this device
                break
                
            # Run the experiment
            result = self.run_single_experiment(experiment, device_id)
            
            # Update results and status
            with self.lock:
                self.results.append(result)
                device_queue.mark_completed(result[1])  # result[1] is success flag
                
                # Log progress
                total_completed = sum(dq.completed_count + dq.failed_count for dq in self.device_queues.values())
                total_experiments = total_completed + sum(dq.get_queue_size() for dq in self.device_queues.values())
                logger.info(f"Progress: {total_completed}/{total_experiments} completed")
    
    def run_all_experiments(self) -> List[tuple[str, bool, str]]:
        """Run all experiments using available devices."""
        logger.info("Starting multi-device experiment execution...")
        
        # Start worker threads for each device
        threads = []
        for device_id in self.device_queues.keys():
            thread = threading.Thread(target=self.device_worker, args=(device_id,))
            thread.start()
            threads.append(thread)
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        logger.info("All experiments completed!")
        return self.results
    
    def print_summary(self):
        """Print summary of experiment results."""
        print("\n" + "="*80)
        print("MULTI-DEVICE EXPERIMENT SUMMARY")
        print("="*80)
        
        successful = [r for r in self.results if r[1]]
        failed = [r for r in self.results if not r[1]]
        
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        # Device-specific summary
        print("\nDevice Performance:")
        for device_id, device_queue in self.device_queues.items():
            print(f"  {device_id}: {device_queue.completed_count} success, {device_queue.failed_count} failed")
        
        if failed:
            print("\nFailed Experiments:")
            for experiment_id, _, message in failed[:10]:  # Show first 10 failures
                print(f"  ❌ {experiment_id}: {message}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")


def create_experiment_configs(base_config_path: Path, sae_types: List[str]) -> List[Dict[str, Any]]:
    """Create all experiment configurations dynamically."""
    
    # Load base config as dict
    with open(base_config_path, 'r') as f:
        import yaml
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter grids for each SAE type (simplified for multi-device testing)
    param_grids = {
        "relu": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
        },
        "hard_concrete": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
            "input_dependent_gates": [True, False],
            "initial_beta": [0.5, 1.0],
        },
        "gated": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
            "aux_coeff": [0.03125, 0.1],
        },
        "gated_hard_concrete": {
            "dict_size_to_input_ratio": [16.0, 32.0],
            "sparsity_coeff": [0.03, 0.1],
            "aux_coeff": [0.03125, 0.1],
            "initial_beta": [0.5, 1.0],
        }
    }
    
    # Learning rate grid (common to all)
    lr_values = [3e-4, 1e-3]
    
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


def main():
    parser = argparse.ArgumentParser(description="Multi-device SAE experiment sweep")
    parser.add_argument("--base_config", type=str, default="example_configs/tinystories-relu.yaml",
                        help="Base configuration file")
    parser.add_argument("--sae_types", type=str, default="relu,hard_concrete",
                        help="Comma-separated list of SAE types to test")
    parser.add_argument("--devices", type=str, default="auto",
                        help="Devices to use: 'auto' or comma-separated list (e.g., 'cuda:0,cuda:1,cpu')")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show device detection and experiment distribution without running")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of experiments to run (for testing)")
    parser.add_argument("--show_devices", action="store_true",
                        help="Just show available devices and exit")
    
    args = parser.parse_args()
    
    # Show devices if requested
    if args.show_devices:
        devices = detect_available_devices()
        print("Available devices:")
        for device in devices:
            print(f"  {device}")
        return
    
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    # Parse device specification
    device_ids = parse_device_list(args.devices)
    logger.info(f"Using devices: {device_ids}")
    
    # Parse SAE types
    sae_types = [s.strip() for s in args.sae_types.split(',')]
    
    # Generate experiment configs
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
        print(f"\nWould distribute across {len(device_ids)} devices:")
        for device_id in device_ids:
            device_configs = [c for i, c in enumerate(configs) if i % len(device_ids) == device_ids.index(device_id)]
            print(f"  {device_id}: {len(device_configs)} experiments")
        return
    
    # Run experiments
    start_time = time.time()
    
    runner = MultiDeviceExperimentRunner(device_ids)
    runner.add_experiments(configs)
    
    results = runner.run_all_experiments()
    
    total_duration = time.time() - start_time
    
    # Print summary
    runner.print_summary()
    print(f"\nTotal sweep duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Average time per experiment: {total_duration/len(results):.1f}s")


if __name__ == "__main__":
    main() 