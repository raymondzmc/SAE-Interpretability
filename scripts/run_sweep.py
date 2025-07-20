#!/usr/bin/env python3
"""
Run a sweep of experiments from generated config files.

Usage:
    python scripts/run_sweep.py --config_dir sweep_configs --max_parallel 2
"""

import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_experiment(config_path: Path) -> tuple[str, int, str]:
    """Run a single experiment with the given config."""
    start_time = time.time()
    config_name = config_path.name
    
    try:
        logger.info(f"Starting experiment: {config_name}")
        result = subprocess.run(
            ["python", "run.py", str(config_path)],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Completed {config_name} in {duration:.1f}s")
            return config_name, 0, f"Success in {duration:.1f}s"
        else:
            logger.error(f"❌ Failed {config_name}: {result.stderr}")
            return config_name, result.returncode, f"Failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout {config_name}")
        return config_name, -1, "Timeout after 2 hours"
    except Exception as e:
        logger.error(f"💥 Error {config_name}: {e}")
        return config_name, -2, f"Error: {e}"


def get_config_files(config_dir: Path, pattern: str = "*.yaml") -> List[Path]:
    """Get all config files matching the pattern."""
    config_files = list(config_dir.glob(pattern))
    config_files.sort()  # Ensure consistent ordering
    return config_files


def run_experiments_sequential(config_files: List[Path]) -> List[tuple[str, int, str]]:
    """Run experiments sequentially."""
    results = []
    
    for i, config_path in enumerate(config_files):
        logger.info(f"Running experiment {i+1}/{len(config_files)}: {config_path.name}")
        result = run_single_experiment(config_path)
        results.append(result)
    
    return results


def run_experiments_parallel(config_files: List[Path], max_workers: int) -> List[tuple[str, int, str]]:
    """Run experiments in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_experiment, config_path): config_path 
            for config_path in config_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_config):
            config_path = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed: {result[0]} ({len(results)}/{len(config_files)})")
            except Exception as e:
                error_result = (config_path.name, -3, f"Process error: {e}")
                results.append(error_result)
                logger.error(f"Process failed: {config_path.name}: {e}")
    
    return results


def print_summary(results: List[tuple[str, int, str]]):
    """Print a summary of all experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r[1] == 0]
    failed = [r for r in results if r[1] != 0]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFAILED EXPERIMENTS:")
        for config_name, return_code, message in failed:
            print(f"  ❌ {config_name}: {message}")
    
    if successful:
        print(f"\nSUCCESSFUL EXPERIMENTS: {len(successful)}")
        for config_name, _, message in successful:
            print(f"  ✅ {config_name}: {message}")


def main():
    parser = argparse.ArgumentParser(description="Run SAE experiment sweep")
    parser.add_argument("--config_dir", type=str, default="sweep_configs",
                        help="Directory containing config files")
    parser.add_argument("--pattern", type=str, default="*.yaml",
                        help="Pattern to match config files")
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Maximum number of parallel experiments (1 = sequential)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter configs by SAE type (relu, hard_concrete, gated, gated_hard_concrete)")
    parser.add_argument("--dry_run", action="store_true",
                        help="List configs that would be run without actually running them")
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Get config files
    if args.filter:
        pattern = f"{args.filter}_*.yaml"
    else:
        pattern = args.pattern
    
    config_files = get_config_files(config_dir, pattern)
    
    if not config_files:
        logger.warning(f"No config files found in {config_dir} matching {pattern}")
        return
    
    print(f"Found {len(config_files)} config files")
    
    if args.dry_run:
        print("DRY RUN - Would run the following configs:")
        for config_path in config_files:
            print(f"  {config_path}")
        return
    
    # Run experiments
    start_time = time.time()
    
    if args.max_parallel == 1:
        logger.info("Running experiments sequentially...")
        results = run_experiments_sequential(config_files)
    else:
        logger.info(f"Running experiments with max {args.max_parallel} parallel workers...")
        results = run_experiments_parallel(config_files, args.max_parallel)
    
    total_duration = time.time() - start_time
    
    # Print summary
    print_summary(results)
    print(f"\nTotal sweep duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")


if __name__ == "__main__":
    main() 