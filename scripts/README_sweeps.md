# SAE Hyperparameter Sweep Methods

This directory contains multiple approaches for running hyperparameter sweeps with your SAE experiments. Each method has different trade-offs in terms of complexity, features, and use cases.

## Overview of Approaches

| Approach | Complexity | Features | Best For |
|----------|------------|----------|----------|
| **Config Generator + Runner** | Low | File-based, resume support | Systematic grid searches |
| **Simple Bash Script** | Very Low | Shell-based, minimal deps | Quick local experiments |
| **Dynamic Python Runner** | Medium | In-memory configs, efficient | Development & testing |
| **Wandb Sweeps** | Medium | Professional tracking, web UI | Production experiments |
| **Multi-Device Scheduler** | Medium-High | GPU auto-detection, threading | Multi-GPU systems |
| **Device-Aware Runner** | Medium | Simple device distribution | Multi-device optimization |

## Approach 1: Config Generator + Simple Runner

**Files:** `generate_sweep_configs.py`, `run_sweep.py`

This approach pre-generates all config files and then runs them.

### Usage:
```bash
# Generate configs for all SAE types
python scripts/generate_sweep_configs.py --output_dir sweep_configs

# Run all configs sequentially
python scripts/run_sweep.py --config_dir sweep_configs

# Run only ReLU SAE configs with 2 parallel jobs
python scripts/run_sweep.py --config_dir sweep_configs --filter relu --max_parallel 2

# Dry run to see what would be executed
python scripts/run_sweep.py --config_dir sweep_configs --dry_run
```

### Advantages:
- ✅ Easy to inspect generated configs
- ✅ Can resume failed experiments
- ✅ Good for systematic grid searches
- ✅ Parallel execution support

### Disadvantages:
- ❌ Creates many files
- ❌ Subprocess overhead

## Approach 2: Simple Bash Script

**File:** `run_simple_sweep.sh`

A straightforward bash script for quick sweeps.

### Usage:
```bash
# Make executable (run this once)
chmod +x scripts/run_simple_sweep.sh

# Run ReLU SAE sweep sequentially
./scripts/run_simple_sweep.sh relu 1

# Run Hard Concrete SAE sweep with 4 parallel jobs
./scripts/run_simple_sweep.sh hard_concrete 4

# Run Gated SAE sweep
./scripts/run_simple_sweep.sh gated 2
```

### Advantages:
- ✅ Minimal dependencies (just bash)
- ✅ Easy to modify parameters
- ✅ Good for quick local experiments
- ✅ Parallel execution support

### Disadvantages:
- ❌ Limited error handling
- ❌ Less sophisticated than Python alternatives

## Approach 3: Dynamic Python Runner

**File:** `run_dynamic_sweep.py`

Creates configs in memory and calls the `run` function directly.

### Usage:
```bash
# Run sweep for specific SAE types
python scripts/run_dynamic_sweep.py --sae_types relu,hard_concrete --max_parallel 2

# Test with limited experiments
python scripts/run_dynamic_sweep.py --sae_types relu --limit 5 --dry_run

# Run all SAE types sequentially
python scripts/run_dynamic_sweep.py --max_parallel 1
```

### Advantages:
- ✅ No config files created
- ✅ More efficient (no subprocess overhead)
- ✅ Easy to modify parameter grids in code
- ✅ Good error handling and logging

### Disadvantages:
- ❌ Harder to inspect individual configs
- ❌ Can't easily resume failed experiments

## Approach 4: Wandb Sweeps

**File:** `setup_wandb_sweep.py`

Uses Wandb's professional sweep functionality.

### Usage:
```bash
# Setup and create a ReLU SAE sweep
python scripts/setup_wandb_sweep.py --sae_type relu --project tinystories-sweeps --create_agent_script

# This creates a sweep and gives you a command like:
# wandb agent <sweep_id> --program scripts/sweep_agent_relu.py

# Setup sweeps for all SAE types
for sae_type in relu hard_concrete gated gated_hard_concrete; do
    python scripts/setup_wandb_sweep.py --sae_type $sae_type --project tinystories-sweeps --create_agent_script
done
```

### Advantages:
- ✅ Professional experiment tracking
- ✅ Web-based dashboard and visualization
- ✅ Can run across multiple machines
- ✅ Advanced sweep methods (Bayesian optimization, etc.)
- ✅ Automatic hyperparameter importance analysis

### Disadvantages:
- ❌ Requires Wandb account and setup
- ❌ More complex initial setup

## Approach 5: Multi-Device Scheduler

**File:** `run_multi_device_sweep.py`

Advanced multi-device experiment scheduler with thread-based queue management.

### Usage:
```bash
# Auto-detect and use all available devices
python scripts/run_multi_device_sweep.py --sae_types relu,hard_concrete --devices auto

# Use specific devices
python scripts/run_multi_device_sweep.py --devices cuda:0,cuda:1,cpu --limit 20

# Show available devices
python scripts/run_multi_device_sweep.py --show_devices

# Dry run to see distribution
python scripts/run_multi_device_sweep.py --sae_types relu --dry_run
```

### Advantages:
- ✅ Automatic GPU detection with memory info
- ✅ Thread-based queue management per device
- ✅ Real-time experiment distribution
- ✅ Advanced device performance tracking
- ✅ Supports CUDA, MPS (Apple Silicon), CPU

### Disadvantages:
- ❌ More complex architecture
- ❌ Requires PyTorch for device detection

## Approach 6: Device-Aware Runner

**File:** `run_device_aware_sweep.py`

Simpler device-aware experiment runner that extends existing sweep scripts.

### Usage:
```bash
# Use with pre-generated configs
python scripts/run_device_aware_sweep.py --config_dir sweep_configs --devices auto

# Generate configs dynamically and distribute across devices
python scripts/run_device_aware_sweep.py --sae_types relu,hard_concrete --devices cuda:0,cuda:1

# Show available devices
python scripts/run_device_aware_sweep.py --show_devices

# Test with limited experiments
python scripts/run_device_aware_sweep.py --sae_types relu --limit 8 --dry_run
```

### Advantages:
- ✅ Extends existing sweep approaches
- ✅ Automatic device detection
- ✅ Works with both pre-generated and dynamic configs
- ✅ Simple round-robin device distribution
- ✅ Device-specific performance metrics

### Disadvantages:
- ❌ Less sophisticated than full multi-device scheduler
- ❌ No dynamic load balancing

## Hyperparameter Grids

All approaches use the following parameter grids:

### Common Parameters (All SAE Types):
- **Learning Rate**: [1e-4, 3e-4, 1e-3]
- **Dictionary Size Ratio**: [8.0, 16.0, 32.0, 64.0]
- **Sparsity Coefficient**: [0.01, 0.03, 0.1, 0.3]

### SAE-Specific Parameters:

**ReLU SAE:**
- Only common parameters
- **Total combinations**: 3 × 4 × 4 = 48

**Hard Concrete SAE:**
- **Initial Beta**: [0.1, 0.5, 1.0]
- **Input Dependent Gates**: [True, False]
- **Total combinations**: 3 × 4 × 4 × 3 × 2 = 288

**Gated SAE:**
- **Auxiliary Coefficient**: [0.01, 0.03125, 0.1]
- **Total combinations**: 3 × 4 × 4 × 3 = 144

**Gated Hard Concrete SAE:**
- **Auxiliary Coefficient**: [0.01, 0.03125, 0.1]
- **Initial Beta**: [0.1, 0.5, 1.0]
- **Total combinations**: 3 × 4 × 4 × 3 × 3 = 432

**Grand total**: 912 experiments across all SAE types

## Recommendations

### For Development & Testing:
Use **Dynamic Python Runner** (`run_dynamic_sweep.py`) with `--limit` flag for quick iteration.

### For Systematic Research:
Use **Wandb Sweeps** (`setup_wandb_sweep.py`) for professional experiment tracking and analysis.

### For Simple Local Runs:
Use **Bash Script** (`run_simple_sweep.sh`) for straightforward, dependency-minimal execution.

### For Complex Workflows:
Use **Config Generator + Runner** (`generate_sweep_configs.py` + `run_sweep.py`) when you need to inspect/modify individual configs.

### For Multi-GPU Systems:
Use **Multi-Device Scheduler** (`run_multi_device_sweep.py`) for sophisticated device management with automatic load balancing.

### For Simple Multi-Device Setups:
Use **Device-Aware Runner** (`run_device_aware_sweep.py`) for straightforward distribution across available devices.

## Parallel Execution Tips

- **GPU Memory**: If running on GPU, limit parallel jobs based on GPU memory (typically 1-2 for large models)
- **CPU/RAM**: For CPU runs, you can typically run 2-4 parallel jobs per CPU core
- **I/O Bound**: If using cloud storage, fewer parallel jobs may be better to avoid I/O bottlenecks

## Example Workflows

### Quick Development Test:
```bash
# Test a few configs quickly
python scripts/run_dynamic_sweep.py --sae_types relu --limit 3 --dry_run
python scripts/run_dynamic_sweep.py --sae_types relu --limit 3
```

### Full Research Sweep:
```bash
# Setup Wandb sweeps for all SAE types
for sae_type in relu hard_concrete gated gated_hard_concrete; do
    python scripts/setup_wandb_sweep.py --sae_type $sae_type --project my-sae-research --create_agent_script
done

# Run agents (can be on different machines)
# wandb agent <sweep_id_1> --program scripts/sweep_agent_relu.py
# wandb agent <sweep_id_2> --program scripts/sweep_agent_hard_concrete.py
# etc.
```

### Local Parallel Sweep:
```bash
# Generate configs and run with limited parallelism
python scripts/generate_sweep_configs.py --output_dir sweep_configs
python scripts/run_sweep.py --config_dir sweep_configs --max_parallel 2
```

### Multi-GPU Acceleration:
```bash
# Auto-detect all devices and run sophisticated scheduler
python scripts/run_multi_device_sweep.py --sae_types relu,hard_concrete,gated --devices auto

# Use specific GPUs with advanced threading
python scripts/run_multi_device_sweep.py --devices cuda:0,cuda:1,cuda:2 --limit 50
```

### Simple Multi-Device Distribution:
```bash
# Quick device-aware distribution
python scripts/run_device_aware_sweep.py --config_dir sweep_configs --devices auto

# Test device detection and distribution
python scripts/run_device_aware_sweep.py --show_devices
python scripts/run_device_aware_sweep.py --sae_types relu --limit 8 --dry_run
```

## Modifying Hyperparameter Grids

To modify the parameter grids, edit the relevant files:

- **Config Generator**: Edit the `sae_configs` dictionary in `generate_sweep_configs.py`
- **Dynamic Runner**: Edit the `param_grids` dictionary in `run_dynamic_sweep.py`  
- **Bash Script**: Edit the array declarations at the top of `run_simple_sweep.sh`
- **Wandb Sweeps**: Edit the `param_configs` dictionary in `setup_wandb_sweep.py`
- **Multi-Device Scheduler**: Edit the `param_grids` dictionary in `run_multi_device_sweep.py`
- **Device-Aware Runner**: Edit the `param_grids` dictionary in `run_device_aware_sweep.py`

All approaches are designed to be easily customizable for your specific research needs!

## Multi-Device Performance Benefits

When you have multiple GPUs available, the multi-device approaches can provide significant speedups:

- **2 GPUs**: ~1.8-2x speedup (near-linear scaling)
- **4 GPUs**: ~3.5-4x speedup (excellent scaling)
- **Mixed devices** (GPUs + CPU): CPU handles overflow when GPUs are busy

### Device Detection Output Example:
```
Found CUDA device: cuda:0 (24.0GB) - NVIDIA RTX 4090
Found CUDA device: cuda:1 (24.0GB) - NVIDIA RTX 4090  
Found CPU device

Using devices: ['cuda:0', 'cuda:1', 'cpu']
Experiment distribution:
  cuda:0: 34 experiments
  cuda:1: 33 experiments  
  cpu: 33 experiments
```

The multi-device schedulers automatically balance the workload and provide detailed per-device performance metrics, making it easy to optimize your experimental throughput. 