# SAE-Interpretability

A comprehensive framework for training and evaluating Sparse Autoencoders (SAEs) on transformer models, with support for multiple SAE architectures, hyperparameter sweeps, and multi-device training.

## 🎯 Overview

This repository provides tools for:
- **Training SAEs** on TransformerLens models (TinyStories, GPT-2, etc.)
- **Multiple SAE architectures**: ReLU, Hard Concrete, Gated, Gated Hard Concrete
- **Hyperparameter sweeps** with automatic experiment management
- **Multi-device training** with tmux-based monitoring
- **Experiment tracking** via Weights & Biases integration

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch with appropriate CUDA support (if using GPU)
- tmux (for experiment monitoring)

### Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd SAE-Interpretability

# Create conda environment (recommended)
conda create -n sae python=3.9
conda activate sae

# Install dependencies
pip install torch torchvision transformerlens datasets wandb
pip install pydantic jaxtyping tqdm huggingface_hub
```

### Configure Weights & Biases (Optional)
```bash
wandb login
# Follow prompts to authenticate
```

## 🚀 Quick Start

### Single Experiment
Train a ReLU SAE on TinyStories:
```bash
python run.py --config_path configs/tinystories/tinystories-relu.yaml --device cuda:0
```

### Simple Hyperparameter Sweep
Create a sweep config and run experiments:
```bash
# Create sweep config
cat > my_sweep.yaml << EOF
learning_rates: [1e-4, 3e-4, 1e-3]
sparsity_coeffs: [0.01, 0.03, 0.1]
dict_size_ratios: [16.0, 32.0]
max_experiments: 6
EOF

# Run sweep with automatic device detection
python run_experiments.py \
  --base_config configs/tinystories/tinystories-relu.yaml \
  --sweep_config my_sweep.yaml
```

### Multi-Device Training
```bash
# Run on specific devices
python run_experiments.py \
  --base_config configs/tinystories/tinystories-relu.yaml \
  --sweep_config my_sweep.yaml \
  --devices cuda:0,cuda:1,cpu

# Monitor experiments in real-time
python monitor_experiments.py --watch
```

## ⚙️ Configuration System

### Base Configuration Files
Located in `configs/`:
- `tinystories/tinystories-relu.yaml` - ReLU SAE configuration
- `tinystories/tinystories-gated.yaml` - Gated SAE configuration  
- `tinystories/tinystories-hardconcrete.yaml` - Hard Concrete SAE configuration
- `tinystories/tinystories-gated-hardconcrete.yaml` - Combined architecture

### Configuration Structure
```yaml
# Experiment Settings
wandb_project: "my-project"
wandb_run_name: null  # Auto-generated from parameters
wandb_tags: ["relu"]
seed: 42

# Model Configuration
tlens_model_name: "roneneldan/TinyStories-1M"
tlens_model_path: null

# Training Configuration
lr: 1e-3
lr_schedule: "cosine"
gradient_accumulation_steps: 20
warmup_samples: 20_000
max_grad_norm: 1.0

# Data Configuration
data:
  dataset_name: "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
  tokenizer_name: "roneneldan/TinyStories"
  context_length: 1024
  n_train_samples: 100_000
  train_batch_size: 20

# SAE Configuration
saes:
  sae_type: "relu"  # or "gated", "hard_concrete", "gated_hard_concrete"
  dict_size_to_input_ratio: 32.0
  sparsity_coeff: 0.1
  sae_positions: ["blocks.2.hook_resid_pre", "blocks.4.hook_resid_pre"]
```

### Sweep Configuration
Define parameter grids for hyperparameter sweeps:
```yaml
# Top-level parameters
learning_rates: [1e-4, 3e-4, 1e-3]
gradient_accumulation_steps: [20, 40]

# Data parameters  
n_train_samples: [50_000, 100_000]
train_batch_size: [20, 40]

# SAE parameters
sparsity_coeffs: [0.01, 0.03, 0.1]
dict_size_ratios: [16.0, 32.0, 64.0]

# SAE-specific parameters
sae_specific:
  aux_coeffs: [0.03125, 0.0625]        # For gated SAEs
  initial_betas: [0.5, 1.0]            # For hard concrete SAEs
  input_dependent_gates: [true, false]  # For hard concrete SAEs

# Experiment limits
max_experiments: 20
```

## 🧠 SAE Architectures

### 1. ReLU SAE (`relu`)
Standard sparse autoencoder with ReLU activations.
```yaml
saes:
  sae_type: "relu"
  sparsity_coeff: 0.1
```

### 2. Gated SAE (`gated`)
Uses gating mechanism with auxiliary loss.
```yaml
saes:
  sae_type: "gated"
  sparsity_coeff: 0.1
  aux_coeff: 0.03125
```

### 3. Hard Concrete SAE (`hard_concrete`)
Learnable sparse gates with concrete distribution.
```yaml
saes:
  sae_type: "hard_concrete"
  sparsity_coeff: 0.1
  initial_beta: 0.5
  input_dependent_gates: true
  beta_annealing: true
```

### 4. Gated Hard Concrete SAE (`gated_hard_concrete`)
Combines gating and hard concrete mechanisms.
```yaml
saes:
  sae_type: "gated_hard_concrete"
  sparsity_coeff: 0.1
  aux_coeff: 0.03125
  initial_beta: 0.5
```

## 🔬 Running Experiments

### Single Experiment with run.py
```bash
# Basic training
python run.py --config_path configs/tinystories/tinystories-relu.yaml

# Override device
python run.py --config_path configs/tinystories/tinystories-relu.yaml --device cpu

# Disable wandb logging
python run.py --config_path configs/tinystories/tinystories-relu.yaml --wandb_project null
```

### Hyperparameter Sweeps with run_experiments.py

#### Basic Sweep
```bash
python run_experiments.py \
  --base_config configs/tinystories/tinystories-relu.yaml \
  --sweep_config my_sweep.yaml
```

#### Advanced Options
```bash
python run_experiments.py \
  --base_config configs/tinystories/tinystories-gated.yaml \
  --sweep_config comprehensive_sweep.yaml \
  --devices cuda:0,cuda:1,mps,cpu \
  --limit 10 \
  --output_dir my_experiment_configs
```

#### Sequential Debugging Mode
```bash
# Run experiments one by one for easier debugging
python run_experiments.py \
  --base_config configs/tinystories/tinystories-relu.yaml \
  --sweep_config debug_sweep.yaml \
  --sequential \
  --limit 2
```

#### Disable tmux Sessions
```bash
# Use direct subprocess calls instead of tmux
python run_experiments.py \
  --base_config configs/tinystories/tinystories-relu.yaml \
  --sweep_config my_sweep.yaml \
  --no_tmux
```

### Device Management
```bash
# Show available devices
python run_experiments.py --show_devices

# Auto-detect devices
python run_experiments.py ... --devices auto

# Specify devices explicitly  
python run_experiments.py ... --devices cuda:0,cuda:1,cpu

# Force CPU only
python run_experiments.py ... --devices cpu,cpu,cpu
```

## 📊 Monitoring and Outputs

### Real-time Monitoring
```bash
# Show current experiment status
python monitor_experiments.py

# Live monitoring dashboard
python monitor_experiments.py --watch

# Interactive session attachment
python monitor_experiments.py --attach

# Kill all running experiments
python monitor_experiments.py --kill
```

### tmux Session Management
```bash
# List all tmux sessions
tmux list-sessions

# Attach to specific experiment
tmux attach -t exp_experiment_000_cuda_0

# Kill specific experiment
tmux kill-session -t exp_experiment_000_cuda_0

# Detach from session (while attached)
Ctrl+B, D
```

### Output Structure
```
📁 Repository Structure
├── output/                              # 🎯 TRAINED MODELS (MOST IMPORTANT)
│   ├── relu_lr_1e-3_sparsity_0.1_2024-XX-XX/
│   │   ├── final_config.yaml           # Complete config used
│   │   └── samples_100000.pt           # Trained SAE checkpoint
│   └── gated_lr_3e-4_aux_0.03_2024-XX-XX/
│       ├── final_config.yaml
│       └── samples_100000.pt
├── experiment_outputs/                  # Temporary experiment configs
│   ├── experiment_000.yaml            # Auto-deleted after completion
│   └── experiment_001.yaml
├── wandb/                              # Experiment tracking
│   ├── run-2024XXXX-XXXXX/           # Individual run logs/metrics
│   └── latest-run -> run-...          # Symlink to most recent
├── configs/                            # Base configurations
│   ├── tinystories/
│   │   ├── tinystories-relu.yaml
│   │   ├── tinystories-gated.yaml
│   │   └── ...
│   └── gpt2/
└── scripts/                           # Alternative sweep tools
    ├── run_multi_device_sweep.py     # Advanced multi-device scheduler
    ├── setup_wandb_sweep.py          # Wandb cloud sweeps
    └── ...
```

### Important Files
- **`output/*/samples_*.pt`** - Your trained SAE models (PyTorch checkpoints)
- **`output/*/final_config.yaml`** - Exact configuration used for training
- **Wandb dashboard** - Training metrics, loss curves, hyperparameter comparisons

## 🛠️ Utility Scripts

### Experiment Generation
```bash
# Generate multiple config files
python scripts/generate_sweep_configs.py

# Setup Wandb cloud sweeps
python scripts/setup_wandb_sweep.py
```

### Alternative Sweep Tools
```bash
# Dynamic in-memory sweeps
python scripts/run_dynamic_sweep.py

# Advanced multi-device scheduler
python scripts/run_multi_device_sweep.py

# Simple bash-based sweeps
./scripts/run_simple_sweep.sh
```

## 🔧 Advanced Usage

### Custom SAE Implementation
1. Create new SAE class in `models/saes/`
2. Register in `models/saes/__init__.py`
3. Add configuration support in `config.py`
4. Update `utils/enums.py` with new SAE type

### Custom Dataset
```yaml
data:
  dataset_name: "your-org/your-dataset"
  tokenizer_name: "your-tokenizer"
  context_length: 1024
  column_name: "input_ids"  # Column containing tokenized text
```

### Environment Configuration
Create `settings.py` overrides:
```python
# settings.py
WANDB_API_KEY = "your-api-key"
WANDB_ENTITY = "your-entity"
HF_ACCESS_TOKEN = "your-hf-token"
```

## 🐛 Troubleshooting

### Common Issues

#### Experiments finish too quickly
- Check wandb logs for errors
- Verify config parameters (especially `n_train_samples`)
- Run single experiment with `--no_tmux` for direct output

#### CUDA out of memory
```yaml
# Reduce batch size or gradient accumulation
gradient_accumulation_steps: 40  # Increase this
data:
  train_batch_size: 10          # Decrease this
```

#### tmux sessions not found
```bash
# Check if tmux is installed
tmux -V

# If not installed:
# macOS: brew install tmux
# Ubuntu: apt install tmux

# Disable tmux if needed
python run_experiments.py ... --no_tmux
```

#### Wandb authentication issues
```bash
wandb login --relogin
# Or disable wandb
python run.py ... --wandb_project null
```

### Debugging Commands
```bash
# Test single experiment
python run.py --config_path configs/tinystories/tinystories-relu.yaml --device cpu

# Dry run sweep
python run_experiments.py ... --dry_run

# Sequential execution for debugging
python run_experiments.py ... --sequential --limit 2

# Check device availability
python run_experiments.py --show_devices
```

### Performance Optimization
```bash
# Multi-device training
python run_experiments.py ... --devices cuda:0,cuda:1,cuda:2

# Increase parallelism
python run_experiments.py ... --devices cpu,cpu,cpu,cpu  # 4 CPU workers

# Optimize batch size
gradient_accumulation_steps: 20
data:
  train_batch_size: 40  # Effective batch = 20 * 40 = 800
```

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `run.py` | Single experiment training |
| `run_experiments.py` | Hyperparameter sweeps with multi-device support |
| `monitor_experiments.py` | Experiment monitoring and management |
| `config.py` | Configuration schema definitions |
| `settings.py` | Environment and API key configuration |
| `models/saes/` | SAE architecture implementations |
| `data/` | Data loading and preprocessing |
| `utils/` | Shared utilities and helpers |

## 🤝 Contributing

1. Add new SAE architectures in `models/saes/`
2. Extend configuration schema in `config.py`
3. Add utility functions in `utils/`
4. Update documentation for new features

## 📄 License

[Add your license information here]

---

🚀 **Happy SAE Training!** For questions or issues, please check the troubleshooting section or open an issue.