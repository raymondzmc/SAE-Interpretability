# Wandb Artifacts Structure Overview

This document provides a comprehensive overview of how activation data and evaluation results are organized in Wandb artifacts.

## Artifact Types

### 1. Activation Data Artifacts (`activation_data`)

**Artifact Name Pattern:** `activation_data_{run_id}`  
**Type:** `activation_data`

#### File Structure:
```
activation_data/
├── blocks--0--mlp--hook_post.pt          # Layer 0 MLP activations
├── blocks--1--mlp--hook_post.pt          # Layer 1 MLP activations  
├── blocks--2--mlp--hook_post.pt          # Layer 2 MLP activations
├── ...                                   # Additional layers
└── all_token_ids.pt                      # Token sequences (optional)
```

#### Content Details:
- **Layer Files (*.pt)**: Each contains a dictionary with:
  - `'nonzero_activations'`: `torch.Tensor` of shape `(N, 64)` - activation values
  - `'data_indices'`: `torch.Tensor` of shape `(N,)` - global data indices  
  - `'neuron_indices'`: `torch.Tensor` of shape `(N,)` - neuron indices

- **Token IDs File (`all_token_ids.pt`)**: List of token sequences
  - Type: `list[list[str]]`
  - Each inner list contains 64 token strings

#### Metadata:
```json
{
  "run_id": "abc123def",
  "run_name": "relu_sweep_run_1", 
  "num_layers": 3,
  "layer_names": ["blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post", "blocks.2.mlp.hook_post"],
  "has_token_ids": true
}
```

### 2. Evaluation Results Artifacts (`evaluation_results`)

**Artifact Name Pattern:** `evaluation_results_{run_id}`  
**Type:** `evaluation_results`

#### File Structure:
```
evaluation_results/
├── evaluation_metrics.json              # Layer-wise performance metrics
├── explanations.json                    # Neuron explanations and scores
└── summary_stats.json                   # Summary statistics
```

#### Content Details:

**evaluation_metrics.json:**
```json
{
  "blocks.0.mlp.hook_post": {
    "alive_dict_components": 245,
    "alive_dict_components_proportion": 0.479,
    "sparsity_l0": 12.34,
    "mse": 0.0023,
    "explained_variance": 0.89
  },
  "blocks.1.mlp.hook_post": { ... }
}
```

**explanations.json:**
```json
{
  "blocks.0.mlp.hook_post_neuron_42": {
    "text": "This neuron activates on mathematical expressions...",
    "score": 0.73,
    "sae_position": "blocks.0.mlp.hook_post", 
    "neuron_index": 42,
    "num_examples": 25
  },
  "blocks.0.mlp.hook_post_neuron_137": { ... }
}
```

**summary_stats.json:**
```json
{
  "num_layers": 3,
  "layer_names": ["blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post", "blocks.2.mlp.hook_post"],
  "num_explanations": 156,
  "explained_neurons_per_layer": {
    "blocks.0.mlp.hook_post": 52,
    "blocks.1.mlp.hook_post": 48, 
    "blocks.2.mlp.hook_post": 56
  }
}
```

#### Metadata:
```json
{
  "run_id": "abc123def",
  "run_name": "relu_sweep_run_1",
  "num_layers": 3,
  "layer_names": ["blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post", "blocks.2.mlp.hook_post"],
  "num_explanations": 156,
  "explained_neurons_per_layer": {
    "blocks.0.mlp.hook_post": 52,
    "blocks.1.mlp.hook_post": 48,
    "blocks.2.mlp.hook_post": 56
  }
}
```

## Usage Examples

### Uploading Data
```python
from utils.io import save_activation_data, save_evaluation_results_to_wandb

# Upload activation data
save_activation_data(
    accumulated_data=activation_dict,
    run_dir="local/path", 
    upload_to_wandb=True,
    run_id="abc123def",
    run_name="relu_sweep_run_1",
    all_token_ids=token_sequences
)

# Upload evaluation results 
save_evaluation_results_to_wandb(
    metrics=metrics_dict,
    explanations=explanations_dict,
    run_id="abc123def", 
    run_name="relu_sweep_run_1"
)
```

### Downloading Data
```python
from utils.io import load_activation_data_from_wandb, load_evaluation_results_from_wandb

# Download activation data
activation_data, token_ids = load_activation_data_from_wandb(
    run_id="abc123def",
    project="raymondl/tinystories-1m"
)

# Download evaluation results
metrics, explanations = load_evaluation_results_from_wandb(
    run_id="abc123def", 
    project="raymondl/tinystories-1m"
)
```

## Benefits

1. **Storage Efficiency**: Large activation files stored in Wandb cloud
2. **Reproducibility**: Data accessible from any machine with Wandb access
3. **Version Control**: Wandb artifacts provide versioning and lineage tracking
4. **Rich Metadata**: Comprehensive metadata for easy discovery and filtering
5. **Collaboration**: Easy sharing of large datasets between team members

## File Size Considerations

- **Activation Data**: Can be several GB per run depending on:
  - Number of SAE positions (layers)
  - Number of samples evaluated (N_EVAL_SAMPLES)  
  - Sparsity levels (affects number of nonzero activations)

- **Evaluation Results**: Typically <100MB per run:
  - Metrics: Small JSON files
  - Explanations: Text-based, scales with number of explained neurons

## Configuration

Set `UPLOAD_TO_WANDB = True` in `evaluation.py` to enable automatic uploads.
Requires valid Wandb authentication and project access. 