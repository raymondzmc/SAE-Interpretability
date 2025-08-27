import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss
from models import SAETransformerOutput
from utils.enums import SAEType


def layer_norm(x: Float[Tensor, "... dim"], eps: float = 1e-5) -> Float[Tensor, "... dim"]:
    """Layernorm without the affine transformation."""
    x = x - x.mean(dim=-1, keepdim=True)
    scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return x / scale


def explained_variance(
    pred: Float[Tensor, "... dim"], target: Float[Tensor, "... dim"], layer_norm_flag: bool = False
) -> Float[Tensor, "..."]:
    """Calculate the explained variance of the pred and target.

    Args:
        pred: The prediction to compare to the target.
        target: The target to compare the prediction to.
        layer_norm_flag: Whether to apply layer norm to the pred and target before calculating the loss.

    Returns:
        The explained variance between the prediction and target for each batch and sequence pos.
    """
    if layer_norm_flag:
        pred = layer_norm(pred)
        target = layer_norm(target)
    
    # Calculate residual sum of squares (squared errors)
    residual_sum_of_squares = (pred - target).pow(2).sum(dim=-1)
    
    # Calculate total sum of squares (variance in target)
    # We want to compute variance across the feature dimension only
    target_mean = target.mean(dim=-1, keepdim=True)
    total_sum_of_squares = (target - target_mean).pow(2).sum(dim=-1)
    
    # Explained variance: 1 - (RSS / TSS)
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    explained_var = 1 - residual_sum_of_squares / (total_sum_of_squares + eps)
    
    return explained_var


def get_activations_for_sae_type(sae_output, sae_type: SAEType) -> torch.Tensor:
    """Get the appropriate activations tensor based on SAE type.
    
    This matches the logic in evaluation.py lines 208-221.
    """
    if sae_type == SAEType.HARD_CONCRETE:
        return sae_output.c
    elif sae_type == SAEType.LAGRANGIAN_HARD_CONCRETE:
        return sae_output.z
    elif sae_type == SAEType.RELU:
        return sae_output.c
    elif sae_type == SAEType.GATED:
        return sae_output.z if hasattr(sae_output, 'z') else sae_output.c
    elif sae_type == SAEType.TOPK:
        return sae_output.code
    elif sae_type == SAEType.GUMBEL_TOPK:
        return sae_output.z_st
    else:
        return sae_output.c  # Default to main activations


@torch.inference_mode()
def reconstruction_metrics(output: SAETransformerOutput, sae_type: SAEType) -> dict[str, float]:
    """Get metrics on the outputs of the SAE-augmented model and the original model.

    Args:
        output: The output of the SAE-augmented Transformer.
        sae_type: The type of SAE being used.

    Returns:
        Dictionary of output metrics
    """
    reconstruction_metrics = {}
    for name, sae_output in output.sae_outputs.items():
        # Compute MSE loss (same as evaluation.py)
        mse = mse_loss(
            sae_output.output,
            sae_output.input,
            reduction='mean'
        ).item()
        
        # Fix parameter order: pred first, target second
        var = explained_variance(
            sae_output.output, sae_output.input.detach().clone(), layer_norm_flag=False
        )
        var_ln = explained_variance(
            sae_output.output, sae_output.input.detach().clone(), layer_norm_flag=True
        )
        
        # Add MSE to the metrics
        reconstruction_metrics[f"mse/{name}"] = mse
        reconstruction_metrics[f"explained_variance/{name}"] = var.mean()
        reconstruction_metrics[f"explained_variance_ln/{name}"] = var_ln.mean()
    return reconstruction_metrics


@torch.inference_mode()
def sparsity_metrics(output: SAETransformerOutput, sae_type: SAEType, n_dict_components: dict[str, int] | None = None) -> dict[str, float]:
    """Collect sparsity metrics for logging.

    Args:
        output: The output of the SAE-augmented Transformer.
        sae_type: The type of SAE being used.
        n_dict_components: Dictionary mapping SAE position to number of dictionary components.

    Returns:
        Dictionary of sparsity metrics.
    """
    sparsity_metrics = {}
    for name, sae_output in output.sae_outputs.items():
        # Get activations based on SAE type (consistent with evaluation.py)
        acts = get_activations_for_sae_type(sae_output, sae_type)
        
        # Use consistent L0 norm calculation for all SAE types
        l_0_norm = torch.norm(acts, p=0, dim=-1).mean().item()
        frac_zeros = ((acts == 0).sum() / acts.numel()).item()
        
        # Compute alive dictionary components (like evaluation.py)
        if n_dict_components is not None and name in n_dict_components:
            # Get indices of non-zero (alive) activations
            nonzero_indices = acts.sum(0).sum(0).nonzero().squeeze().cpu()
            if nonzero_indices.numel() == 0:
                alive_indices = []
            elif nonzero_indices.numel() == 1:
                alive_indices = [nonzero_indices.item()]
            else:
                alive_indices = nonzero_indices.tolist()
            
            num_alive = len(alive_indices)
            sparsity_metrics[f"alive_dict_components/{name}"] = num_alive

        # Store with consistent naming for cross-SAE comparison
        sparsity_metrics[f"L_0/{name}"] = l_0_norm

    return sparsity_metrics


def loss_metrics(output: SAETransformerOutput) -> dict[str, float]:
    """Calculate loss metrics for the output of the SAE-augmented Transformer.
    """
    loss_metrics = {}
    for name, loss_output in output.loss_outputs.items():
        for loss_name, loss_value in loss_output.loss_dict.items():
            loss_metrics[f"loss/{name}/{loss_name}"] = loss_value.item()
    return loss_metrics


def all_metrics(output: SAETransformerOutput, train: bool = True, sae_type: SAEType | None = None, n_dict_components: dict[str, int] | None = None) -> dict[str, float]:
    """Calculate all metrics for the output of the SAE-augmented Transformer.

    Args:
        output: The output of the SAE-augmented Transformer.
        train: Whether this is for training (True) or evaluation (False).
        sae_type: The type of SAE being used (needed for correct activation selection).
        n_dict_components: Dictionary mapping SAE position to number of dictionary components.

    Returns:
        Dictionary of all metrics.
    """
    prefix = "train" if train else "eval"
    
    # Use SAEType.RELU as default if not provided (for backwards compatibility)
    if sae_type is None:
        sae_type = SAEType.RELU
    
    return {
        **{f"{prefix}/{k}": v for k, v in reconstruction_metrics(output, sae_type).items()},
        **{f"{prefix}/{k}": v for k, v in sparsity_metrics(output, sae_type, n_dict_components).items()},
        **{f"{prefix}/{k}": v for k, v in loss_metrics(output).items()},
    }