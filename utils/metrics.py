import torch
from jaxtyping import Float
from torch import Tensor
from models import SAETransformerOutput


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
    sample_dims = tuple(range(pred.ndim - 1))
    per_token_l2_loss = (pred - target).pow(2).sum(dim=-1)
    total_variance = (target - target.mean(dim=sample_dims)).pow(2).sum(dim=-1)
    return 1 - per_token_l2_loss / total_variance


@torch.inference_mode()
def reconstruction_metrics(output: SAETransformerOutput,) -> dict[str, float]:
    """Get metrics on the outputs of the SAE-augmented model and the original model.

    Args:
        output: The output of the SAE-augmented Transformer.

    Returns:
        Dictionary of output metrics
    """
    reconstruction_metrics = {}
    for name, sae_output in output.sae_outputs.items():
        var = explained_variance(
            sae_output.input.detach().clone(), sae_output.output, layer_norm_flag=False
        )
        var_ln = explained_variance(
            sae_output.input.detach().clone(), sae_output.output, layer_norm_flag=True
        )
        reconstruction_metrics[f"explained_variance/{name}"] = var.mean()
        reconstruction_metrics[f"explained_variance_std/{name}"] = var.std()
        reconstruction_metrics[f"explained_variance_ln/{name}"] = var_ln.mean()
        reconstruction_metrics[f"explained_variance_ln_std/{name}"] = var_ln.std()
    return reconstruction_metrics


@torch.inference_mode()
def sparsity_metrics(output: SAETransformerOutput) -> dict[str, float]:
    """Collect sparsity metrics for logging.

    Args:
        output: The output of the SAE-augmented Transformer.

    Returns:
        Dictionary of sparsity metrics.
    """
    sparsity_metrics = {}
    for name, sae_output in output.sae_outputs.items():
        
        # Use consistent L0 norm calculation for all SAE types
        # Hard Concrete SAEs now apply thresholding in their forward pass during evaluation
        l_0_norm = torch.norm(sae_output.c, p=0, dim=-1).mean().item()
        frac_zeros = ((sae_output.c == 0).sum() / sae_output.c.numel()).item()

        # Store with consistent naming for cross-SAE comparison
        sparsity_metrics[f"L_0/{name}"] = l_0_norm
        sparsity_metrics[f"frac_zeros/{name}"] = frac_zeros

    return sparsity_metrics


def loss_metrics(output: SAETransformerOutput) -> dict[str, float]:
    """Calculate loss metrics for the output of the SAE-augmented Transformer.
    """
    loss_metrics = {}
    for name, loss_output in output.loss_outputs.items():
        for loss_name, loss_value in loss_output.loss_dict.items():
            loss_metrics[f"loss/{name}/{loss_name}"] = loss_value.item()
    return loss_metrics


def all_metrics(output: SAETransformerOutput, train: bool = True) -> dict[str, float]:
    """Calculate all metrics for the output of the SAE-augmented Transformer.

    Args:
        output: The output of the SAE-augmented Transformer.

    Returns:
        Dictionary of all metrics.
    """
    prefix = "train" if train else "eval"
    return {
        **{f"{prefix}/{k}": v for k, v in reconstruction_metrics(output).items()},
        **{f"{prefix}/{k}": v for k, v in sparsity_metrics(output).items()},
        **{f"{prefix}/{k}": v for k, v in loss_metrics(output).items()},
    }