"""Script for training SAEs on top of a transformerlens model.

Usage:
    python main.py <path/to/config.yaml>
"""

from pathlib import Path
from datetime import datetime

import torch
import wandb
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader
from huggingface_hub import login as hf_login
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from data import create_dataloaders
from models import (
    SAETransformer,
    SAETransformerOutput,
    HardConcreteSAEConfig,
    HardConcreteSAE,
)
from models.loader import load_tlens_model, load_pretrained_saes
from utils.enums import SAEType
from utils.misc import set_seed, get_run_name
from utils.io import load_config, save_module
from utils.constants import CONFIG_FILE
from utils.logging import logger
from utils.schedulers import (
    get_linear_lr_schedule,
    get_cosine_schedule_with_warmup,
    get_exponential_beta_schedule,
)
from utils.metrics import all_metrics
from settings import settings
from config import Config


@torch.inference_mode()
def evaluate(
    config: Config,
    model: SAETransformer,
    eval_loader: DataLoader,
    device: torch.device,
    cache_positions: list[str] | None,
) -> dict[str, float]:
    """Evaluate the model on the eval dataset.

    Accumulates metrics over the entire eval dataset and then divides by the total number of tokens.

    Args:
        config: The config object.
        model: The SAETransformer model.
        device: The device to run the model on.
        cache_positions: The positions to cache activations at.
    Returns:
        Dictionary of metrics.
    """
    model.saes.eval()

    eval_cache_positions = cache_positions
    total_tokens = 0
    accumulated_metrics: dict[str, float] = {}

    for batch in tqdm(eval_loader, desc="Eval Steps"):
        tokens = batch[config.data.column_name].to(device=device)
        n_tokens = tokens.shape[0] * tokens.shape[1]
        total_tokens += n_tokens

        # Run through the SAE-augmented model
        output: SAETransformerOutput = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_positions,
            cache_positions=eval_cache_positions,
            compute_loss=True,
        )
        batch_metrics = all_metrics(output, train=False)
        
        for k, v in batch_metrics.items():
            accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v * n_tokens

    # Get the mean for all metrics
    for key in accumulated_metrics:
        accumulated_metrics[key] /= total_tokens

    model.saes.train()
    return accumulated_metrics


@logging_redirect_tqdm()
def train(
    config: Config,
    model: SAETransformer,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    trainable_param_names: list[str],
    device: torch.device,
    cache_positions: list[str] | None = None,
) -> None:
    model.saes.train()

    # TODO: Handling end-to-end training
    is_local = True

    for name, param in model.named_parameters():
        if name.startswith("saes.") and name.split("saes.")[1] in trainable_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )

    if config.lr_schedule == "cosine":
        assert config.data.n_train_samples is not None, "Cosine schedule requires n_samples."
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_samples // config.effective_batch_size,
            num_training_steps=config.data.n_train_samples // config.effective_batch_size,
            min_lr_factor=config.min_lr_factor,
        )
    else:
        assert config.lr_schedule == "linear"
        lr_schedule = get_linear_lr_schedule(
            warmup_samples=config.warmup_samples,
            cooldown_samples=config.cooldown_samples,
            n_samples=config.data.n_train_samples,
            effective_batch_size=config.effective_batch_size,
            min_lr_factor=config.min_lr_factor,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    # Prepare beta annealing schedule for Hard Concrete SAEs
    beta_schedule = None
    if config.saes.sae_type == SAEType.HARD_CONCRETE and config.saes.beta_annealing:
        total_steps = config.data.n_train_samples // config.effective_batch_size
        warmup_steps = config.warmup_samples // config.effective_batch_size
        hc_config = config.saes if isinstance(config.saes, HardConcreteSAEConfig) else None
        if hc_config is None:
            raise ValueError("Expected HardConcreteSAEConfig for Hard Concrete SAE type")
        beta_schedule = get_exponential_beta_schedule(
            initial_beta=hc_config.initial_beta,
            final_beta=hc_config.final_beta,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

    stop_at_layer = None
    if all(name.startswith("blocks.") for name in model.raw_sae_positions) and is_local:
        # We don't need to run through the whole model for local runs
        stop_at_layer = max([int(name.split(".")[1]) for name in model.raw_sae_positions]) + 1

    run_name = get_run_name(config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.save_dir / f"{run_name}_{timestamp}" if config.save_dir else None

    total_samples = 0
    total_samples_at_last_save = 0
    total_samples_at_last_eval = 0
    total_tokens = 0
    grad_updates = 0
    grad_norm: float | None = None
    samples_since_act_frequency_collection: int = 0

    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Steps"):
        tokens: Int[Tensor, "batch pos"] = batch[config.data.column_name].to(device=device)

        # Update beta in Hard Concrete SAE modules based on schedule
        current_beta = None
        if config.saes.sae_type == SAEType.HARD_CONCRETE and beta_schedule is not None:
            current_beta = beta_schedule(grad_updates)
            for sae_module in model.saes.modules():
                if isinstance(sae_module, HardConcreteSAE):
                    # Ensure beta is on the correct device
                    beta_tensor = torch.tensor(current_beta, device=sae_module.beta.device, dtype=sae_module.beta.dtype)
                    sae_module.beta.copy_(beta_tensor)

        total_samples += tokens.shape[0]
        total_tokens += tokens.shape[0] * tokens.shape[1]
        samples_since_act_frequency_collection += tokens.shape[0]

        is_last_batch: bool = (batch_idx == len(train_loader) - 1)
        is_grad_step: bool = (batch_idx + 1) % config.gradient_accumulation_steps == 0
        is_eval_step: bool = config.eval_every_n_samples is not None and (
            (batch_idx == 0)
            or total_samples - total_samples_at_last_eval >= config.eval_every_n_samples
            or is_last_batch
        )
        is_log_step: bool = (
            batch_idx == 0
            or (is_grad_step and (grad_updates + 1) % config.log_every_n_grad_steps == 0)
            or is_eval_step
            or is_last_batch
        )
        is_save_model_step: bool = save_dir is not None and (
            (
                config.save_every_n_samples
                and total_samples - total_samples_at_last_save >= config.save_every_n_samples
            )
            or is_last_batch
        )

        output: SAETransformerOutput = model.forward(
            tokens=tokens,
            sae_positions=model.raw_sae_positions,
            cache_positions=cache_positions,
            stop_at_layer=stop_at_layer,
            compute_loss=True,
        )
        # Validate and aggregate losses
        if not output.loss_outputs:
            raise ValueError("No loss outputs found. Check SAE compute_loss implementation.")
        
        loss = sum(loss_output.loss for loss_output in output.loss_outputs.values())
        loss /= config.gradient_accumulation_steps
        loss.backward()

        if is_grad_step:
            if config.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.saes.parameters(), config.max_grad_norm
                ).item()
            optimizer.step()
            optimizer.zero_grad()
            grad_updates += 1
            lr_scheduler.step()

        if is_log_step:
            tqdm.write(
                f"Samples {total_samples} Batch_idx {batch_idx} GradUpdates {grad_updates} "
                f"Loss {loss.item():.5f}"
            )
            if config.wandb_project:
                log_info = {
                    "loss": loss.item(),
                    "grad_updates": grad_updates,
                    "total_tokens": total_tokens,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if current_beta is not None:
                    log_info["beta"] = current_beta

                if grad_norm is not None:
                    log_info["grad_norm"] = grad_norm  # Norm of grad before clipping

                log_info.update(all_metrics(output, train=True))

                if is_eval_step and eval_loader is not None:
                    eval_metrics = evaluate(
                        config=config, model=model, eval_loader=eval_loader, device=device, cache_positions=cache_positions
                    )
                    total_samples_at_last_eval = total_samples
                    log_info.update(eval_metrics)

                wandb.log(log_info, step=total_samples)

        if is_save_model_step:
            assert save_dir is not None
            total_samples_at_last_save = total_samples
            save_module(
                config_dict=config.model_dump(mode="json"),
                save_dir=save_dir,
                module=model.saes,
                model_filename=f"samples_{total_samples}.pt",
                config_filename=CONFIG_FILE,
            )


        if is_last_batch:
            break

    # If the model wasn't saved at the last step of training (which may happen if n_samples: null
    # and the dataset is an IterableDataset), save it now.
    if save_dir and not (save_dir / f"samples_{total_samples}.pt").exists():
        save_module(
            config_dict=config.model_dump(mode="json"),
            save_dir=save_dir,
            module=model.saes,
            model_filename=f"samples_{total_samples}.pt",
            config_filename=CONFIG_FILE,
        )


    if config.wandb_project:
        wandb.finish()


def run(config_path_or_obj: Path | str | Config, device_override: str | None = None) -> None:
    # Set device context FIRST, before any model loading or tensor creation
    if device_override:
        device = torch.device(device_override)
        # Set default CUDA device to avoid device mismatches - do this FIRST
        if device.type == 'cuda':
            torch.cuda.set_device(device)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    
    config: Config = load_config(config_path_or_obj, config_model=Config)
    run_name = get_run_name(config)
    if config.wandb_project:
        if not wandb.api.api_key:
            if settings.wandb_api_key is not None:
                wandb.login(key=settings.wandb_api_key)
            else:
                wandb.login()

        wandb.init(
            entity=settings.wandb_entity,
            project=config.wandb_project,
            name=run_name,
            tags=config.wandb_tags,
            save_code=True
        )
        wandb.config.update(config.model_dump(mode="json"))
    
    if settings.has_hf_config():
        hf_login(token=settings.hf_access_token)

    set_seed(config.seed)
    logger.info(config)

    # Create train and eval loaders using simplified data config
    train_loader, eval_loader = create_dataloaders(
        data_config=config.data,
        global_seed=config.seed
    )
    tlens_model = load_tlens_model(
        tlens_model_name=config.tlens_model_name, tlens_model_path=config.tlens_model_path
    )
    
    # Force move to exact device (not just "cuda")
    print(f"Moving tlens_model from {next(tlens_model.parameters()).device} to {device}")
    tlens_model = tlens_model.to(device)
    print(f"tlens_model now on: {next(tlens_model.parameters()).device}")
    
    cache_positions: list[str] | None = None

    # Set device context again before SAE creation
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    model = SAETransformer(
        tlens_model=tlens_model,
        sae_config=config.saes
    )
    
    # Force move entire model to exact device
    print(f"Moving SAETransformer to {device}")
    model = model.to(device)
    
    # Final verification and force correction if needed
    print(f"Final verification - tlens_model device: {next(model.tlens_model.parameters()).device}")
    if next(model.tlens_model.parameters()).device != device:
        print(f"Force correcting tlens_model device")
        model.tlens_model = model.tlens_model.to(device)
    
    for name, sae_module in model.saes.named_modules():
        if hasattr(sae_module, 'weight') or len(list(sae_module.parameters())) > 0:
            param_device = next(sae_module.parameters()).device
            if param_device != device:
                print(f"Force correcting SAE {name} from {param_device} to {device}")
                sae_module.to(device)

    all_param_names = [name for name, _ in model.saes.named_parameters()]
    if config.saes.pretrained_sae_paths is not None:
        trainable_param_names = load_pretrained_saes(
            saes=model.saes,
            pretrained_sae_paths=config.saes.pretrained_sae_paths,
            all_param_names=all_param_names,
            retrain_saes=config.saes.retrain_saes,
        )
    else:
        trainable_param_names = all_param_names

    assert len(trainable_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable parameters: {trainable_param_names}")
    
    # Final device check before training
    print(f"Pre-training device check:")
    print(f"  Target device: {device}")
    print(f"  tlens_model device: {next(model.tlens_model.parameters()).device}")
    
    # Force move any components that are still on wrong device
    if next(model.tlens_model.parameters()).device != device:
        print(f"  Force-moving tlens_model to {device}")
        model.tlens_model = model.tlens_model.to(device)
    
    for name, sae_module in model.saes.named_modules():
        if hasattr(sae_module, 'parameters') and len(list(sae_module.parameters())) > 0:
            param_device = next(sae_module.parameters()).device
            if param_device != device:
                print(f"  Force-moving SAE {name} to {device}")
                sae_module.to(device)
    
    print(f"Device check complete. Starting training...")
    
    # Run training within device context to ensure consistency
    if device.type == 'cuda':
        with torch.cuda.device(device):
            train(
                config=config,
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                trainable_param_names=trainable_param_names,
                device=device,
                cache_positions=cache_positions,
            )
    else:
        train(
            config=config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            trainable_param_names=trainable_param_names,
            device=device,
            cache_positions=cache_positions,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SAEs on a TransformerLens model")
    parser.add_argument("--config_path", type=str, help="Path to the configuration YAML file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    # Set device context IMMEDIATELY if specified, before any model loading
    if args.device:
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            print(f"Set CUDA device context to: {device}")
    
    # Load config and apply any overrides
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config first
    config = load_config(config_path, config_model=Config)
    
    print(f"Running training with config: {config_path}")
    print(f"SAE type: {config.saes.sae_type}")
    print(f"Wandb project: {config.wandb_project}")
    if args.device:
        print(f"Using device: {args.device}")
    
    # Run training
    run(config, device_override=args.device)
