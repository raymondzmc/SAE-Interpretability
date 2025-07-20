import yaml
from pathlib import Path
from typing import Any

import torch
from torch import nn
from utils.logging import logger
from utils.types import BaseModelType
from utils.constants import CONFIG_FILE


def save_module(
    config_dict: dict[str, Any],
    save_dir: Path,
    module: nn.Module,
    model_filename: str,
    config_filename: str = CONFIG_FILE,
) -> None:
    """Save the pytorch module and config to the save_dir.

    The config will only be saved if the save_dir doesn't exist (i.e. the first time the module is
    saved assuming the save_dir is unique to the module).

    Args:
        config_dict: Dictionary representation of the config to save.
        save_dir: Directory to save the module.
        module: The module to save.
        model_filename: The filename to save the model to.
        config_filename: The filename to save the config to.
    """
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        with open(save_dir / config_filename, "w") as f:
            yaml.dump(config_dict, f)
        logger.info("Saved config to %s", save_dir / config_filename)

    torch.save(module.state_dict(), save_dir / model_filename)
    logger.info("Saved model to %s", save_dir / model_filename)


def load_config(config_path_or_obj: Path | str | BaseModelType, config_model: type[BaseModelType]) -> BaseModelType:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)
