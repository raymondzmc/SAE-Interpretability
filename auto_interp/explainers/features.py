from dataclasses import dataclass
import blobfile as bf
import orjson
import torch
from typing import Optional


@dataclass
class Example:
    """
    A single example of feature data.

    Attributes:
        tokens (TensorType["seq"]): Tokenized input sequence.
        activations (TensorType["seq"]): Activation values for the input sequence.
        normalized_activations (TensorType["seq"]): Normalized activation values.
    """
    tokens: list[str]
    activations: torch.Tensor
    normalized_activations: Optional[torch.Tensor] = None
    str_toks: Optional[list[str]] = None
    
    @property
    def max_activation(self):
        """
        Get the maximum activation value.

        Returns:
            float: The maximum activation value.
        """
        return self.activations.max()


@dataclass
class Feature:
    """
    A feature extracted from a model's activations.

    Attributes:
        module_name (str): The module name associated with the feature.
        feature_index (int): The index of the feature within the module.
    """
    module_name: str
    feature_index: int

    def __repr__(self) -> str:
        """
        Return a string representation of the feature.

        Returns:
            str: A string representation of the feature.
        """
        return f"{self.module_name}_feature{self.feature_index}"


class FeatureRecord:
    """
    A record of feature data.

    Attributes:
        feature (Feature): The feature associated with the record.
    """

    def __init__(
        self,
        feature: Feature,
    ):
        """
        Initialize the feature record.

        Args:
            feature (Feature): The feature associated with the record.
        """
        self.feature: Feature = feature
        self.examples: list[Example] = []
        self.train: list[Example] = []
        self.test: list[Example] = []

    @property
    def max_activation(self):
        """
        Get the maximum activation value for the feature.

        Returns:
            float: The maximum activation value.
        """
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples=False):
        """
        Save the feature record to a file.

        Args:
            directory (str): The directory to save the file in.
            save_examples (bool): Whether to save the examples. Defaults to False.
        """
        path = f"{directory}/{self.feature}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")
            serializable.pop("train")
            serializable.pop("test")

        serializable.pop("feature")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))
