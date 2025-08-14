from math import ceil
from typing import List

import torch
from transformers import PreTrainedTokenizer

from ...clients import Client
from ...explainers.features import FeatureRecord
from ...explainers.explainer import ExplainerResult
from ..scorer import Scorer
from .classifier import Classifier
from .prompts.fuzz_prompt import prompt
from .sample import Sample, examples_to_samples


class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        batch_size: int = 1,
        threshold: float = 0.3,
        log_prob: bool = False,
        **generation_kwargs,
    ):
        super().__init__(
            client=client,
            tokenizer=tokenizer,
            verbose=verbose,
            batch_size=batch_size,
            log_prob=log_prob,
            **generation_kwargs,
        )

        self.threshold = threshold
        self.prompt = prompt

    def average_n_activations(self, examples) -> float:
        """Calculate average number of non-zero activations across examples."""
        if not examples:
            return 0
        
        total_nonzero = 0
        for example in examples:
            # Handle both tensor and list activations
            if isinstance(example.activations, torch.Tensor):
                nonzero_count = len(torch.nonzero(example.activations))
            else:
                # For lists, count non-zero values
                nonzero_count = sum(1 for a in example.activations if a != 0)
            total_nonzero += nonzero_count
        
        avg = total_nonzero / len(examples)
        return ceil(avg)

    def _prepare(self, result: ExplainerResult) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        # Extract the FeatureRecord from ExplainerResult
        record = result.record
        
        defaults = {
            "highlighted": True,
            "tokenizer": self.tokenizer,
        }
        
        # Calculate average number of activations from positive examples
        n_incorrect = 0
        if record.positive_examples:
            n_incorrect = self.average_n_activations(record.positive_examples)
        
        # Negative examples
        samples = examples_to_samples(
            record.negative_examples,  # Updated from record.extra_examples
            distance=-1,
            ground_truth=False,
            n_incorrect=n_incorrect,
            **defaults,
        )

        # Positive examples
        if record.positive_examples:
            samples.extend(
                examples_to_samples(
                    record.positive_examples,
                    distance=1,
                    ground_truth=True,
                    n_incorrect=0,
                    **defaults,
                )
            )
        else:
            raise ValueError("No positive examples found for fuzzing scorer")

        return samples
