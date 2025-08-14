from typing import List

from transformers import PreTrainedTokenizer

from auto_interp.clients import Client
from auto_interp.explainers.features import FeatureRecord
from auto_interp.explainers.explainer import ExplainerResult
from auto_interp.scorers.classifier.classifier import Classifier
from auto_interp.scorers.classifier.prompts.detection_prompt import prompt
from auto_interp.scorers.classifier.sample import Sample, examples_to_samples


class DetectionScorer(Classifier):
    name = "detection"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        batch_size: int = 10,
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

        self.prompt = prompt

    def _prepare(self, result: ExplainerResult) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        # Extract the FeatureRecord from ExplainerResult
        record = result.record

        # Negative examples (contrastive)
        samples = examples_to_samples(
            record.negative_examples,  # Updated from record.random_examples
            distance=-1,
            ground_truth=False,
            tokenizer=self.tokenizer,
        )

        # Positive examples 
        # Note: positive_examples is a simple list, not list of lists
        if record.positive_examples:
            samples.extend(
                examples_to_samples(
                    record.positive_examples,
                    distance=1,
                    ground_truth=True,
                    tokenizer=self.tokenizer,
                )
            )

        return samples
