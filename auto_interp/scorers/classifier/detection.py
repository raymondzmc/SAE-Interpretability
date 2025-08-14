from typing import List

from transformers import PreTrainedTokenizer

from auto_interp.clients import Client
from auto_interp.explainers.features import FeatureRecord
from auto_interp.scorers.classifier import Classifier
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

    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = examples_to_samples(
            record.random_examples,
            distance=-1,
            ground_truth=False,
            tokenizer=self.tokenizer,
        )

        for i, examples in enumerate(record.test):
            samples.extend(
                examples_to_samples(
                    examples,
                    distance=i + 1,
                    ground_truth=True,
                    tokenizer=self.tokenizer,
                )
            )

        return samples
