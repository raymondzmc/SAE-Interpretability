import asyncio
import json
import random
import re
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, NamedTuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
from ...clients import Client
from ...explainers.features import Example, FeatureRecord
from ...explainers.explainer import ExplainerResult
from ..scorer import Scorer, ScorerResult


@dataclass
class EmbeddingOutput:
    text: str
    """The text that was used to evaluate the similarity"""

    distance: float | int
    """Quantile or neighbor distance"""

    similarity: float = 0.0
    """What is the similarity of the example to the explanation"""


class Sample(NamedTuple):
    text: str
    activations: list[float]
    data: EmbeddingOutput


class EmbeddingScorer(Scorer):
    name = "embedding"

    def __init__(
        self,
        model,
        tokenizer,
        verbose: bool,
        batch_size: int,
        **generation_kwargs,
    ):
        self.model = model
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        
    async def __call__(
        self,
        result: ExplainerResult,
    ) -> ScorerResult:
        samples = self._prepare(result)

        random.shuffle(samples)
        results = await self._query(
            result.explanation,
            samples,
        )
        
        return ScorerResult(record=result.record, score=results)

    def _prepare(self, result: ExplainerResult) -> list[Sample]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        # Extract the FeatureRecord from ExplainerResult
        record = result.record

        defaults = {
            "tokenizer": self.tokenizer,
        }
        
        # Negative examples
        samples = examples_to_samples(
            record.negative_examples,  # Updated from record.extra_examples
            distance=-1,
            **defaults,
        )

        # Positive examples
        if record.positive_examples:
            samples.extend(
                examples_to_samples(
                    record.positive_examples,
                    distance=1,
                    **defaults,
                )
            )

        return samples



    async def _query(self, explanation: str, samples: list[Sample]) -> list[EmbeddingOutput]:
        # Run the synchronous embedding computation in an executor to not block
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_embeddings, explanation, samples)
    
    def _compute_embeddings(self, explanation: str, samples: list[Sample]) -> list[EmbeddingOutput]:
        explanation_prompt = "Instruct: Retrieve sentences that could be related to the explanation.\nQuery:" + explanation 
        query_embedding = self.model.encode(explanation_prompt)
        
        samples_text = [sample.text for sample in samples]
    
        # Compute embeddings for all samples
        sample_embeddings = self.model.encode(samples_text)
        similarity = self.model.similarity(query_embedding, sample_embeddings)[0]
        
        results = []
        for i in range(len(samples)):
            # Create new EmbeddingOutput with the similarity score
            output = EmbeddingOutput(
                text=samples[i].data.text,
                distance=samples[i].data.distance,
                similarity=similarity[i].item()
            )
            results.append(output)
        return results
        


def examples_to_samples(
    examples: list[Example],
    tokenizer: PreTrainedTokenizer,
    **sample_kwargs,
) -> list[Sample]:
    samples = []    
    for example in examples:
        # Handle tokens that are already strings
        if hasattr(tokenizer, 'is_mock') and tokenizer.is_mock:
            # Tokens are already strings, just join them
            text = "".join(example.tokens)
        else:
            # Normal case: decode tokens
            text = "".join(tokenizer.batch_decode(example.tokens))
        
        # Handle both tensor and list activations
        if isinstance(example.activations, torch.Tensor):
            activations = example.activations.tolist()
        else:
            activations = example.activations
        samples.append(
            Sample(
                text=text,
                activations=activations,
                data=EmbeddingOutput(
                    text=text,
                    **sample_kwargs
                ),
            )
        )

    return samples
    
