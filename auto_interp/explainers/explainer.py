import re
from abc import ABC, abstractmethod
from typing import NamedTuple
from transformers import PreTrainedTokenizer
from auto_interp.clients import Client, Response
from auto_interp.explainers.features import FeatureRecord, Example
from auto_interp.explainers.prompts import build_prompt
from utils.logging import logger


class ExplainerResult(NamedTuple):
    record: FeatureRecord
    """Feature record passed through to scorer."""

    explanation: str
    """Generated explanation for feature."""


class Explainer(ABC):
    @abstractmethod
    async def __call__(self, record: FeatureRecord) -> ExplainerResult:
        pass


class DefaultExplainer(Explainer):
    name = "default"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer | None = None,
        verbose: bool = False,
        activations: bool = False,
        cot: bool = False,
        threshold: float = 0.6,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.activations = activations
        self.cot = cot
        self.threshold = threshold
        self.generation_kwargs = generation_kwargs


    def __call__(self, record: FeatureRecord) -> ExplainerResult:
        messages = self._build_prompt(record.explanation_examples)
        
        response: Response = self.client.generate(messages, **self.generation_kwargs)
        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                return (
                    messages[-1]["content"],
                    response,
                    ExplainerResult(record=record, explanation=explanation),
                )

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed: {e}")
            return ExplainerResult(record=record, explanation="Explanation could not be parsed.")

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {e}")
            raise
        
    def _highlight(self, index: int, example: Example) -> str:
        result = [f"Example {index}: "]

        threshold = example.max_activation * self.threshold
        
        # Check if tokens are already strings or need decoding
        if not example.tokens or len(example.tokens) == 0:
            # Handle empty tokens case
            example.str_toks = []
            return "".join(result) + "[No tokens]"
        elif isinstance(example.tokens[0], str):
            # Tokens are already strings
            example.str_toks = example.tokens
        elif self.tokenizer is not None:
            # Tokens are IDs, need decoding
            str_toks = self.tokenizer.batch_decode(example.tokens)
            example.str_toks = str_toks
        else:
            assert all(isinstance(token, str) for token in example.tokens), "Tokenizer is required to highlight examples"
            example.str_toks = example.tokens

        activations = example.activations
        str_toks = example.str_toks  # Use the str_toks we just set

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result.append("<<")

                while i < len(str_toks) and check(i):
                    result.append(str_toks[i])
                    i += 1
                result.append(">>")
            else:
                result.append(str_toks[i])
                i += 1

        return "".join(result)

    def _join_activations(self, example: Example) -> str:
        activations = []

        threshold = 0.6
        for i, normalized in enumerate(example.normalized_activations):
            if example.normalized_activations[i] > threshold:
                activations.append((example.str_toks[i], int(normalized)))

        acts = ", ".join(f'("{item[0]}" : {item[1]})' for item in activations)

        return "Activations: " + acts

    def _build_prompt(self, examples: list[Example]) -> list[dict[str, str]]:
        highlighted_examples: list[str] = []

        for i, example in enumerate(examples):
            highlighted_examples.append(self._highlight(i + 1, example))

            if self.activations:
                highlighted_examples.append(self._join_activations(example))

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            activations=self.activations,
            cot=self.cot,
        )
