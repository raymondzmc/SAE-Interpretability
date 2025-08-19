import openai
import together
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Type, List
from pydantic import BaseModel, ValidationError
from together.types.chat_completions import ChatCompletionChoicesData, LogprobsPart

T = TypeVar('T', bound=BaseModel)


@dataclass
class TopLogProb:
    """Represents a single token alternative with its log probability."""
    token: str
    logprob: float


@dataclass  
class LogProb:
    """Represents a token with its alternatives."""
    token: str
    logprob: float
    top_logprobs: List[TopLogProb]


@dataclass
class Response:
    text: str
    logprobs: List[LogProb]
    cost: float
    input_tokens: int
    output_tokens: int
    structured_response: Optional[BaseModel] = None


class Client(ABC):
    api_key: str
    model: str
    max_logprobs: int

    @abstractmethod
    async def generate(
        self, 
        messages: list[dict[str, str]],
        response_model: Optional[Type[T]] = None,
        **kwargs
    ) -> Response:
        raise NotImplementedError

# OpenAI model mappings and pricing (per 1M tokens)
OPENAI_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-11-20",
}

# OpenAI pricing per 1M tokens (input, output)
OPENAI_PRICING = {
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
}


class OpenAIClient(Client):
    max_logprobs: int = 20

    def __init__(self, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        if model not in OPENAI_MODELS:
            raise ValueError(f"Invalid model: {model}. Available models: {list(OPENAI_MODELS.keys())}")
        self.model_name = OPENAI_MODELS[model]

    async def generate(self, messages: list[dict[str, str]], response_model: Optional[Type[T]] = None, **kwargs) -> Response:
        """Async generation with optional structured output."""
        structured_response = None
        logprobs = []

        if "logprobs" not in kwargs:
            kwargs["logprobs"] = True
        if "top_logprobs" not in kwargs and kwargs.get("logprobs"):
            kwargs["top_logprobs"] = self.max_logprobs

        response = None
        if response_model:
            response = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=response_model,
                **kwargs
            )
            structured_response = response.choices[0].message.parsed
        else:
            response = await self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages, 
                **kwargs
            )

        text = response.choices[0].message.content or ""
        
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            for token_data in response.choices[0].logprobs.content:
                top_logprobs = []
                if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                    for alt in token_data.top_logprobs:
                        top_logprobs.append(TopLogProb(
                            token=alt.token,
                            logprob=alt.logprob
                        ))
                logprobs.append(LogProb(
                    token=token_data.token,
                    logprob=token_data.logprob,
                    top_logprobs=top_logprobs
                ))

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        pricing = OPENAI_PRICING.get(self.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        
        return Response(
            text=text,
            logprobs=logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            structured_response=structured_response,
        )


# TogetherAI model mappings
TOGETHERAI_MODELS = {
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
}

# TogetherAI pricing per 1M tokens (input, output) - approximate values
TOGETHERAI_PRICING = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {"input": 0.18, "output": 0.18},
}


class TogetherAIClient(Client):
    max_logprobs: int = 5

    def __init__(self, api_key: str, model: str):
        self.client = together.AsyncTogether(api_key=api_key)
        if model not in TOGETHERAI_MODELS:
            raise ValueError(f"Invalid model: {model}. Available models: {list(TOGETHERAI_MODELS.keys())}")
        self.model_name = TOGETHERAI_MODELS[model]
        # Llama 3.2 doesn't support alternatives, so set max to 1
        if "11B" in self.model_name:
            self.max_logprobs = 1

    async def generate(self, messages: list[dict[str, str]], response_model: Optional[Type[T]] = None, **kwargs) -> Response:
        """Async generation with optional structured output."""
        structured_response = None
        logprobs = []

        # TogetherAI uses 'logprobs' as an integer, not 'top_logprobs'
        # Cap at model's maximum
        kwargs["logprobs"] = min(self.max_logprobs, kwargs.get("logprobs", self.max_logprobs))
        
        if response_model is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "schema": response_model.model_json_schema(),
            }

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if len(response.choices) == 0:
            raise Exception(f"No response choices from model: {response}")

        response_choice: ChatCompletionChoicesData = response.choices[0]
        text = response_choice.message.content or ""
        if text == "":
            raise Exception(f"No text in response choice: {response_choice}")

        # Parse structured response
        if response_model is not None:
            try:
                structured_response = response_model.model_validate_json(text)
            except ValidationError as e:
                raise Exception(f"Failed to parse structured response {e.message}: {text}")

        # Extract logprobs if available
        if response_choice.logprobs:
            logprobs_data: LogprobsPart = response_choice.logprobs
            tokens = logprobs_data.tokens if hasattr(logprobs_data, 'tokens') and logprobs_data.tokens else []
            token_logprobs = logprobs_data.token_logprobs if hasattr(logprobs_data, 'token_logprobs') and logprobs_data.token_logprobs else []
            top_logprobs_data = logprobs_data.top_logprobs if hasattr(logprobs_data, 'top_logprobs') and logprobs_data.top_logprobs else []
            
            # Build LogProb entries
            for i, token in enumerate(tokens):
                top_logprobs = []
                if top_logprobs_data and i < len(top_logprobs_data) and top_logprobs_data[i]:
                    for alt_token, alt_logprob in top_logprobs_data[i].items():
                        top_logprobs.append(TopLogProb(
                            token=alt_token,
                            logprob=alt_logprob
                        ))
                elif token_logprobs and i < len(token_logprobs):
                    # If no alternatives, just include the selected token
                    top_logprobs.append(TopLogProb(
                        token=token,
                        logprob=token_logprobs[i] if token_logprobs[i] is not None else 0.0
                    ))
                
                logprobs.append(LogProb(
                    token=token,
                    logprob=token_logprobs[i] if token_logprobs and i < len(token_logprobs) and token_logprobs[i] is not None else 0.0,
                    top_logprobs=top_logprobs
                ))
        
        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost (pricing is per 1M tokens)
        pricing = TOGETHERAI_PRICING.get(self.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        
        return Response(
            text=text,
            logprobs=logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            structured_response=structured_response,
        )
