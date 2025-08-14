import openai
import together
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class Response:
    text: str
    tokens: list[str]  # Output tokens
    logprobs: list[float]  # Log probabilities for output tokens
    prompt_logprobs: list[float]  # Log probabilities for prompt tokens
    cost: float
    input_tokens: int  # Number of input tokens
    output_tokens: int  # Number of output tokens


class Client(ABC):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def generate(
        self, 
        messages: list[dict[str, str]],
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


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        self.client = openai.OpenAI(api_key=api_key)
        if model not in OPENAI_MODELS:
            raise ValueError(f"Invalid model: {model}. Available models: {list(OPENAI_MODELS.keys())}")
        self.model_name = OPENAI_MODELS[model]

    def generate(self, messages: list[dict[str, str]], **kwargs):
        # Request logprobs if not explicitly disabled
        if "logprobs" not in kwargs:
            kwargs["logprobs"] = True
        if "top_logprobs" not in kwargs and kwargs.get("logprobs"):
            kwargs["top_logprobs"] = 1
            
        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages, 
            **kwargs
        )
        
        # Extract text
        text = response.choices[0].message.content or ""
        
        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost (pricing is per 1M tokens)
        pricing = OPENAI_PRICING.get(self.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        
        # For OpenAI, we return empty lists for logprobs as requested
        tokens = []
        logprobs = []
        prompt_logprobs = []
        
        return Response(
            text=text,
            tokens=tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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


class TogetherAIClient:
    def __init__(self, api_key: str, model: str):
        self.client = together.Client(api_key=api_key)
        if model not in TOGETHERAI_MODELS:
            raise ValueError(f"Invalid model: {model}. Available models: {list(TOGETHERAI_MODELS.keys())}")
        self.model_name = TOGETHERAI_MODELS[model]

    def generate(self, messages: list[dict[str, str]], **kwargs):
        # Request logprobs if not explicitly disabled
        if "logprobs" not in kwargs:
            kwargs["logprobs"] = 1  # TogetherAI uses integer for number of logprobs
            
        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages, 
            **kwargs
        )
        
        # Extract text
        text = response.choices[0].message.content or ""
        
        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost (pricing is per 1M tokens)
        pricing = TOGETHERAI_PRICING.get(self.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        
        # Extract logprobs if available
        tokens = []
        logprobs = []
        prompt_logprobs = []
        
        # TogetherAI may provide logprobs in a different format
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            logprobs_data = response.choices[0].logprobs
            if hasattr(logprobs_data, 'content') and logprobs_data.content:
                for token_data in logprobs_data.content:
                    if hasattr(token_data, 'token'):
                        tokens.append(token_data.token)
                    if hasattr(token_data, 'logprob'):
                        logprobs.append(token_data.logprob)
        
        return Response(
            text=text,
            tokens=tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
