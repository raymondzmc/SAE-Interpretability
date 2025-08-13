import openai


class OpenAIClient:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.models = {
            "gpt-4o-mini": "gpt-4o-mini-07-18",
            "gpt-4o": "gpt-4o-2024-11-20",
        }

    def generate(self, model: str, messages: list[dict[str, str]], **kwargs):
        try:
            model = self.models[model]
        except KeyError:
            raise ValueError(f"Invalid model: {model}")
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)
