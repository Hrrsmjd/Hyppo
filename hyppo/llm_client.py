from hyppo.config import get_api_key

BASE_URLS = {
    "anthropic": "https://api.anthropic.com/v1/",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str | None = None):
        from openai import OpenAI

        self.provider = provider
        self.model = model
        if api_key is None:
            api_key = get_api_key(provider)
        if not api_key:
            raise ValueError(f"No API key found for provider '{provider}'")
        base_url = BASE_URLS.get(provider)
        if not base_url:
            raise ValueError(f"Unknown provider '{provider}'. Choose: {list(BASE_URLS.keys())}")
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)
