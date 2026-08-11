"""
Step 6: a single class hides which LLM provider is behind the app, so
the rest of the RAG pipeline never needs to change if you switch models.
"""
from app.config import settings


class LLM:
    def __init__(self):
        self.provider = settings.llm_provider.lower()

        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.model = settings.anthropic_model
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_model
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        # openai
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


_llm_instance = None


def get_llm() -> LLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLM()
    return _llm_instance
