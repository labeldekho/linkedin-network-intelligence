"""
LLM Provider Abstraction Layer

This module provides a unified interface for multiple LLM backends:
- Anthropic (Claude)
- OpenAI (GPT)
- Ollama (Local Llama, Qwen, etc.)
- vLLM (Production local deployment)

Usage:
    from src.llm import get_provider

    provider = get_provider("anthropic", model="claude-sonnet-4-20250514")
    response = await provider.complete("Your prompt here")
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Optional[dict] = None  # Token counts if available


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement these methods to ensure
    consistent behavior across backends.
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'openai')."""
        pass

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt to complete
            system: Optional system prompt for context
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        schema: dict,
        system: Optional[str] = None,
        temperature: float = 0.3
    ) -> dict:
        """Generate a structured response matching the provided schema.

        Args:
            prompt: The user prompt
            schema: JSON schema for expected response format
            system: Optional system prompt
            temperature: Lower default for structured output

        Returns:
            Parsed dictionary matching the schema
        """
        pass

    def _build_structured_prompt(self, prompt: str, schema: dict) -> str:
        """Helper to build a prompt that requests structured output.

        Works across all providers by embedding format instructions.
        """
        schema_str = str(schema)
        return f"""{prompt}

Respond with a JSON object matching this schema:
{schema_str}

Respond with ONLY the JSON object, no additional text or markdown formatting."""


def get_provider(
    provider_name: str,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """Factory function to get the appropriate LLM provider.

    Args:
        provider_name: One of 'anthropic', 'openai', 'ollama', 'vllm'
        model: Model name (uses config default if not specified)
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider_name is not recognized
    """
    providers = {
        'anthropic': 'src.llm.anthropic.AnthropicProvider',
        'openai': 'src.llm.openai.OpenAIProvider',
        'ollama': 'src.llm.local.OllamaProvider',
        'vllm': 'src.llm.local.VLLMProvider',
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(providers.keys())}"
        )

    # Dynamic import to avoid loading unused providers
    module_path, class_name = providers[provider_name].rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)

    return provider_class(model=model, **kwargs)
