"""
OpenAI LLM Provider Implementation
"""

import json
import logging
import os
from typing import Optional

import openai

from src.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT API provider."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (default: gpt-4o)
            api_key: API key (default: from environment)
            api_key_env: Environment variable name for API key
            **kwargs: Additional configuration
        """
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)

        # Get API key from argument or environment
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"OpenAI API key not found. Set {api_key_env} environment variable "
                "or pass api_key argument."
            )

        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                }
            )

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def complete_structured(
        self,
        prompt: str,
        schema: dict,
        system: Optional[str] = None,
        temperature: float = 0.3
    ) -> dict:
        """Generate structured response matching schema.

        Uses OpenAI's JSON mode when available.
        """
        messages = []

        json_system = (system or "") + "\nRespond with valid JSON only."
        messages.append({"role": "system", "content": json_system})

        structured_prompt = self._build_structured_prompt(prompt, schema)
        messages.append({"role": "user", "content": structured_prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON in response: {e}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
