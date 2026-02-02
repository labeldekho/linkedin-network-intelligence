"""
Anthropic (Claude) LLM Provider Implementation
"""

import json
import logging
import os
from typing import Optional

import anthropic

from src.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        **kwargs
    ):
        """Initialize Anthropic provider.

        Args:
            model: Model name (default: claude-sonnet-4-20250514)
            api_key: API key (default: from environment)
            api_key_env: Environment variable name for API key
            **kwargs: Additional configuration
        """
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)

        # Get API key from argument or environment
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"Anthropic API key not found. Set {api_key_env} environment variable "
                "or pass api_key argument."
            )

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate completion using Claude API."""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract text content
            content = ""
            for block in message.content:
                if hasattr(block, "text"):
                    content += block.text

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                usage={
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def complete_structured(
        self,
        prompt: str,
        schema: dict,
        system: Optional[str] = None,
        temperature: float = 0.3
    ) -> dict:
        """Generate structured response matching schema."""
        structured_prompt = self._build_structured_prompt(prompt, schema)

        response = await self.complete(
            prompt=structured_prompt,
            system=system,
            temperature=temperature,
            max_tokens=2048,
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from response content
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response.content}")
            raise ValueError(f"Invalid JSON in response: {e}")
