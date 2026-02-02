"""
Local LLM Provider Implementations (Ollama and vLLM)
"""

import json
import logging
from typing import Optional

import httpx

from src.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    DEFAULT_MODEL = "llama3.1:8b"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """Initialize Ollama provider.

        Args:
            model: Model name (default: llama3.1:8b)
            base_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate completion using Ollama API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "You are a helpful assistant.",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()

                return LLMResponse(
                    content=data.get("response", ""),
                    model=self.model,
                    provider=self.provider_name,
                    usage={
                        "total_duration": data.get("total_duration"),
                        "eval_count": data.get("eval_count"),
                    }
                )

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")

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


class VLLMProvider(LLMProvider):
    """vLLM server provider for production local deployments."""

    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """Initialize vLLM provider.

        Args:
            model: Model name (default: meta-llama/Llama-3.1-8B-Instruct)
            base_url: vLLM server URL (default: http://localhost:8000)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "vllm"

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate completion using vLLM OpenAI-compatible API."""
        url = f"{self.base_url}/v1/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider=self.provider_name,
                    usage=data.get("usage", {}),
                )

        except httpx.HTTPError as e:
            logger.error(f"vLLM HTTP error: {e}")
            raise ConnectionError(f"Failed to connect to vLLM at {self.base_url}: {e}")

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
