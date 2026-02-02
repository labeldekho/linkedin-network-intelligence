"""
LLM Provider Abstraction Layer

Provides a unified interface for multiple LLM backends:
- Anthropic (Claude)
- OpenAI (GPT)
- Ollama (Local models)
- vLLM (Production local deployment)
"""

from src.llm.base import LLMProvider, LLMResponse, get_provider

__all__ = ["LLMProvider", "LLMResponse", "get_provider"]
