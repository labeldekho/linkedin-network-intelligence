"""
Configuration Management

Loads configuration from YAML files with environment variable resolution.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "anthropic"
    models: dict[str, str] = Field(default_factory=lambda: {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "ollama": "llama3.1:8b",
        "vllm": "meta-llama/Llama-3.1-8B-Instruct",
    })
    api_key_env: dict[str, str] = Field(default_factory=lambda: {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    })
    local: dict[str, str] = Field(default_factory=lambda: {
        "ollama_base_url": "http://localhost:11434",
        "vllm_base_url": "http://localhost:8000",
    })
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1

    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API key from environment variable."""
        provider = provider or self.provider
        env_var = self.api_key_env.get(provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    def get_model(self, provider: Optional[str] = None) -> str:
        """Get model name for provider."""
        provider = provider or self.provider
        return self.models.get(provider, self.models["anthropic"])


class RelationshipConfig(BaseModel):
    """Relationship strength calculation configuration."""
    decay_half_life_days: int = 180
    minimum_strength: float = 0.05
    depth_multiplier_max: float = 2.0
    institutional_multiplier: float = 1.5
    interaction_weights: dict[str, float] = Field(default_factory=lambda: {
        "message_sent": 1.0,
        "message_received": 1.0,
        "recommendation_written": 5.0,
        "recommendation_received": 5.0,
        "endorsement_given": 0.5,
        "endorsement_received": 0.5,
    })


class ReciprocityConfig(BaseModel):
    """Reciprocity ledger configuration."""
    scores: dict[str, int] = Field(default_factory=lambda: {
        "recommendation_written": 10,
        "recommendation_received": -10,
        "endorsement_given": 2,
        "endorsement_received": -2,
    })
    thresholds: dict[str, int] = Field(default_factory=lambda: {
        "strong_credit": 15,
        "credit": 5,
        "balanced_min": -5,
        "balanced_max": 5,
        "debit": -15,
        "strong_debit": -999,
    })


class MessagesConfig(BaseModel):
    """Message analysis configuration."""
    min_thread_length: int = 2
    thread_break_days: int = 7
    batch_size: int = 5
    min_message_length: int = 20


class ResurrectionConfig(BaseModel):
    """Resurrection detection configuration."""
    min_dormant_days: int = 90
    max_dormant_days: int = 1095
    min_strength_threshold: float = 0.2


class WarmPathsConfig(BaseModel):
    """Warm path discovery configuration."""
    max_candidates: int = 10
    min_bridge_strength: float = 0.3
    weights: dict[str, float] = Field(default_factory=lambda: {
        "relationship_strength": 0.4,
        "company_relevance": 0.4,
        "recency": 0.2,
    })


class CacheConfig(BaseModel):
    """LLM response caching configuration."""
    enabled: bool = True
    path: str = ".cache/llm_responses.db"
    ttl_days: int = 7
    max_size_mb: int = 500


class OutputConfig(BaseModel):
    """Output generation configuration."""
    directory: str = "./outputs"
    formats: list[str] = Field(default_factory=lambda: ["csv", "markdown", "json"])
    timestamp_filenames: bool = True
    markdown: dict[str, Any] = Field(default_factory=lambda: {
        "include_methodology": True,
        "max_items_per_section": 20,
    })


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None
    timestamps: bool = True
    redact_fields: list[str] = Field(default_factory=lambda: [
        "message_content", "email", "phone"
    ])


class ProcessingConfig(BaseModel):
    """Data processing configuration."""
    max_connections: int = 0
    max_connection_age_days: int = 0
    parallel_calls: int = 1
    progress_every_n: int = 50


class PromptsConfig(BaseModel):
    """Prompt version configuration."""
    message_depth: str = "v1"
    resurrection: str = "v1"
    warm_path: str = "v1"
    archetype: str = "v1"


class Config(BaseModel):
    """Root configuration object."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    relationship: RelationshipConfig = Field(default_factory=RelationshipConfig)
    reciprocity: ReciprocityConfig = Field(default_factory=ReciprocityConfig)
    messages: MessagesConfig = Field(default_factory=MessagesConfig)
    resurrection: ResurrectionConfig = Field(default_factory=ResurrectionConfig)
    warm_paths: WarmPathsConfig = Field(default_factory=WarmPathsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve environment variables in config values.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
    """
    if isinstance(data, str):
        if data.startswith("${") and data.endswith("}"):
            var_expr = data[2:-1]
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_expr, data)
        return data
    elif isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def load_config(
    config_path: Optional[Path] = None,
    local_config_path: Optional[Path] = None,
) -> Config:
    """Load configuration from YAML files.

    Args:
        config_path: Path to main config file (default: config.yaml)
        local_config_path: Path to local overrides (default: config.local.yaml)

    Returns:
        Merged and validated Config object
    """
    # Find project root
    project_root = Path(__file__).parent.parent.parent

    # Default paths
    if config_path is None:
        config_path = project_root / "config.yaml"
    if local_config_path is None:
        local_config_path = project_root / "config.local.yaml"

    config_data: dict[str, Any] = {}

    # Load main config
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    # Merge local overrides
    if local_config_path.exists():
        with open(local_config_path) as f:
            local_data = yaml.safe_load(f) or {}
            config_data = _deep_merge(config_data, local_data)

    # Resolve environment variables
    config_data = _resolve_env_vars(config_data)

    return Config(**config_data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
