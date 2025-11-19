"""LLM Generators Module

Provides OpenAI and Claude generators, retry functionality, and schema definitions.
"""

from .openai_custom_generator import OpenAIVisionGenerator, RetryOpenAIGenerator
from .retry_util import openai_retry
from .schema import (
    RetryConfig,
    GPTConfig,
    OpenAIGenerationConfig,
    ClaudeConfig,
    ClaudeGenerationConfig,
    Model,
    ProviderType,
    APIKeySettings,
    get_api_key_settings,
    get_provider_api_key,
)
from .factory import (
    GeneratorFactory,
    detect_provider_from_model,
    debug_model_detection,
)
from .pipeline_factory import (
    create_pipeline,
    create_vision_pipeline,
    PipelineCreationError,
)

__all__ = [
    # OpenAI
    "OpenAIVisionGenerator",
    "RetryOpenAIGenerator",
    "openai_retry",
    # Schema
    "RetryConfig",
    "GPTConfig",
    "OpenAIGenerationConfig",
    "ClaudeConfig",
    "ClaudeGenerationConfig",
    "Model",
    "ProviderType",
    "APIKeySettings",
    "get_api_key_settings",
    "get_provider_api_key",
    # Factory
    "GeneratorFactory",
    "detect_provider_from_model",
    "debug_model_detection",
    # Pipeline Factory
    "create_pipeline",
    "create_vision_pipeline",
    "PipelineCreationError",
]

# Claude generators - optional import to handle missing anthropic dependency
try:
    from .claude_custom_generator import (  # noqa: F401
        RetryClaudeGenerator,
        ClaudeVisionGenerator,
    )

    __all__.extend(
        [
            "RetryClaudeGenerator",
            "ClaudeVisionGenerator",
        ]
    )
except ImportError:
    # Claude generators not available due to missing anthropic dependency
    pass
