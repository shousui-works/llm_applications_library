"""Generator Factory for automatic selection between OpenAI and Claude."""

import logging
from typing import Any, Protocol, Union
from enum import StrEnum

from .schema import Model, RetryConfig, GPTConfig, ClaudeConfig
from .openai_custom_generator import (
    RetryOpenAIGenerator,
    OpenAIVisionGenerator,
)

logger = logging.getLogger(__name__)

# Check Claude availability
try:
    import anthropic  # noqa: F401

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Claude generators not available - anthropic package not installed")


class ProviderType(StrEnum):
    """LLM Provider types."""

    OPENAI = "openai"
    CLAUDE = "claude"


class TextGenerator(Protocol):
    """Protocol for text generators."""

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate text response."""
        ...


# Type alias for vision generators
VisionGenerator = Any


class ConfiguredGenerator:
    """Universal wrapper for generators with presets."""

    def __init__(self, generator, presets, generator_type):
        self._generator = generator
        self._presets = presets
        self._generator_type = generator_type
        # Copy attributes from base generator
        for attr in dir(generator):
            if not attr.startswith("_") and not callable(getattr(generator, attr)):
                setattr(self, attr, getattr(generator, attr))

    def run(self, *args, **kwargs):
        """Run with preset merging."""
        if self._generator_type == "text":
            # For text: merge generation_kwargs
            if "generation_kwargs" in self._presets:
                merged_kwargs = self._presets["generation_kwargs"].copy()
                if "generation_kwargs" in kwargs:
                    merged_kwargs.update(kwargs["generation_kwargs"])
                kwargs["generation_kwargs"] = merged_kwargs
            return self._generator.run(*args, **kwargs)

        elif self._generator_type == "vision":
            # For vision: use model_config if not provided
            if "model_config" in self._presets and "model_config" not in kwargs:
                kwargs["model_config"] = self._presets["model_config"]
            return self._generator.run(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate to wrapped generator."""
        return getattr(self._generator, name)


def detect_provider_from_model(model: str) -> ProviderType:
    """
    Detect which provider (OpenAI or Claude) a model belongs to.

    Args:
        model: Model name (can be from Model enum or custom string)

    Returns:
        ProviderType: The detected provider

    Raises:
        ValueError: If model provider cannot be determined
    """
    model_lower = model.lower()

    # Check if it's a known enum value
    try:
        model_enum = Model(model)
        model_name = model_enum.value
    except ValueError:
        # Custom model name, use the original
        model_name = model

    # OpenAI model patterns
    openai_patterns = [
        "gpt-",
        "chatgpt",
        "text-davinci",
        "text-curie",
        "text-babbage",
        "text-ada",
        "code-davinci",
        "code-cushman",
        "whisper",
        "dall-e",
        "tts",
    ]

    # Claude model patterns
    claude_patterns = ["claude-", "anthropic"]

    # Check patterns
    for pattern in openai_patterns:
        if pattern in model_lower:
            return ProviderType.OPENAI

    for pattern in claude_patterns:
        if pattern in model_lower:
            return ProviderType.CLAUDE

    # Check against known model values
    openai_models = [
        Model.GPT_4O,
        Model.GPT_4O_MINI,
        Model.GPT_4_1,
        Model.GPT_4_1_MINI,
        Model.GPT_4_1_NANO,
        Model.GPT_4,
        Model.GPT_4_TURBO,
        Model.GPT_3_5_TURBO,
    ]

    claude_models = [
        Model.CLAUDE_3_5_SONNET,
        Model.CLAUDE_3_5_HAIKU,
        Model.CLAUDE_3_OPUS,
        Model.CLAUDE_3_SONNET,
        Model.CLAUDE_3_HAIKU,
    ]

    if model_name in [m.value for m in openai_models]:
        return ProviderType.OPENAI

    if model_name in [m.value for m in claude_models]:
        return ProviderType.CLAUDE

    raise ValueError(f"Cannot detect provider for model: {model}")


def get_default_config_for_provider(
    provider: ProviderType, retry_config: RetryConfig | None = None
) -> Union[GPTConfig, ClaudeConfig]:
    """
    Get default configuration for a provider.

    Args:
        provider: The provider type
        retry_config: Optional retry configuration

    Returns:
        Default configuration for the provider
    """
    if provider == ProviderType.OPENAI:
        return GPTConfig(retry_config=retry_config or RetryConfig())
    elif provider == ProviderType.CLAUDE:
        return ClaudeConfig(retry_config=retry_config or RetryConfig())
    else:
        raise ValueError(f"Unknown provider: {provider}")


class GeneratorFactory:
    """
    Factory class for creating appropriate generators based on model names.

    Automatically selects between OpenAI and Claude generators based on the
    provided model name, with support for both text and vision capabilities.
    """

    @staticmethod
    def create_text_generator(
        model: str,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> TextGenerator:
        """
        Create a text generator for the specified model.

        Args:
            model: Model name (from Model enum or custom string)
            api_key: API key for the provider (optional, can use env vars)
            retry_config: Retry configuration (optional)

        Returns:
            Appropriate text generator instance

        Raises:
            ValueError: If model provider cannot be determined or is unsupported
            ImportError: If required packages for the provider are not installed
        """
        provider = detect_provider_from_model(model)

        if provider == ProviderType.OPENAI:
            return RetryOpenAIGenerator(
                model=model,
                api_key=api_key,
                retry_config=retry_config,
            )
        elif provider == ProviderType.CLAUDE:
            if not CLAUDE_AVAILABLE:
                raise ImportError(
                    "Claude generators require the 'anthropic' package. "
                    "Install with: pip install anthropic"
                )
            # Import here to avoid unbound variable issues
            from .claude_custom_generator import RetryClaudeGenerator

            return RetryClaudeGenerator(
                model=model,
                api_key=api_key,
                retry_config=retry_config,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def create_vision_generator(
        model: str,
        api_key: str | None = None,
    ) -> VisionGenerator:
        """
        Create a vision generator for the specified model.

        Args:
            model: Model name (from Model enum or custom string)
            api_key: API key for the provider (optional, can use env vars)

        Returns:
            Appropriate vision generator instance

        Raises:
            ValueError: If model provider cannot be determined or is unsupported
            ImportError: If required packages for the provider are not installed
        """
        provider = detect_provider_from_model(model)

        if provider == ProviderType.OPENAI:
            return OpenAIVisionGenerator(
                model=model,
                api_key=api_key,
            )
        elif provider == ProviderType.CLAUDE:
            if not CLAUDE_AVAILABLE:
                raise ImportError(
                    "Claude generators require the 'anthropic' package. "
                    "Install with: pip install anthropic"
                )
            # Import here to avoid unbound variable issues
            from .claude_custom_generator import ClaudeVisionGenerator

            return ClaudeVisionGenerator(
                model=model,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_provider_for_model(model: str) -> ProviderType:
        """
        Get the provider type for a given model.

        Args:
            model: Model name

        Returns:
            Provider type
        """
        return detect_provider_from_model(model)

    @staticmethod
    def get_default_config(
        model: str, retry_config: RetryConfig | None = None
    ) -> Union[GPTConfig, ClaudeConfig]:
        """
        Get default configuration for a model.

        Args:
            model: Model name
            retry_config: Optional retry configuration

        Returns:
            Default configuration for the model's provider
        """
        provider = detect_provider_from_model(model)
        return get_default_config_for_provider(provider, retry_config)


# Convenience functions for direct usage
def create_generator(
    model: str,
    generator_type: str = "text",
    api_key: str | None = None,
    retry_config: RetryConfig | None = None,
    **preset_kwargs,
) -> Union[TextGenerator, VisionGenerator]:
    """
    Convenience function to create a generator with optional presets.

    Args:
        model: Model name
        generator_type: Type of generator ("text" or "vision")
        api_key: API key (optional)
        retry_config: Retry configuration (optional)
        **preset_kwargs: Preset parameters (generation_kwargs for text, model_config for vision)

    Returns:
        Generator instance with optional preset parameters

    Examples:
        # Basic usage
        gen = create_generator("gpt-4o", "text")

        # Text with preset generation parameters
        gen = create_generator(
            "claude-3-haiku", "text",
            generation_kwargs={"temperature": 0.8, "max_tokens": 200}
        )

        # Vision with preset model config
        config = GPTConfig(model="gpt-4o")
        vision_gen = create_generator("gpt-4o", "vision", model_config=config)
    """

    # Create base generator
    if generator_type == "text":
        base_generator = GeneratorFactory.create_text_generator(
            model, api_key, retry_config
        )
    elif generator_type == "vision":
        base_generator = GeneratorFactory.create_vision_generator(model, api_key)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

    # Wrap with presets if provided
    if preset_kwargs:
        return ConfiguredGenerator(base_generator, preset_kwargs, generator_type)
    else:
        return base_generator
