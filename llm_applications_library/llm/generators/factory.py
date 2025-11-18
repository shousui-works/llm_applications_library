"""Generator Factory for automatic selection between OpenAI and Claude."""

import logging
from typing import Any, Protocol, Union

from .schema import (
    Model,
    RetryConfig,
    GPTConfig,
    ClaudeConfig,
    ProviderType,
    get_provider_api_key,
)
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
    logger.debug(
        f"Detecting provider for model: '{model}' (lowercase: '{model_lower}')"
    )

    # Check if it's a known enum value
    try:
        model_enum = Model(model)
        model_name = model_enum.value
        logger.debug(f"Model found in enum with value: '{model_name}'")
    except ValueError:
        # Custom model name, use the original
        model_name = model
        logger.debug(f"Model not in enum, using original name: '{model_name}'")

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
    claude_patterns = [
        "claude-",
        "claude-3",
        "claude-3-5",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "anthropic",
    ]

    # Check patterns
    for pattern in openai_patterns:
        if pattern in model_lower:
            logger.debug(f"Model '{model}' matched OpenAI pattern: '{pattern}'")
            return ProviderType.OPENAI

    for pattern in claude_patterns:
        if pattern in model_lower:
            logger.debug(f"Model '{model}' matched Claude pattern: '{pattern}'")
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
        Model.CLAUDE_SONNET_4_5,
        Model.CLAUDE_OPUS_4_1,
        Model.CLAUDE_OPUS_4,
        Model.CLAUDE_SONNET_4,
        Model.CLAUDE_3_7_SONNET,
        Model.CLAUDE_3_5_HAIKU,
        Model.CLAUDE_3_OPUS,
        Model.CLAUDE_3_HAIKU,
        Model.CLAUDE_3_5_SONNET,  # Legacy alias
    ]

    if model_name in [m.value for m in openai_models]:
        logger.debug(f"Model '{model}' found in OpenAI model list")
        return ProviderType.OPENAI

    if model_name in [m.value for m in claude_models]:
        logger.debug(f"Model '{model}' found in Claude model list")
        return ProviderType.CLAUDE

    logger.error(
        f"Cannot detect provider for model: '{model}'. Checked patterns: OpenAI={openai_patterns}, Claude={claude_patterns}"
    )
    raise ValueError(f"Cannot detect provider for model: {model}")


def debug_model_detection(model: str) -> dict[str, Any]:
    """
    Debug function to show detailed model detection information.

    Args:
        model: Model name to debug

    Returns:
        Dictionary with detection details
    """
    model_lower = model.lower()
    result = {
        "input_model": model,
        "lowercase_model": model_lower,
        "matched_patterns": [],
        "checked_patterns": {
            "openai": [
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
            ],
            "claude": [
                "claude-",
                "claude-3",
                "claude-3-5",
                "claude-3-opus",
                "claude-3-sonnet",
                "claude-3-haiku",
                "anthropic",
            ],
        },
    }

    # Check patterns
    for pattern in result["checked_patterns"]["openai"]:
        if pattern in model_lower:
            result["matched_patterns"].append(f"OpenAI: {pattern}")

    for pattern in result["checked_patterns"]["claude"]:
        if pattern in model_lower:
            result["matched_patterns"].append(f"Claude: {pattern}")

    try:
        result["detected_provider"] = detect_provider_from_model(model).value
    except ValueError as e:
        result["detection_error"] = str(e)

    return result


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
        retry_config: RetryConfig | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> TextGenerator:
        """
        Create a text generator for the specified model.

        Args:
            model: Model name (from Model enum or custom string)
            retry_config: Retry configuration (optional)
            generation_kwargs: Default generation parameters (optional)

        Returns:
            Appropriate text generator instance (API key auto-detected from environment)

        Raises:
            ValueError: If model provider cannot be determined or is unsupported
            ImportError: If required packages for the provider are not installed
        """
        provider = detect_provider_from_model(model)

        # Get appropriate API key from environment
        api_key = get_provider_api_key(provider)

        if provider == ProviderType.OPENAI:
            base_generator = RetryOpenAIGenerator(
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

            base_generator = RetryClaudeGenerator(
                model=model,
                api_key=api_key,
                retry_config=retry_config,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # If generation_kwargs provided, wrap with preset functionality
        if generation_kwargs:

            class PresetTextGenerator:
                def __init__(self, generator, preset_kwargs):
                    self._generator = generator
                    self._preset_kwargs = preset_kwargs
                    # Copy attributes from base generator
                    for attr in dir(generator):
                        if not attr.startswith("_") and not callable(
                            getattr(generator, attr)
                        ):
                            setattr(self, attr, getattr(generator, attr))

                def run(
                    self,
                    prompt: str,
                    system_prompt: str | None = None,
                    generation_kwargs: dict[str, Any] | None = None,
                ) -> dict[str, Any]:
                    # Merge preset kwargs with provided kwargs
                    merged_kwargs = self._preset_kwargs.copy()
                    if generation_kwargs:
                        merged_kwargs.update(generation_kwargs)
                    return self._generator.run(prompt, system_prompt, merged_kwargs)

                def __getattr__(self, name):
                    return getattr(self._generator, name)

            return PresetTextGenerator(base_generator, generation_kwargs)
        else:
            return base_generator

    @staticmethod
    def create_vision_generator(
        model: str,
        model_config: Union[GPTConfig, ClaudeConfig] | None = None,
        retry_config: RetryConfig | None = None,
    ) -> VisionGenerator:
        """
        Create a vision generator for the specified model.

        Args:
            model: Model name (from Model enum or custom string)
            model_config: Default model configuration (optional)
            retry_config: Retry configuration (optional)

        Returns:
            Appropriate vision generator instance (API key auto-detected from environment)

        Raises:
            ValueError: If model provider cannot be determined or is unsupported
            ImportError: If required packages for the provider are not installed
        """
        provider = detect_provider_from_model(model)

        # Get appropriate API key from environment
        api_key = get_provider_api_key(provider)

        # Vision generators now support retry_config directly

        if provider == ProviderType.OPENAI:
            base_generator = OpenAIVisionGenerator(
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
            from .claude_custom_generator import ClaudeVisionGenerator

            base_generator = ClaudeVisionGenerator(
                model=model,
                api_key=api_key,
                retry_config=retry_config,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # If model_config provided, wrap with preset functionality
        if model_config:

            class PresetVisionGenerator:
                def __init__(self, generator, preset_config):
                    self._generator = generator
                    self._preset_config = preset_config
                    # Copy attributes from base generator
                    for attr in dir(generator):
                        if not attr.startswith("_") and not callable(
                            getattr(generator, attr)
                        ):
                            setattr(self, attr, getattr(generator, attr))

                def run(
                    self,
                    base64_image: str,
                    mime_type: str,
                    system_prompt: str | None = None,
                    generation_kwargs: dict[str, Any] | None = None,
                ) -> dict[str, Any]:
                    # Merge preset config with provided generation_kwargs
                    merged_kwargs = {}
                    if hasattr(self._preset_config, "generation_config"):
                        merged_kwargs.update(
                            self._preset_config.generation_config.model_dump()
                        )
                    if generation_kwargs:
                        merged_kwargs.update(generation_kwargs)

                    return self._generator.run(
                        base64_image, mime_type, system_prompt, merged_kwargs
                    )

                def __getattr__(self, name):
                    return getattr(self._generator, name)

            return PresetVisionGenerator(base_generator, model_config)
        else:
            return base_generator

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
    retry_config: RetryConfig | None = None,
) -> Union[TextGenerator, VisionGenerator]:
    """
    Convenience function to create a generator.

    Args:
        model: Model name
        generator_type: Type of generator ("text" or "vision")
        retry_config: Retry configuration (optional)

    Returns:
        Appropriate generator instance (API keys auto-detected from environment)

    Examples:
        # Basic usage (API keys from environment variables)
        gen = create_generator("gpt-4o", "text")

        # With retry configuration
        retry_config = RetryConfig(max_attempts=5)
        gen = create_generator("claude-3-haiku", "text", retry_config)
    """
    if generator_type == "text":
        return GeneratorFactory.create_text_generator(model, retry_config=retry_config)
    elif generator_type == "vision":
        return GeneratorFactory.create_vision_generator(
            model, retry_config=retry_config
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
