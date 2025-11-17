"""Tests for the generator factory module."""

import pytest
from unittest.mock import patch, MagicMock

from llm_applications_library.llm.generators.factory import (
    GeneratorFactory,
    detect_provider_from_model,
    get_default_config_for_provider,
    create_generator,
    ProviderType,
    CLAUDE_AVAILABLE,
)
from llm_applications_library.llm.generators.schema import (
    Model,
    RetryConfig,
    GPTConfig,
    ClaudeConfig,
)


class TestProviderDetection:
    """Test provider detection functionality."""

    def test_detect_openai_models(self):
        """Test detection of OpenAI models."""
        openai_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
            "chatgpt",
            "text-davinci-003",
        ]

        for model in openai_models:
            provider = detect_provider_from_model(model)
            assert provider == ProviderType.OPENAI, (
                f"Failed to detect OpenAI for {model}"
            )

    def test_detect_claude_models(self):
        """Test detection of Claude models."""
        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "anthropic-claude",
        ]

        for model in claude_models:
            provider = detect_provider_from_model(model)
            assert provider == ProviderType.CLAUDE, (
                f"Failed to detect Claude for {model}"
            )

    def test_detect_enum_models(self):
        """Test detection using Model enum values."""
        # Test OpenAI enum models
        openai_enums = [
            Model.GPT_4O,
            Model.GPT_4O_MINI,
            Model.GPT_4,
        ]

        for model_enum in openai_enums:
            provider = detect_provider_from_model(model_enum.value)
            assert provider == ProviderType.OPENAI

        # Test Claude enum models
        claude_enums = [
            Model.CLAUDE_3_HAIKU,
            Model.CLAUDE_3_SONNET,
            Model.CLAUDE_3_5_SONNET,
        ]

        for model_enum in claude_enums:
            provider = detect_provider_from_model(model_enum.value)
            assert provider == ProviderType.CLAUDE

    def test_detect_unknown_model_raises_error(self):
        """Test that unknown models raise ValueError."""
        unknown_models = [
            "unknown-model",
            "custom-llm-v1",
            "",
        ]

        for model in unknown_models:
            with pytest.raises(ValueError, match="Cannot detect provider"):
                detect_provider_from_model(model)

    def test_case_insensitive_detection(self):
        """Test that detection is case insensitive."""
        test_cases = [
            ("GPT-4O", ProviderType.OPENAI),
            ("Claude-3-Haiku", ProviderType.CLAUDE),
            ("CHATGPT", ProviderType.OPENAI),
        ]

        for model, expected_provider in test_cases:
            provider = detect_provider_from_model(model)
            assert provider == expected_provider


class TestDefaultConfiguration:
    """Test default configuration creation."""

    def test_get_default_openai_config(self):
        """Test getting default OpenAI configuration."""
        config = get_default_config_for_provider(ProviderType.OPENAI)
        assert isinstance(config, GPTConfig)
        assert isinstance(config.retry_config, RetryConfig)

    def test_get_default_claude_config(self):
        """Test getting default Claude configuration."""
        config = get_default_config_for_provider(ProviderType.CLAUDE)
        assert isinstance(config, ClaudeConfig)
        assert isinstance(config.retry_config, RetryConfig)

    def test_get_default_config_with_custom_retry(self):
        """Test getting default config with custom retry settings."""
        custom_retry = RetryConfig(max_attempts=5, initial_wait=2.0)
        config = get_default_config_for_provider(ProviderType.OPENAI, custom_retry)

        assert config.retry_config == custom_retry
        assert config.retry_config.max_attempts == 5
        assert config.retry_config.initial_wait == 2.0

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_default_config_for_provider("invalid_provider")


class TestGeneratorFactory:
    """Test the GeneratorFactory class."""

    @patch("llm_applications_library.llm.generators.factory.RetryOpenAIGenerator")
    def test_create_openai_text_generator(self, mock_openai_gen):
        """Test creation of OpenAI text generator."""
        mock_instance = MagicMock()
        mock_openai_gen.return_value = mock_instance

        generator = GeneratorFactory.create_text_generator("gpt-4o")

        mock_openai_gen.assert_called_once_with(
            model="gpt-4o",
            api_key=None,
            retry_config=None,
        )
        assert generator == mock_instance

    @patch("llm_applications_library.llm.generators.factory.RetryOpenAIGenerator")
    def test_create_openai_text_generator_with_params(self, mock_openai_gen):
        """Test creation of OpenAI text generator with parameters."""
        mock_instance = MagicMock()
        mock_openai_gen.return_value = mock_instance

        retry_config = RetryConfig(max_attempts=5)
        api_key = "test-key"

        GeneratorFactory.create_text_generator(
            "gpt-4o", api_key=api_key, retry_config=retry_config
        )

        mock_openai_gen.assert_called_once_with(
            model="gpt-4o",
            api_key=api_key,
            retry_config=retry_config,
        )

    @pytest.mark.skipif(not CLAUDE_AVAILABLE, reason="Claude not available")
    @patch(
        "llm_applications_library.llm.generators.claude_custom_generator.RetryClaudeGenerator"
    )
    def test_create_claude_text_generator(self, mock_claude_gen):
        """Test creation of Claude text generator when available."""
        mock_instance = MagicMock()
        mock_claude_gen.return_value = mock_instance

        generator = GeneratorFactory.create_text_generator("claude-3-haiku-20240307")

        mock_claude_gen.assert_called_once_with(
            model="claude-3-haiku-20240307",
            api_key=None,
            retry_config=None,
        )
        assert generator == mock_instance

    @patch("llm_applications_library.llm.generators.factory.OpenAIVisionGenerator")
    def test_create_openai_vision_generator(self, mock_openai_vision):
        """Test creation of OpenAI vision generator."""
        mock_instance = MagicMock()
        mock_openai_vision.return_value = mock_instance

        generator = GeneratorFactory.create_vision_generator("gpt-4o")

        mock_openai_vision.assert_called_once_with(
            model="gpt-4o",
            api_key=None,
        )
        assert generator == mock_instance

    @pytest.mark.skipif(not CLAUDE_AVAILABLE, reason="Claude not available")
    @patch(
        "llm_applications_library.llm.generators.claude_custom_generator.ClaudeVisionGenerator"
    )
    def test_create_claude_vision_generator(self, mock_claude_vision):
        """Test creation of Claude vision generator when available."""
        mock_instance = MagicMock()
        mock_claude_vision.return_value = mock_instance

        generator = GeneratorFactory.create_vision_generator("claude-3-haiku-20240307")

        mock_claude_vision.assert_called_once_with(
            model="claude-3-haiku-20240307",
            api_key=None,
        )
        assert generator == mock_instance

    def test_create_text_generator_unknown_model_raises_error(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect provider"):
            GeneratorFactory.create_text_generator("unknown-model")

    def test_create_vision_generator_unknown_model_raises_error(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect provider"):
            GeneratorFactory.create_vision_generator("unknown-model")

    def test_get_provider_for_model(self):
        """Test getting provider for model."""
        assert GeneratorFactory.get_provider_for_model("gpt-4o") == ProviderType.OPENAI
        assert (
            GeneratorFactory.get_provider_for_model("claude-3-haiku")
            == ProviderType.CLAUDE
        )

    def test_get_default_config(self):
        """Test getting default config for model."""
        openai_config = GeneratorFactory.get_default_config("gpt-4o")
        assert isinstance(openai_config, GPTConfig)

        claude_config = GeneratorFactory.get_default_config("claude-3-haiku")
        assert isinstance(claude_config, ClaudeConfig)


class TestClaudeUnavailableScenario:
    """Test behavior when Claude is not available."""

    @patch("llm_applications_library.llm.generators.factory.CLAUDE_AVAILABLE", False)
    def test_claude_text_generator_unavailable_raises_import_error(self):
        """Test that Claude unavailability raises ImportError."""
        with pytest.raises(ImportError, match="Claude generators require.*anthropic"):
            GeneratorFactory.create_text_generator("claude-3-haiku-20240307")

    @patch("llm_applications_library.llm.generators.factory.CLAUDE_AVAILABLE", False)
    def test_claude_vision_generator_unavailable_raises_import_error(self):
        """Test that Claude unavailability raises ImportError."""
        with pytest.raises(ImportError, match="Claude generators require.*anthropic"):
            GeneratorFactory.create_vision_generator("claude-3-haiku-20240307")


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch(
        "llm_applications_library.llm.generators.factory.GeneratorFactory.create_text_generator"
    )
    def test_create_text_generator_convenience(self, mock_create_text):
        """Test convenience function for text generator."""
        mock_generator = MagicMock()
        mock_create_text.return_value = mock_generator

        retry_config = RetryConfig()
        result = create_generator("gpt-4o", "text", "test-key", retry_config)

        mock_create_text.assert_called_once_with("gpt-4o", "test-key", retry_config)
        assert result == mock_generator

    @patch(
        "llm_applications_library.llm.generators.factory.GeneratorFactory.create_vision_generator"
    )
    def test_create_vision_generator_convenience(self, mock_create_vision):
        """Test convenience function for vision generator."""
        mock_generator = MagicMock()
        mock_create_vision.return_value = mock_generator

        result = create_generator("gpt-4o", "vision", "test-key")

        mock_create_vision.assert_called_once_with("gpt-4o", "test-key")
        assert result == mock_generator

    def test_create_generator_invalid_type_raises_error(self):
        """Test that invalid generator type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown generator type"):
            create_generator("gpt-4o", "invalid_type")

    def test_create_generator_default_parameters(self):
        """Test create_generator with default parameters."""
        with patch(
            "llm_applications_library.llm.generators.factory.GeneratorFactory.create_text_generator"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            create_generator("gpt-4o")
            mock_create.assert_called_once_with("gpt-4o", None, None)

    def test_factory_create_text_generator_with_generation_kwargs(self):
        """Test that create_text_generator can accept generation_kwargs."""
        with patch(
            "llm_applications_library.llm.generators.factory.RetryOpenAIGenerator"
        ) as mock_openai_gen:
            mock_instance = MagicMock()
            mock_openai_gen.return_value = mock_instance

            generation_kwargs = {"temperature": 0.8, "max_tokens": 100}
            generator = GeneratorFactory.create_text_generator(
                "gpt-4o", generation_kwargs=generation_kwargs
            )

            # Should create a PresetTextGenerator wrapper
            assert hasattr(generator, "_generator")
            assert hasattr(generator, "_preset_kwargs")

    def test_factory_create_vision_generator_with_model_config(self):
        """Test that create_vision_generator can accept model_config."""
        with patch(
            "llm_applications_library.llm.generators.factory.OpenAIVisionGenerator"
        ) as mock_openai_vision:
            mock_instance = MagicMock()
            mock_openai_vision.return_value = mock_instance

            from llm_applications_library.llm.generators.schema import GPTConfig

            model_config = GPTConfig()
            generator = GeneratorFactory.create_vision_generator(
                "gpt-4o", model_config=model_config
            )

            # Should create a PresetVisionGenerator wrapper
            assert hasattr(generator, "_generator")
            assert hasattr(generator, "_preset_config")


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_model_string_raises_error(self):
        """Test that empty model string raises error."""
        with pytest.raises(ValueError, match="Cannot detect provider"):
            detect_provider_from_model("")

    def test_whitespace_model_string_raises_error(self):
        """Test that whitespace-only model string raises error."""
        with pytest.raises(ValueError, match="Cannot detect provider"):
            detect_provider_from_model("   ")

    def test_special_characters_in_model_name(self):
        """Test model names with special characters."""
        # Valid model with special characters
        provider = detect_provider_from_model("gpt-4o-2024-05-13")
        assert provider == ProviderType.OPENAI

        provider = detect_provider_from_model("claude-3-5-sonnet-20241022")
        assert provider == ProviderType.CLAUDE

    @patch("llm_applications_library.llm.generators.factory.RetryOpenAIGenerator")
    def test_none_api_key_handling(self, mock_openai_gen):
        """Test that None API key is handled properly."""
        mock_openai_gen.return_value = MagicMock()

        GeneratorFactory.create_text_generator("gpt-4o", api_key=None)

        mock_openai_gen.assert_called_once_with(
            model="gpt-4o",
            api_key=None,
            retry_config=None,
        )
