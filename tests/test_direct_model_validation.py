"""Tests for direct Pydantic model validation."""

import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from llm_applications_library.llm.generators.schema import (
    OpenAIGenerationConfig,
    ClaudeGenerationConfig,
)


class TestDirectModelValidation:
    """Test direct Pydantic model validation."""

    def test_openai_valid_params(self):
        """Test OpenAI model validation with valid parameters."""
        valid_kwargs = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
        }

        config = OpenAIGenerationConfig.model_validate(valid_kwargs)
        result = config.model_dump(exclude_none=True)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
        assert result["top_p"] == 0.9
        assert result["frequency_penalty"] == 0.1

    def test_openai_invalid_params_raises_error(self):
        """Test OpenAI model validation with invalid parameters."""
        invalid_kwargs = {
            "temperature": 0.7,
            "top_k": 50,  # Claude-specific param
            "invalid_param": "value",
        }

        with pytest.raises(ValidationError) as exc_info:
            OpenAIGenerationConfig.model_validate(invalid_kwargs)

        error_msg = str(exc_info.value)
        assert "Extra inputs are not permitted" in error_msg

    def test_claude_valid_params(self):
        """Test Claude model validation with valid parameters."""
        valid_kwargs = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_k": 50,
            "stop_sequences": ["END"],
        }

        config = ClaudeGenerationConfig.model_validate(valid_kwargs)
        result = config.model_dump(exclude_none=True)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
        assert result["top_k"] == 50
        assert result["stop_sequences"] == ["END"]

    def test_claude_invalid_params_raises_error(self):
        """Test Claude model validation with invalid parameters."""
        invalid_kwargs = {
            "temperature": 0.7,
            "frequency_penalty": 0.1,  # OpenAI-specific param
            "invalid_param": "value",
        }

        with pytest.raises(ValidationError) as exc_info:
            ClaudeGenerationConfig.model_validate(invalid_kwargs)

        error_msg = str(exc_info.value)
        assert "Extra inputs are not permitted" in error_msg

    def test_type_coercion(self):
        """Test automatic type coercion."""
        kwargs = {
            "temperature": "0.7",  # string -> float
            "max_tokens": "1000",  # string -> int
        }

        config = OpenAIGenerationConfig.model_validate(kwargs)
        result = config.model_dump(exclude_none=True)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
        assert isinstance(result["temperature"], float)
        assert isinstance(result["max_tokens"], int)

    def test_exclude_none_filtering(self):
        """Test that None values are properly excluded."""
        kwargs = {"temperature": 0.7}  # Only one field set

        config = OpenAIGenerationConfig.model_validate(kwargs)
        result = config.model_dump(exclude_none=True)

        assert "temperature" in result
        assert "max_tokens" not in result  # None value excluded
        assert "top_p" not in result  # None value excluded

    def test_complex_types(self):
        """Test complex types like dicts and lists."""
        kwargs = {
            "response_format": {"type": "json_object"},
            "stop": ["END", "STOP"],
            "tools": [{"type": "function", "function": {"name": "test"}}],
        }

        config = OpenAIGenerationConfig.model_validate(kwargs)
        result = config.model_dump(exclude_none=True)

        assert result["response_format"]["type"] == "json_object"
        assert result["stop"] == ["END", "STOP"]
        assert len(result["tools"]) == 1


class TestIntegrationWithGenerators:
    """Test integration with actual generators."""

    def test_openai_generator_direct_validation(self):
        """Test direct validation in OpenAI generator context."""
        from llm_applications_library.llm.generators.openai_custom_generator import (
            RetryOpenAIGenerator,
        )

        generator = RetryOpenAIGenerator(api_key="test-key")

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage = Mock()
            mock_response.usage.model_dump.return_value = {"total_tokens": 10}
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Valid params should work
            result = generator.run("test", generation_kwargs={"temperature": 0.7})
            assert "replies" in result

            # Invalid params should raise ValidationError
            with pytest.raises(ValidationError):
                generator.run("test", generation_kwargs={"invalid_param": "value"})

    def test_claude_generator_direct_validation(self):
        """Test direct validation in Claude generator context."""
        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            RetryClaudeGenerator(api_key="test-key")

            # Test validation directly without running the full generator
            # Valid params should not raise error during validation
            try:
                ClaudeGenerationConfig.model_validate({"temperature": 0.7})
            except Exception as e:
                pytest.fail(f"Valid params should not raise error: {e}")

            # Invalid params should raise ValidationError during validation
            with pytest.raises(ValidationError):
                ClaudeGenerationConfig.model_validate({"invalid_param": "value"})
