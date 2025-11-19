"""Tests for Claude custom generators."""

import base64
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from llm_applications_library.llm.generators.schema import RetryConfig

# Mock anthropic module if not available
try:
    import anthropic
except ImportError:
    anthropic = Mock()
    anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
    anthropic.APITimeoutError = type("APITimeoutError", (Exception,), {})
    anthropic.InternalServerError = type("InternalServerError", (Exception,), {})
    anthropic.APIConnectionError = type("APIConnectionError", (Exception,), {})
    anthropic.BadRequestError = type("BadRequestError", (Exception,), {})
    anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
    anthropic.PermissionDeniedError = type("PermissionDeniedError", (Exception,), {})
    anthropic.NotFoundError = type("NotFoundError", (Exception,), {})
    anthropic.UnprocessableEntityError = type(
        "UnprocessableEntityError", (Exception,), {}
    )


class TestRetryClaudeGenerator:
    """Test cases for RetryClaudeGenerator class."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator(api_key="test-api-key")
            assert generator.api_key == "test-api-key"
            assert generator.model == "claude-3-haiku-20240307"
            assert generator.retry_config is not None

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-api-key"}),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator()
            assert generator.api_key == "env-api-key"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            with pytest.raises(ValueError, match="API key is required"):
                RetryClaudeGenerator()

    def test_init_without_anthropic_raises_error(self):
        """Test that initialization without anthropic package raises error."""
        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic",
            None,
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            with pytest.raises(ImportError, match="anthropic package is required"):
                RetryClaudeGenerator(api_key="test-key")

    def test_run_success(self):
        """Test successful text generation."""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated text"
        mock_response.content = [mock_content]

        mock_usage = Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_response.usage = mock_usage

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator(api_key="test-key")
            result = generator.run("Test prompt")

            assert result["replies"] == ["Generated text"]
            assert result["meta"][0]["input_tokens"] == 10
            assert result["meta"][0]["output_tokens"] == 20
            assert result["meta"][0]["total_tokens"] == 30

    def test_run_with_system_prompt(self):
        """Test text generation with system prompt."""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated text"
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator(api_key="test-key")
            generator.run("Test prompt", system_prompt="System message")

            # Verify system prompt was passed
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "System message"

    def test_run_with_generation_kwargs(self):
        """Test text generation with additional kwargs."""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated text"
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator(api_key="test-key")
            generator.run("Test prompt", generation_kwargs={"temperature": 0.8})

            # Verify additional kwargs were passed
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["temperature"] == 0.8

    def test_run_error_handling(self):
        """Test error handling in run method."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator(api_key="test-key")
            result = generator.run("Test prompt")

            assert result["replies"] == []
            assert "error" in result["meta"][0]
            assert "API Error" in result["meta"][0]["error"]

    def test_retry_config(self):
        """Test custom retry configuration."""
        custom_config = RetryConfig(max_attempts=5, initial_wait=2.0)

        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )

            generator = RetryClaudeGenerator(
                api_key="test-key", retry_config=custom_config
            )
            assert generator.retry_config.max_attempts == 5
            assert generator.retry_config.initial_wait == 2.0


class TestClaudeVisionGenerator:
    """Test cases for ClaudeVisionGenerator class."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            generator = ClaudeVisionGenerator(api_key="test-api-key")
            assert generator.api_key == "test-api-key"
            assert generator.model == "claude-3-haiku-20240307"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            with pytest.raises(ValueError, match="API key is required"):
                ClaudeVisionGenerator()

    def test_init_without_anthropic_raises_error(self):
        """Test that initialization without anthropic package raises error."""
        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic",
            None,
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            with pytest.raises(ImportError, match="anthropic package is required"):
                ClaudeVisionGenerator(api_key="test-key")

    def test_run_vision_success(self):
        """Test successful vision analysis."""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Image analysis result"
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            generator = ClaudeVisionGenerator(api_key="test-key")

            # Create a test image (small base64 encoded image)
            test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

            result = generator.run(
                base64_image=test_image_b64,
                mime_type="image/png",
                system_prompt="Analyze this image",
                generation_kwargs={"temperature": 0.1, "max_tokens": 100},
            )

            assert "replies" in result
            assert len(result["replies"]) == 1

            # Check the response structure - it should contain the _chat_completion response
            response = result["replies"][0]
            assert "success" in response
            assert "content" in response
            assert "usage" in response

            # Verify the correct message format was sent
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            messages = call_kwargs["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert len(messages[0]["content"]) == 2
            assert messages[0]["content"][0]["type"] == "image"
            assert messages[0]["content"][1]["type"] == "text"

    def test_run_vision_with_system_prompt(self):
        """Test vision analysis with system prompt."""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Image analysis result"
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            generator = ClaudeVisionGenerator(api_key="test-key")
            test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

            result = generator.run(
                base64_image=test_image_b64,
                mime_type="image/png",
                system_prompt="You are an expert image analyst",
                generation_kwargs={"temperature": 0.1, "max_tokens": 100},
            )

            # Check response structure
            assert "replies" in result
            response = result["replies"][0]
            assert "success" in response

            # Verify system prompt was passed
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are an expert image analyst"

    def test_run_from_file_success(self):
        """Test successful vision analysis from file."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Write a minimal PNG file
            tmp_file.write(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                )
            )
            tmp_file.flush()

            mock_response = Mock()
            mock_content = Mock()
            mock_content.text = "Image analysis result"
            mock_response.content = [mock_content]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)

            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response

            try:
                with (
                    patch(
                        "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
                    ),
                    patch(
                        "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                        return_value=mock_client,
                    ),
                ):
                    from llm_applications_library.llm.generators.claude_custom_generator import (
                        ClaudeVisionGenerator,
                    )

                    generator = ClaudeVisionGenerator(api_key="test-key")

                    result = generator.run_from_file(
                        image_path=tmp_file.name,
                        system_prompt="Analyze this image",
                        generation_kwargs={"temperature": 0.1, "max_tokens": 100},
                    )

                    assert "replies" in result
                    response = result["replies"][0]
                    assert "success" in response
                    mock_client.messages.create.assert_called_once()

            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)

    def test_run_from_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with patch(
            "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            generator = ClaudeVisionGenerator(api_key="test-key")

            with pytest.raises(FileNotFoundError, match="Image file not found"):
                generator.run_from_file(
                    image_path="non_existent_file.png",
                    system_prompt="Analyze this image",
                    generation_kwargs={"temperature": 0.1, "max_tokens": 100},
                )

    def test_run_from_file_invalid_type(self):
        """Test that ValueError is raised for invalid file type."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not an image")
            tmp_file.flush()

            try:
                with patch(
                    "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
                ):
                    from llm_applications_library.llm.generators.claude_custom_generator import (
                        ClaudeVisionGenerator,
                    )

                    generator = ClaudeVisionGenerator(api_key="test-key")

                    with pytest.raises(ValueError, match="Unsupported file type"):
                        generator.run_from_file(
                            image_path=tmp_file.name,
                            system_prompt="Analyze this image",
                            generation_kwargs={"temperature": 0.1},
                        )

            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)

    def test_chat_completion_error_handling(self):
        """Test error handling in _chat_completion method."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with (
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic"
            ),
            patch(
                "llm_applications_library.llm.generators.claude_custom_generator.anthropic.Anthropic",
                return_value=mock_client,
            ),
        ):
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )

            generator = ClaudeVisionGenerator(api_key="test-key")

            result = generator._chat_completion(
                messages=[{"role": "user", "content": "test"}]
            )

            assert result["success"] is False
            assert result["content"] is None
            assert result["usage"] is None
            assert "API Error" in result["error"]
