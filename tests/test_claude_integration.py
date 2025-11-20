"""Integration tests for Claude API with actual API calls.

These tests require a valid ANTHROPIC_API_KEY environment variable.
They will be skipped if the API key is not available or if the 'integration' marker is not used.
"""

import base64
import os
import tempfile
from unittest.mock import patch

import pytest

from llm_applications_library.llm.generators.schema import ClaudeConfig, RetryConfig

# Skip all tests in this module if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration tests",
)


@pytest.fixture
def api_key():
    """Get API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")


@pytest.fixture
def claude_config():
    """Default Claude configuration for testing."""
    return ClaudeConfig()


@pytest.fixture
def retry_claude_config():
    """Claude configuration with fast retry settings for testing."""
    retry_config = RetryConfig(
        max_attempts=2, initial_wait=0.1, max_wait=1.0, multiplier=2.0
    )
    return ClaudeConfig(retry_config=retry_config)


@pytest.fixture
def test_image_base64():
    """Small test image encoded in base64."""
    # 1x1 pixel PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


class TestRetryClaudeGeneratorIntegration:
    """Integration tests for RetryClaudeGenerator with actual API calls."""

    @pytest.mark.integration
    def test_simple_text_generation(self, api_key):
        """Test basic text generation with actual API call."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = RetryClaudeGenerator(api_key=api_key)

        result = generator.run("Hello, please respond with just the word 'success'")

        assert "replies" in result
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], str)
        assert len(result["replies"][0]) > 0

        assert "meta" in result
        assert len(result["meta"]) == 1
        assert "input_tokens" in result["meta"][0]
        assert "output_tokens" in result["meta"][0]
        assert "total_tokens" in result["meta"][0]
        assert result["meta"][0]["input_tokens"] > 0
        assert result["meta"][0]["output_tokens"] > 0

    @pytest.mark.integration
    def test_text_generation_with_system_prompt(self, api_key):
        """Test text generation with system prompt."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = RetryClaudeGenerator(api_key=api_key)

        result = generator.run(
            "What is 2+2?",
            system_prompt="You are a calculator. Only respond with numbers.",
        )

        assert "replies" in result
        assert len(result["replies"]) == 1
        assert "4" in result["replies"][0]

    @pytest.mark.integration
    def test_text_generation_with_custom_config(self, api_key, retry_claude_config):
        """Test text generation with custom retry configuration."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = RetryClaudeGenerator(
            api_key=api_key, retry_config=retry_claude_config.retry_config
        )

        result = generator.run("Say 'configured' if you can see this.")

        assert "replies" in result
        assert len(result["replies"]) == 1
        assert "configured" in result["replies"][0].lower()

    @pytest.mark.integration
    def test_text_generation_with_generation_kwargs(self, api_key):
        """Test text generation with additional generation parameters."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = RetryClaudeGenerator(api_key=api_key)

        result = generator.run(
            "Write exactly one word.",
            generation_kwargs={"temperature": 0.1, "max_tokens": 10},
        )

        assert "replies" in result
        assert len(result["replies"]) == 1
        # With low temperature and max_tokens, response should be short
        assert len(result["replies"][0].split()) <= 5

    @pytest.mark.integration
    def test_error_handling_with_invalid_model(self, api_key):
        """Test error handling with invalid model name."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = RetryClaudeGenerator(api_key=api_key, model="invalid-model-name")

        result = generator.run("This should fail")

        # Should return empty replies with error in meta (for invalid model)
        assert "replies" in result
        assert result["replies"] == []
        assert "meta" in result
        assert len(result["meta"]) == 1
        assert "error" in result["meta"][0]


class TestClaudeVisionGeneratorIntegration:
    """Integration tests for ClaudeVisionGenerator with actual API calls."""

    @pytest.mark.integration
    def test_vision_analysis_with_base64(
        self, api_key, claude_config, test_image_base64
    ):
        """Test vision analysis with base64 encoded image."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = ClaudeVisionGenerator(api_key=api_key)

        result = generator.run(
            base64_images=[test_image_base64],
            mime_types=["image/png"],
            prompt="What do you see in this image? Respond briefly.",
            generation_kwargs=claude_config,
        )

        assert "replies" in result
        assert len(result["replies"]) == 1

        # Check the response structure
        response = result["replies"][0]
        assert "success" in response
        assert response["success"] is True
        assert "content" in response
        assert isinstance(response["content"], str)
        assert len(response["content"]) > 0
        assert "usage" in response
        assert response["usage"]["input_tokens"] > 0

    @pytest.mark.integration
    def test_vision_analysis_with_system_prompt(
        self, api_key, claude_config, test_image_base64
    ):
        """Test vision analysis with system prompt."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = ClaudeVisionGenerator(api_key=api_key)

        result = generator.run(
            base64_images=[test_image_base64],
            mime_types=["image/png"],
            prompt="Describe what you see.",
            generation_kwargs=claude_config,
            system_prompt="You are a concise image analyst. Keep responses under 20 words.",
        )

        assert "replies" in result
        response = result["replies"][0]
        assert response["success"] is True
        # With the system prompt, response should be concise
        assert len(response["content"].split()) <= 25  # Allowing some flexibility

    @pytest.mark.integration
    def test_vision_analysis_from_file(self, api_key, claude_config):
        """Test vision analysis from file path."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Write minimal PNG data
            tmp_file.write(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                )
            )
            tmp_file.flush()

            try:
                generator = ClaudeVisionGenerator(api_key=api_key)

                result = generator.run_from_file(
                    image_paths=[tmp_file.name],
                    prompt="What is this?",
                    generation_kwargs=claude_config,
                )

                assert "replies" in result
                response = result["replies"][0]
                assert response["success"] is True
                assert len(response["content"]) > 0

            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)

    @pytest.mark.integration
    def test_vision_error_handling_invalid_image(self, api_key, claude_config):
        """Test error handling with invalid base64 image data."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                ClaudeVisionGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        generator = ClaudeVisionGenerator(api_key=api_key)

        result = generator.run(
            base64_images=["invalid_base64_data"],
            mime_types=["image/png"],
            prompt="What do you see?",
            generation_kwargs=claude_config,
        )

        assert "replies" in result
        response = result["replies"][0]
        assert response["success"] is False
        assert response["content"] is None
        assert "error" in response


class TestRateLimitingIntegration:
    """Integration tests for rate limiting behavior."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_retry_on_rate_limit(self, api_key):
        """Test that the generator properly retries on rate limits.

        Note: This test may not always trigger rate limits.
        It's more of a stress test to verify retry behavior works.
        """
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        # Use fast retry config for quicker testing
        retry_config = RetryConfig(
            max_attempts=3, initial_wait=0.1, max_wait=2.0, multiplier=2.0
        )

        generator = RetryClaudeGenerator(api_key=api_key, retry_config=retry_config)

        # Make multiple rapid requests - might trigger rate limiting
        results = []
        for i in range(3):
            result = generator.run(f"Say 'request {i}' and nothing else.")
            results.append(result)

        # All requests should eventually succeed or fail gracefully
        for i, result in enumerate(results):
            assert "replies" in result
            # Either successful response or error in meta
            if result["replies"]:
                assert len(result["replies"]) == 1
                assert f"{i}" in result["replies"][0] or len(result["replies"][0]) > 0
            else:
                assert "meta" in result
                assert "error" in result["meta"][0]


@pytest.mark.integration
class TestEnvironmentConfiguration:
    """Test environment-based configuration."""

    def test_api_key_from_environment(self):
        """Test that API key is properly loaded from environment."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        # This should work with environment variable
        generator = RetryClaudeGenerator()
        assert generator.api_key == os.getenv("ANTHROPIC_API_KEY")

    def test_api_key_override(self, api_key):
        """Test that explicit API key overrides environment."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        override_key = "test-override-key"

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": api_key}):
            generator = RetryClaudeGenerator(api_key=override_key)
            assert generator.api_key == override_key

    def test_missing_api_key_error(self):
        """Test that missing API key raises appropriate error."""
        try:
            from llm_applications_library.llm.generators.claude_custom_generator import (
                RetryClaudeGenerator,
            )
        except ImportError:
            pytest.skip("anthropic package not available")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                RetryClaudeGenerator()
