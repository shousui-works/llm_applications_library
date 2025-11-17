"""Simplified tests for claude_retry utilities."""

import pytest
from unittest.mock import patch


# Create mock anthropic module for testing
class MockAnthropic:
    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class AuthenticationError(Exception):
        pass


@pytest.fixture
def mock_anthropic():
    """Fixture to provide mock anthropic module."""
    return MockAnthropic()


class TestShouldRetryClaudeError:
    """Test cases for should_retry_claude_error function."""

    def test_should_retry_retryable_errors(self, mock_anthropic):
        """Test that retryable errors return True."""
        from llm_applications_library.utilities.claude_retry import (
            should_retry_claude_error,
        )

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):
            # Test retryable errors
            assert should_retry_claude_error(mock_anthropic.RateLimitError()) is True
            assert should_retry_claude_error(mock_anthropic.APITimeoutError()) is True
            assert (
                should_retry_claude_error(mock_anthropic.InternalServerError()) is True
            )
            assert (
                should_retry_claude_error(mock_anthropic.APIConnectionError()) is True
            )

    def test_should_not_retry_non_retryable_errors(self, mock_anthropic):
        """Test that non-retryable errors return False."""
        from llm_applications_library.utilities.claude_retry import (
            should_retry_claude_error,
        )

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):
            # Test non-retryable errors
            assert should_retry_claude_error(mock_anthropic.BadRequestError()) is False
            assert (
                should_retry_claude_error(mock_anthropic.AuthenticationError()) is False
            )

            # Test generic exceptions
            assert should_retry_claude_error(ValueError()) is False
            assert should_retry_claude_error(Exception()) is False


class TestClaudeRetry:
    """Test cases for claude_retry decorator."""

    def test_claude_retry_decorator_creation(self):
        """Test that claude_retry decorator can be created."""
        from llm_applications_library.utilities.claude_retry import claude_retry

        decorator = claude_retry()
        assert callable(decorator)

        # Test with custom parameters
        decorator = claude_retry(max_attempts=3, initial_wait=0.5)
        assert callable(decorator)

    def test_claude_retry_success_no_retry(self, mock_anthropic):
        """Test that successful function calls don't retry."""
        from llm_applications_library.utilities.claude_retry import claude_retry

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):

            @claude_retry(max_attempts=3, initial_wait=0.01)
            def successful_function():
                return "success"

            result = successful_function()
            assert result == "success"

    def test_claude_retry_with_retryable_error(self, mock_anthropic):
        """Test that retryable errors are retried."""
        from llm_applications_library.utilities.claude_retry import claude_retry

        call_count = 0

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):

            @claude_retry(max_attempts=3, initial_wait=0.01, max_wait=0.05)
            def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise mock_anthropic.RateLimitError("Rate limit")
                return "success"

            result = failing_function()
            assert result == "success"
            assert call_count == 3

    def test_claude_retry_with_non_retryable_error(self, mock_anthropic):
        """Test that non-retryable errors are not retried."""
        from llm_applications_library.utilities.claude_retry import claude_retry

        call_count = 0

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):

            @claude_retry(max_attempts=3)
            def failing_function():
                nonlocal call_count
                call_count += 1
                raise mock_anthropic.BadRequestError("Bad request")

            with pytest.raises(MockAnthropic.BadRequestError):
                failing_function()

            assert call_count == 1  # Should not retry

    def test_claude_retry_exhausts_attempts(self, mock_anthropic):
        """Test that retry exhausts attempts and raises the original error."""
        from llm_applications_library.utilities.claude_retry import claude_retry

        call_count = 0

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):

            @claude_retry(max_attempts=2, initial_wait=0.01, max_wait=0.05)
            def always_failing_function():
                nonlocal call_count
                call_count += 1
                raise mock_anthropic.RateLimitError("Rate limit")

            with pytest.raises(MockAnthropic.RateLimitError):
                always_failing_function()

            assert call_count == 2


class TestClaudeRetryWithConfig:
    """Test cases for claude_retry_with_config function."""

    def test_claude_retry_with_config_default(self):
        """Test claude_retry_with_config with default configuration."""
        from llm_applications_library.utilities.claude_retry import (
            claude_retry_with_config,
        )

        decorator = claude_retry_with_config()
        assert callable(decorator)

    def test_claude_retry_with_config_custom_values(self):
        """Test claude_retry_with_config with custom configuration values."""
        from llm_applications_library.utilities.claude_retry import (
            claude_retry_with_config,
        )

        config = {
            "max_attempts": 3,
            "initial_wait": 0.5,
            "max_wait": 30.0,
            "multiplier": 3.0,
        }

        decorator = claude_retry_with_config(config)
        assert callable(decorator)

    def test_claude_retry_with_config_works_with_retryable_errors(self, mock_anthropic):
        """Test that claude_retry_with_config works with retryable errors."""
        from llm_applications_library.utilities.claude_retry import (
            claude_retry_with_config,
        )

        call_count = 0
        config = {"max_attempts": 2, "initial_wait": 0.01, "max_wait": 0.05}

        with patch(
            "llm_applications_library.utilities.claude_retry.anthropic", mock_anthropic
        ):

            @claude_retry_with_config(config)
            def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise mock_anthropic.RateLimitError("Rate limit")
                return "success"

            result = failing_function()
            assert result == "success"
            assert call_count == 2
