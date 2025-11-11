"""Tests for openai_retry utilities."""

import pytest
from unittest.mock import Mock

# Mock openai module if not available
try:
    import openai
except ImportError:
    openai = Mock()
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
    openai.InternalServerError = type("InternalServerError", (Exception,), {})
    openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
    openai.BadRequestError = type("BadRequestError", (Exception,), {})
    openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
    openai.PermissionDeniedError = type("PermissionDeniedError", (Exception,), {})
    openai.NotFoundError = type("NotFoundError", (Exception,), {})
    openai.UnprocessableEntityError = type("UnprocessableEntityError", (Exception,), {})

from llm_applications_library.utilities.openai_retry import (
    should_retry_openai_error,
    openai_retry,
    openai_retry_with_config,
)


class TestShouldRetryOpenaiError:
    """Test cases for should_retry_openai_error function."""

    def create_mock_response(self):
        """Create a mock response object for OpenAI exceptions."""
        mock_response = Mock()
        mock_response.request = Mock()
        return mock_response

    def test_should_retry_rate_limit_error(self):
        """Test that RateLimitError should be retried."""
        mock_response = self.create_mock_response()
        error = openai.RateLimitError(
            "Rate limit exceeded", response=mock_response, body=None
        )
        assert should_retry_openai_error(error) is True

    def test_should_retry_timeout_error(self):
        """Test that APITimeoutError should be retried."""
        error = openai.APITimeoutError(request=Mock())
        assert should_retry_openai_error(error) is True

    def test_should_retry_internal_server_error(self):
        """Test that InternalServerError should be retried."""
        mock_response = self.create_mock_response()
        error = openai.InternalServerError(
            "Internal server error", response=mock_response, body=None
        )
        assert should_retry_openai_error(error) is True

    def test_should_retry_connection_error(self):
        """Test that APIConnectionError should be retried."""
        error = openai.APIConnectionError(request=Mock())
        assert should_retry_openai_error(error) is True

    def test_should_not_retry_bad_request_error(self):
        """Test that BadRequestError should not be retried."""
        mock_response = self.create_mock_response()
        error = openai.BadRequestError("Bad request", response=mock_response, body=None)
        assert should_retry_openai_error(error) is False

    def test_should_not_retry_authentication_error(self):
        """Test that AuthenticationError should not be retried."""
        mock_response = self.create_mock_response()
        error = openai.AuthenticationError(
            "Authentication failed", response=mock_response, body=None
        )
        assert should_retry_openai_error(error) is False

    def test_should_not_retry_permission_denied_error(self):
        """Test that PermissionDeniedError should not be retried."""
        mock_response = self.create_mock_response()
        error = openai.PermissionDeniedError(
            "Permission denied", response=mock_response, body=None
        )
        assert should_retry_openai_error(error) is False

    def test_should_not_retry_not_found_error(self):
        """Test that NotFoundError should not be retried."""
        mock_response = self.create_mock_response()
        error = openai.NotFoundError("Not found", response=mock_response, body=None)
        assert should_retry_openai_error(error) is False

    def test_should_not_retry_unprocessable_entity_error(self):
        """Test that UnprocessableEntityError should not be retried."""
        mock_response = self.create_mock_response()
        error = openai.UnprocessableEntityError(
            "Unprocessable entity", response=mock_response, body=None
        )
        assert should_retry_openai_error(error) is False

    def test_should_not_retry_non_openai_error(self):
        """Test that non-OpenAI errors should not be retried."""
        error = ValueError("Some other error")
        assert should_retry_openai_error(error) is False

    def test_should_not_retry_generic_exception(self):
        """Test that generic exceptions should not be retried."""
        error = Exception("Generic error")
        assert should_retry_openai_error(error) is False


class TestOpenaiRetry:
    """Test cases for openai_retry decorator."""

    def test_openai_retry_default_parameters(self):
        """Test openai_retry with default parameters."""
        decorator = openai_retry()

        # The decorator should be callable
        assert callable(decorator)

        # Test that it returns a decorator function
        @decorator
        def dummy_function():
            return "success"

        assert callable(dummy_function)

    def test_openai_retry_custom_parameters(self):
        """Test openai_retry with custom parameters."""
        decorator = openai_retry(
            max_attempts=3, initial_wait=1.0, max_wait=30.0, multiplier=2.0
        )

        assert callable(decorator)

    def test_openai_retry_success_no_retry(self):
        """Test that successful function calls don't retry."""
        decorator = openai_retry(max_attempts=3)

        @decorator
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_openai_retry_retries_rate_limit_error(self):
        """Test that rate limit errors are retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=3, initial_wait=0.001, max_wait=0.01)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.RateLimitError(
                    "Rate limit", response=mock_response, body=None
                )
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3

    def test_openai_retry_retries_timeout_error(self):
        """Test that timeout errors are retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=2, initial_wait=0.001, max_wait=0.01)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise openai.APITimeoutError(request=Mock())
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 2

    def test_openai_retry_retries_internal_server_error(self):
        """Test that internal server errors are retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=2, initial_wait=0.001, max_wait=0.01)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.InternalServerError(
                    "Server error", response=mock_response, body=None
                )
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 2

    def test_openai_retry_retries_connection_error(self):
        """Test that connection errors are retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=2, initial_wait=0.001, max_wait=0.01)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise openai.APIConnectionError(request=Mock())
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 2

    def test_openai_retry_does_not_retry_bad_request(self):
        """Test that bad request errors are not retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=3)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            mock_response.request = Mock()
            raise openai.BadRequestError(
                "Bad request", response=mock_response, body=None
            )

        with pytest.raises(openai.BadRequestError):
            failing_function()

        assert call_count == 1  # Should not retry

    def test_openai_retry_does_not_retry_auth_error(self):
        """Test that authentication errors are not retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=3)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            mock_response.request = Mock()
            raise openai.AuthenticationError(
                "Auth error", response=mock_response, body=None
            )

        with pytest.raises(openai.AuthenticationError):
            failing_function()

        assert call_count == 1  # Should not retry

    def test_openai_retry_exhausts_attempts(self):
        """Test that retry exhausts attempts and raises RetryError."""
        call_count = 0

        decorator = openai_retry(max_attempts=2, initial_wait=0.001, max_wait=0.01)

        @decorator
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            mock_response.request = Mock()
            raise openai.RateLimitError("Rate limit", response=mock_response, body=None)

        with pytest.raises(openai.RateLimitError):
            always_failing_function()

        assert call_count == 2  # Should retry max_attempts times

    def test_openai_retry_with_non_openai_error(self):
        """Test that non-OpenAI errors are not retried."""
        call_count = 0

        decorator = openai_retry(max_attempts=3)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Some other error")

        with pytest.raises(ValueError):
            failing_function()

        assert call_count == 1  # Should not retry

    def test_openai_retry_preserves_function_signature(self):
        """Test that decorator preserves function signature and return values."""
        decorator = openai_retry()

        @decorator
        def function_with_args(a, b, c=None):
            return f"a={a}, b={b}, c={c}"

        result = function_with_args("x", "y", c="z")
        assert result == "a=x, b=y, c=z"

    def test_openai_retry_preserves_exceptions(self):
        """Test that decorator preserves exceptions that should not be retried."""
        decorator = openai_retry()

        @decorator
        def function_raising_value_error():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            function_raising_value_error()


class TestOpenaiRetryWithConfig:
    """Test cases for openai_retry_with_config function."""

    def test_openai_retry_with_config_default(self):
        """Test openai_retry_with_config with default configuration."""
        decorator = openai_retry_with_config()

        assert callable(decorator)

        @decorator
        def dummy_function():
            return "success"

        result = dummy_function()
        assert result == "success"

    def test_openai_retry_with_config_empty_dict(self):
        """Test openai_retry_with_config with empty configuration dict."""
        decorator = openai_retry_with_config({})

        assert callable(decorator)

    def test_openai_retry_with_config_custom_values(self):
        """Test openai_retry_with_config with custom configuration values."""
        config = {
            "max_attempts": 3,
            "initial_wait": 0.5,
            "max_wait": 30.0,
            "multiplier": 3.0,
        }

        decorator = openai_retry_with_config(config)
        assert callable(decorator)

    def test_openai_retry_with_config_partial_values(self):
        """Test openai_retry_with_config with partial configuration values."""
        config = {
            "max_attempts": 2,
            "initial_wait": 0.5,
            # max_wait and multiplier should use defaults
        }

        decorator = openai_retry_with_config(config)
        assert callable(decorator)

    def test_openai_retry_with_config_works_with_retryable_errors(self):
        """Test that openai_retry_with_config works with retryable errors."""
        call_count = 0
        config = {"max_attempts": 2, "initial_wait": 0.001, "max_wait": 0.01}

        decorator = openai_retry_with_config(config)

        @decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.RateLimitError(
                    "Rate limit", response=mock_response, body=None
                )
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 2

    def test_openai_retry_with_config_respects_max_attempts(self):
        """Test that openai_retry_with_config respects max_attempts setting."""
        call_count = 0
        config = {"max_attempts": 3, "initial_wait": 0.001, "max_wait": 0.01}

        decorator = openai_retry_with_config(config)

        @decorator
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            mock_response.request = Mock()
            raise openai.RateLimitError("Rate limit", response=mock_response, body=None)

        with pytest.raises(openai.RateLimitError):
            always_failing_function()

        assert call_count == 3  # Should respect the max_attempts setting

    def test_openai_retry_with_config_none_input(self):
        """Test openai_retry_with_config with None as input."""
        decorator = openai_retry_with_config(None)

        assert callable(decorator)

        @decorator
        def dummy_function():
            return "success"

        result = dummy_function()
        assert result == "success"


class TestIntegrationScenarios:
    """Integration test scenarios."""

    def test_real_world_scenario_with_eventual_success(self):
        """Test a realistic scenario where API succeeds after some retries."""
        attempt_count = 0

        @openai_retry(max_attempts=4, initial_wait=0.001, max_wait=0.01)
        def mock_openai_api_call():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.RateLimitError(
                    "Rate limit exceeded", response=mock_response, body=None
                )
            elif attempt_count == 2:
                raise openai.APITimeoutError(request=Mock())
            elif attempt_count == 3:
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.InternalServerError(
                    "Server error", response=mock_response, body=None
                )
            else:
                return {"choices": [{"message": {"content": "Hello!"}}]}

        result = mock_openai_api_call()
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert attempt_count == 4

    def test_immediate_failure_with_non_retryable_error(self):
        """Test that non-retryable errors fail immediately."""
        attempt_count = 0

        @openai_retry(max_attempts=3)
        def mock_openai_api_call():
            nonlocal attempt_count
            attempt_count += 1
            mock_response = Mock()
            mock_response.request = Mock()
            raise openai.AuthenticationError(
                "Invalid API key", response=mock_response, body=None
            )

        with pytest.raises(openai.AuthenticationError):
            mock_openai_api_call()

        assert attempt_count == 1

    def test_mixed_errors_scenario(self):
        """Test scenario with mixed retryable and non-retryable errors."""
        attempt_count = 0

        @openai_retry(max_attempts=3, initial_wait=0.001, max_wait=0.01)
        def mock_openai_api_call():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.RateLimitError(
                    "Rate limit", response=mock_response, body=None
                )
            elif attempt_count == 2:
                # This should not be retried
                mock_response = Mock()
                mock_response.request = Mock()
                raise openai.BadRequestError(
                    "Invalid request", response=mock_response, body=None
                )

        with pytest.raises(openai.BadRequestError):
            mock_openai_api_call()

        assert attempt_count == 2
