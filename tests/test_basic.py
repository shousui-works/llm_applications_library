"""Basic tests to ensure the package structure is correct."""

from llm_applications_library.llm.generators.schema import RetryConfig
from llm_applications_library.llm.generators.openai_custom_generator import openai_retry


def test_imports():
    """Test that basic imports work."""

    # Test that RetryConfig can be instantiated
    config = RetryConfig()
    assert config.max_attempts == 3
    assert config.initial_wait == 1.0

    # Test that openai_retry can be called
    assert callable(openai_retry)


def test_retry_config_defaults():
    """Test RetryConfig default values."""

    config = RetryConfig()
    assert config.max_attempts == 3
    assert config.initial_wait == 1.0
    assert config.max_wait == 60.0
    assert config.multiplier == 2.0


def test_retry_config_custom_values():
    """Test RetryConfig with custom values."""

    config = RetryConfig(
        max_attempts=5, initial_wait=2.0, max_wait=120.0, multiplier=3.0
    )
    assert config.max_attempts == 5
    assert config.initial_wait == 2.0
    assert config.max_wait == 120.0
    assert config.multiplier == 3.0
