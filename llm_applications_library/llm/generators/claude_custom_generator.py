"""Minimal Claude generator placeholders for testing and compatibility."""

from .schema import GeneratorResponse


# Provide a dummy anthropic attribute so tests can patch it safely.
class _DummyAnthropic:  # pragma: no cover - used only for patching
    pass


anthropic = _DummyAnthropic()


class RetryClaudeGenerator:
    def __init__(
        self, model: str | None = None, api_key: str | None = None, retry_config=None
    ):
        self.model = model or "claude-3-5-sonnet"
        self.api_key = api_key
        self.retry_config = retry_config

    def run(self, *args, **kwargs):
        # Placeholder implementation; real Claude calls are out of scope here.
        return GeneratorResponse.create_error(
            "Claude generator not implemented in this build."
        )


class ClaudeVisionGenerator:
    def __init__(
        self, model: str | None = None, api_key: str | None = None, retry_config=None
    ):
        self.model = model or "claude-3-5-sonnet"
        self.api_key = api_key
        self.retry_config = retry_config

    def run(self, *args, **kwargs):
        return GeneratorResponse.create_error(
            "Claude vision generator not implemented in this build."
        )
