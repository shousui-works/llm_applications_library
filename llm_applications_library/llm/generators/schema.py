from enum import StrEnum

from pydantic import BaseModel, model_validator

try:
    from pydantic import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic_settings import BaseSettings


class RetryConfig(BaseModel):
    """リトライ設定"""

    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    multiplier: float = 2.0


class ProviderType(StrEnum):
    """LLM Provider types."""

    OPENAI = "openai"
    CLAUDE = "claude"


class APIKeySettings(BaseSettings):
    """API key settings from environment variables."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars
    }


def get_api_key_settings() -> APIKeySettings:
    """Get API key settings from environment variables."""
    return APIKeySettings()


def get_provider_api_key(provider: ProviderType) -> str | None:
    """Get appropriate API key for the given provider."""
    settings = get_api_key_settings()

    if provider == ProviderType.OPENAI:
        return settings.openai_api_key
    elif provider == ProviderType.CLAUDE:
        return settings.anthropic_api_key
    else:
        return None


class Model(StrEnum):
    """llm model names."""

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Anthropic Claude models (verified working models)
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"  # Latest and most powerful
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250805"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Legacy aliases for backward compatibility
    CLAUDE_3_5_SONNET = CLAUDE_SONNET_4_5  # Redirect to latest
    CLAUDE_3_SONNET = CLAUDE_3_7_SONNET  # Redirect to 3.7 version


class OpenAIGenerationConfig(BaseModel):
    """OpenAI API用の生成設定（全ての有効なパラメーターを含む）"""

    temperature: float | None = None
    max_output_tokens: int | None = None  # Responses API用の新パラメータ
    max_completion_tokens: int | None = None  # GPT-5用の新パラメータ
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    n: int | None = None
    logit_bias: dict | None = None
    user: str | None = None
    tool_choice: str | dict | None = None
    tools: list[dict] | None = None

    model_config = {"extra": "ignore"}  # 未定義フィールドを無視

    @model_validator(mode="before")
    @classmethod
    def _alias_max_tokens(cls, data):
        """Support legacy max_tokens by remapping to max_output_tokens."""
        if isinstance(data, dict):
            if "max_tokens" in data and "max_output_tokens" not in data:
                data = data.copy()
                data["max_output_tokens"] = data.pop("max_tokens")
        return data


class ClaudeGenerationConfig(BaseModel):
    """Claude API用の生成設定（全ての有効なパラメーターを含む）"""

    temperature: float | None = None
    max_tokens: int | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None

    model_config = {"extra": "ignore"}  # 未定義フィールドを無視


class GPTConfig(BaseModel):
    """Configuration for GPT API calls."""

    model: str = Model.GPT_4_1
    generation_config: OpenAIGenerationConfig = OpenAIGenerationConfig()
    retry_config: RetryConfig = RetryConfig()


class ClaudeConfig(BaseModel):
    """Configuration for Claude API calls."""

    model: str = Model.CLAUDE_SONNET_4_5  # 最新・最強のClaude Sonnet 4.5を使用
    generation_config: ClaudeGenerationConfig = ClaudeGenerationConfig()
    retry_config: RetryConfig = RetryConfig()


class GeneratorUsage(BaseModel):
    """LLMジェネレータの使用統計情報（Text/Vision共通）"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # OpenAI特有のフィールドも含める
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    model_config = {"extra": "ignore"}


class GeneratorResponse(BaseModel):
    """LLMジェネレータ（Text/Vision）の統一レスポンスクラス

    OpenAI/Claude両方のText/Vision Generatorで統一された返り値形式を提供
    """

    status: str
    content: str | None = None
    usage: GeneratorUsage | None = None
    error: str | None = None

    model_config = {"extra": "ignore"}

    @classmethod
    def create_success(
        cls,
        content: str,
        usage: dict | GeneratorUsage | None = None,
    ) -> "GeneratorResponse":
        """成功レスポンスを作成"""
        usage_obj = None
        if usage:
            if isinstance(usage, dict):
                # 異なる形式のusage情報を統一
                usage_obj = GeneratorUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            else:
                usage_obj = usage

        return cls(status="success", content=content, usage=usage_obj, error=None)

    @classmethod
    def create_error(
        cls,
        error: str,
        usage: dict | GeneratorUsage | None = None,
    ) -> "GeneratorResponse":
        """エラーレスポンスを作成"""
        usage_obj = None
        if usage:
            if isinstance(usage, dict):
                usage_obj = GeneratorUsage.model_validate(usage)
            else:
                usage_obj = usage

        return cls(status="error", content=None, usage=usage_obj, error=error)

    def is_success(self) -> bool:
        """成功判定"""
        return self.status == "success"
