from enum import StrEnum

from pydantic import BaseModel

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
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    response_format: dict | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    n: int | None = None
    logit_bias: dict | None = None
    user: str | None = None
    tool_choice: str | dict | None = None
    tools: list[dict] | None = None

    model_config = {"extra": "ignore"}  # 未定義フィールドを無視


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


class TextGenerationUsage(BaseModel):
    """Text Generation APIの使用統計情報"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # OpenAI特有のフィールドも含める
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    model_config = {"extra": "ignore"}


class TextGenerationMeta(BaseModel):
    """Text Generatorのメタデータ"""

    # 通常のusage情報
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # エラー情報
    error: str | None = None
    retry_config: dict | None = None

    model_config = {"extra": "ignore"}


class TextGeneratorResponse(BaseModel):
    """Text Generator (RetryClaudeGenerator/RetryOpenAIGenerator) の共通返り値クラス

    OpenAI/Claude両方のText Generatorで統一された返り値形式を提供
    """

    replies: list[str]
    meta: list[TextGenerationMeta]

    model_config = {"extra": "ignore"}

    @classmethod
    def create_success(
        cls,
        content: str,
        usage: dict | None = None,
    ) -> "TextGeneratorResponse":
        """成功レスポンスを作成"""
        meta = TextGenerationMeta()

        if usage:
            # 異なる形式のusage情報を統一
            meta.input_tokens = usage.get("input_tokens", 0)
            meta.output_tokens = usage.get("output_tokens", 0)
            meta.total_tokens = usage.get("total_tokens", 0)
            # OpenAI形式のフィールドもサポート
            meta.prompt_tokens = usage.get("prompt_tokens")
            meta.completion_tokens = usage.get("completion_tokens")

        return cls(replies=[content], meta=[meta])

    @classmethod
    def create_error(
        cls,
        error: str,
        retry_config: dict | None = None,
    ) -> "TextGeneratorResponse":
        """エラーレスポンスを作成"""
        meta = TextGenerationMeta(error=error, retry_config=retry_config)
        return cls(replies=[], meta=[meta])

    def is_success(self) -> bool:
        """成功判定（repliesが空でない）"""
        return len(self.replies) > 0 and self.replies[0] is not None

    def get_content(self) -> str | None:
        """最初のコンテンツを取得"""
        if self.replies and len(self.replies) > 0:
            return self.replies[0]
        return None

    def get_error(self) -> str | None:
        """エラーメッセージを取得"""
        for meta in self.meta:
            if meta.error:
                return meta.error
        return None

    def get_usage(self) -> TextGenerationUsage | None:
        """使用統計を取得"""
        if not self.meta or not self.meta[0]:
            return None

        meta = self.meta[0]
        # Claude形式またはOpenAI形式のいずれかが存在する場合
        if (
            meta.input_tokens is not None
            or meta.output_tokens is not None
            or meta.prompt_tokens is not None
            or meta.completion_tokens is not None
            or meta.total_tokens is not None
        ):
            return TextGenerationUsage(
                input_tokens=meta.input_tokens or 0,
                output_tokens=meta.output_tokens or 0,
                total_tokens=meta.total_tokens or 0,
                prompt_tokens=meta.prompt_tokens,
                completion_tokens=meta.completion_tokens,
            )
        return None


class VisionAnalysisUsage(BaseModel):
    """Vision APIの使用統計情報"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    model_config = {"extra": "ignore"}


class VisionAnalysisResult(BaseModel):
    """Vision Generatorの個別分析結果"""

    success: bool
    content: str | None = None
    usage: VisionAnalysisUsage | None = None
    error: str | None = None

    model_config = {"extra": "ignore"}


class VisionGeneratorResponse(BaseModel):
    """Vision Generatorの共通返り値クラス

    OpenAI/Claude両方のVision Generatorで統一された返り値形式を提供
    """

    replies: list[VisionAnalysisResult]

    model_config = {"extra": "ignore"}

    @classmethod
    def create_success(
        cls,
        content: str,
        usage: dict | VisionAnalysisUsage | None = None,
    ) -> "VisionGeneratorResponse":
        """成功レスポンスを作成"""
        usage_obj: VisionAnalysisUsage | None = None
        if usage:
            if isinstance(usage, dict):
                usage_obj = VisionAnalysisUsage.model_validate(usage)
            else:
                usage_obj = usage

        result = VisionAnalysisResult(
            success=True, content=content, usage=usage_obj, error=None
        )
        return cls(replies=[result])

    @classmethod
    def create_error(
        cls,
        error: str,
        usage: dict | VisionAnalysisUsage | None = None,
    ) -> "VisionGeneratorResponse":
        """エラーレスポンスを作成"""
        usage_obj: VisionAnalysisUsage | None = None
        if usage:
            if isinstance(usage, dict):
                usage_obj = VisionAnalysisUsage.model_validate(usage)
            else:
                usage_obj = usage

        result = VisionAnalysisResult(
            success=False, content=None, usage=usage_obj, error=error
        )
        return cls(replies=[result])

    def is_success(self) -> bool:
        """全ての結果が成功かチェック"""
        return all(reply.success for reply in self.replies)

    def get_content(self) -> str | None:
        """最初の成功した結果のcontentを取得"""
        for reply in self.replies:
            if reply.success and reply.content:
                return reply.content
        return None

    def get_error(self) -> str | None:
        """最初のエラーを取得"""
        for reply in self.replies:
            if not reply.success and reply.error:
                return reply.error
        return None

    def get_total_usage(self) -> VisionAnalysisUsage:
        """全ての使用統計を合計"""
        total_input = 0
        total_output = 0

        for reply in self.replies:
            if reply.usage:
                total_input += reply.usage.input_tokens
                total_output += reply.usage.output_tokens

        return VisionAnalysisUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
        )
