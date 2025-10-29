from enum import StrEnum

from pydantic import BaseModel


class RetryConfig(BaseModel):
    """リトライ設定"""

    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    multiplier: float = 2.0


class Model(StrEnum):
    """llm model names."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class OpenAIGenerationConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 4096
    response_format: dict[str, str] = {"type": "text"}


class GPTConfig(BaseModel):
    """Configuration for GPT API calls."""

    model: str = Model.GPT_4_1
    generation_config: OpenAIGenerationConfig = OpenAIGenerationConfig()
    retry_config: RetryConfig = RetryConfig()
