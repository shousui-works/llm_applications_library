import logging
from typing import Any, TypeVar
from collections.abc import Callable

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import openai
from .schema import RetryConfig

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def openai_retry(
    retry_config: RetryConfig | None = None,
) -> Callable[[F], F]:
    """
    OpenAI API呼び出し用のリトライデコレータ

    Args:
        retry_config: リトライ設定（RetryConfigオブジェクト）

    Returns:
        デコレータ関数
    """
    config = retry_config or RetryConfig()

    return retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
            )
        ),
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential(
            multiplier=config.initial_wait,
            max=config.max_wait,
            exp_base=config.multiplier,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
