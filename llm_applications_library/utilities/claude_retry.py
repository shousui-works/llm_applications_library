"""
Claude API用のtenacityベースのリトライユーティリティ
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# 型定義
F = TypeVar("F", bound=Callable[..., Any])

try:
    import anthropic
except ImportError:
    # anthropic未インストール時のモック
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

        class PermissionDeniedError(Exception):
            pass

        class NotFoundError(Exception):
            pass

        class UnprocessableEntityError(Exception):
            pass

    anthropic = MockAnthropic()


def should_retry_claude_error(exception: Exception) -> bool:
    """Claude APIエラーでリトライすべきかどうかを判定"""
    # Claude APIの一時的なエラーのみリトライ
    if isinstance(exception, anthropic.RateLimitError):
        return True
    if isinstance(exception, anthropic.APITimeoutError):
        return True
    if isinstance(exception, anthropic.InternalServerError):
        return True
    if isinstance(exception, anthropic.APIConnectionError):
        return True
    # BadRequestError, AuthenticationError等の恒久的なエラーはリトライしない
    return False


def claude_retry(
    max_attempts: int = 5,
    initial_wait: float = 3.0,
    max_wait: float = 240.0,
    multiplier: float = 5.0,
) -> Callable[[F], F]:
    """
    Claude API呼び出し用のリトライデコレータ

    Args:
        max_attempts: 最大試行回数
        initial_wait: 初期待機時間（秒）
        max_wait: 最大待機時間（秒）
        multiplier: 待機時間の乗数（exponential backoff）

    Returns:
        デコレータ関数
    """
    return retry(
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
            )
        ),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=initial_wait,
            max=max_wait,
            exp_base=multiplier,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def claude_retry_with_config(
    retry_config: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    設定辞書からリトライデコレータを作成

    Args:
        retry_config: リトライ設定の辞書
            - max_attempts: 最大試行回数（デフォルト: 5）
            - initial_wait: 初期待機時間（デフォルト: 1.0）
            - max_wait: 最大待機時間（デフォルト: 60.0）
            - multiplier: 待機時間の乗数（デフォルト: 2.0）

    Returns:
        デコレータ関数
    """
    config = retry_config or {}

    return claude_retry(
        max_attempts=config.get("max_attempts", 5),
        initial_wait=config.get("initial_wait", 1.0),
        max_wait=config.get("max_wait", 60.0),
        multiplier=config.get("multiplier", 2.0),
    )
