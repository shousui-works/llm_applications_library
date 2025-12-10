"""
Token budget utilities for estimating how many tokens are left for user content.
"""

import logging
from typing import Any

from .token_utils import estimate_prompt_tokens

logger = logging.getLogger(__name__)

# Known context window sizes (tokens)
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 128_000,
    "gpt-4.1-mini": 128_000,
    "gpt-4.1-nano": 8_000,
    "gpt-3.5-turbo": 16_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-4-sonnet": 200_000,
    "claude-4-opus": 200_000,
    "claude-4-haiku": 200_000,
}

DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_RESPONSE_TOKENS = 4000
MIN_AVAILABLE_TOKENS = 500


def _normalize_model_for_encoding(model: str) -> str:
    """Map unsupported models to the nearest tokenizer-compatible name."""
    lower = model.lower()
    if "gpt-5.1" in lower:
        return "gpt-4o"

    if "claude" in lower:
        if "sonnet-4" in lower or "claude-4" in lower:
            return "claude-4-sonnet"
        if "sonnet" in lower:
            return "claude-3-sonnet"
        if "opus" in lower:
            return "claude-4-opus" if "claude-4" in lower else "claude-3-opus"
        if "haiku" in lower:
            return "claude-4-haiku" if "claude-4" in lower else "claude-3-haiku"

    return model


def _coerce_response_tokens(generation_config: Any | None) -> int:
    """Extract response token budget from a generation config-like object."""
    if not generation_config:
        return DEFAULT_RESPONSE_TOKENS

    # Support both attrs and dict-like configs
    for key in ("max_completion_tokens", "max_output_tokens"):
        if isinstance(generation_config, dict) and generation_config.get(key):
            return int(generation_config[key])
        if hasattr(generation_config, key):
            value = getattr(generation_config, key)
            if value:
                return int(value)

    return DEFAULT_RESPONSE_TOKENS


def calculate_available_tokens(
    prompt_template: str,
    variables: dict[str, str],
    model: str,
    generation_config: Any | None = None,
    context_windows: dict[str, int] | None = None,
) -> int:
    """
    推定可能な available tokens を計算する共通ユーティリティ。

    Args:
        prompt_template: プレースホルダーを含むプロンプトテンプレート
        variables: テンプレートに展開する変数
        model: 使用モデル名
        generation_config: max_output_tokens / max_completion_tokens を含む設定
        context_windows: 追加・上書き用のコンテキストウィンドウ定義

    Returns:
        additional_information などユーザー入力に利用可能なトークン数
    """
    normalized_model = _normalize_model_for_encoding(model)
    model_windows = {**MODEL_CONTEXT_WINDOWS, **(context_windows or {})}

    try:
        base_prompt_tokens = estimate_prompt_tokens(
            prompt_template or "", variables, normalized_model
        )
    except Exception as e:
        logger.warning(f"トークン数推定でエラー: {e}, デフォルト値を使用")
        base_prompt_tokens = 2000

    context_window_size = model_windows.get(model, DEFAULT_CONTEXT_WINDOW)
    response_tokens = _coerce_response_tokens(generation_config)

    available_tokens = context_window_size - base_prompt_tokens - response_tokens
    if available_tokens < MIN_AVAILABLE_TOKENS:
        logger.warning(
            f"利用可能トークン数が不足: {available_tokens}, "
            f"最低限の{MIN_AVAILABLE_TOKENS}トークンに設定 "
            f"(context_window={context_window_size}, "
            f"base_prompt={base_prompt_tokens}, response={response_tokens})"
        )
        available_tokens = MIN_AVAILABLE_TOKENS

    return available_tokens
