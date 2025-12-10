"""
Token budget utilities for estimating how many tokens are left for user content.
"""

import logging
from typing import Any

from llm_applications_library.llm.generators.schema import Model
from .token_utils import estimate_prompt_tokens

logger = logging.getLogger(__name__)

# Known context window sizes (tokens)
MODEL_CONTEXT_WINDOWS = {
    Model.GPT_4O.value: 128_000,
    Model.GPT_4O_MINI.value: 128_000,
    Model.GPT_4_1.value: 128_000,
    Model.GPT_4_1_MINI.value: 128_000,
    Model.GPT_4_1_NANO.value: 8_000,
    Model.GPT_5.value: 200_000,
    Model.GPT_5_MINI.value: 200_000,
    Model.GPT_5_1.value: 200_000,
    Model.GPT_5_1_MINI.value: 200_000,
    Model.GPT_3_5_TURBO.value: 16_000,
    Model.CLAUDE_SONNET_4_5.value: 200_000,
    Model.CLAUDE_OPUS_4_1.value: 200_000,
    Model.CLAUDE_OPUS_4.value: 200_000,
    Model.CLAUDE_SONNET_4.value: 200_000,
    Model.CLAUDE_3_7_SONNET.value: 200_000,
    Model.CLAUDE_3_5_SONNET.value: 200_000,
    Model.CLAUDE_3_5_HAIKU.value: 200_000,
    Model.CLAUDE_3_OPUS.value: 200_000,
    Model.CLAUDE_3_HAIKU.value: 200_000,
}

DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_RESPONSE_TOKENS = 4000
MIN_AVAILABLE_TOKENS = 500


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
    model: Model,
    generation_config: Any | None = None,
    context_windows: dict[Model, int] | None = None,
) -> int:
    """
    推定可能な available tokens を計算する共通ユーティリティ。

    Args:
        prompt_template: プレースホルダーを含むプロンプトテンプレート
        variables: テンプレートに展開する変数
        model: 使用モデル名（Model）
        generation_config: max_output_tokens / max_completion_tokens を含む設定
        context_windows: 追加・上書き用のコンテキストウィンドウ定義

    Returns:
        additional_information などユーザー入力に利用可能なトークン数
    """
    model_name = model.value
    normalized_windows = {k.value: v for k, v in (context_windows or {}).items()}
    model_windows = {**MODEL_CONTEXT_WINDOWS, **normalized_windows}

    try:
        base_prompt_tokens = estimate_prompt_tokens(
            prompt_template or "", variables, model_name
        )
    except Exception as e:
        logger.warning(f"トークン数推定でエラー: {e}, デフォルト値を使用")
        base_prompt_tokens = 2000

    context_window_size = model_windows.get(model_name, DEFAULT_CONTEXT_WINDOW)
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
