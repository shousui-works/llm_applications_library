"""OpenAI API client for document analysis."""

import logging
import os
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
from haystack import component
from llm.generators.schema import GPTConfig, RetryConfig

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


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


@component
class OpenAIVisionGenerator:
    def __init__(self, model, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> dict[str, Any]:
        """Direct chat completion with tenacity retry for robust error handling"""

        @openai_retry(retry_config)
        def _make_api_call():
            client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                max_retries=0,
                timeout=1800,
            )

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "usage": response.usage.model_dump() if response.usage else None,
                "error": None,
            }

        try:
            return _make_api_call()
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "usage": None,
                "error": str(e),
            }

    @component.output_types(replies=dict[str, Any])
    def run(
        self,
        base64_image: str,
        mime_type: str,
        prompt: str,
        model_config: GPTConfig,
    ) -> dict[str, Any]:
        """OpenAI Vision APIを使用して画像またはPDFを解析する

        Args:
            image_path (str): 解析する画像ファイルまたはPDFファイルのパス
            system_prompt (str): システムプロンプト
            user_prompt_template (str): ユーザープロンプトテンプレート
            model_config (GPTConfig): モデル設定
            **kargs: テンプレートに渡す変数

        Returns:
            dict[str, Any]: レスポンス辞書
        """

        # PDFファイルの場合
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            },
        ]

        return {
            "replies": self._chat_completion(
                messages=messages,
                api_key=self.api_key,
                retry_config=model_config.retry_config,
                **model_config.generation_config.model_dump(),
            )
        }
