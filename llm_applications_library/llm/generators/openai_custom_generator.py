"""OpenAI API client for document analysis."""

import logging
import os
from typing import Any
import openai
from .schema import GPTConfig, RetryConfig

from .retry_util import openai_retry


logger = logging.getLogger(__name__)


class RetryOpenAIGenerator:
    """
    Retry機能付きのOpenAIGeneratorコンポーネント

    tenacityベースのretry機能を追加。
    OpenAI APIの一時的なエラー（レート制限、タイムアウト等）に対して
    指数バックオフでリトライを実行する。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        retry_config: RetryConfig | None = None,
    ):
        """
        RetryOpenAIGeneratorを初期化

        Args:
            api_key: OpenAI API key
            model: OpenAI model name
            retry_config: リトライ設定（RetryConfigオブジェクト）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.retry_config = retry_config or RetryConfig()

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        retry機能付きでテキスト生成を実行

        Args:
            prompt: テキスト生成用のプロンプト
            system_prompt: システムプロンプト（実行時設定）
            generation_kwargs: 生成用の追加パラメータ

        Returns:
            生成されたレスポンスのリストとメタデータのリストを含む辞書
        """

        @openai_retry(self.retry_config)
        def _run_with_retry():
            """retry機能付きのrun実行"""
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {"model": self.model, "messages": messages}
            if generation_kwargs:
                kwargs.update(generation_kwargs)

            response = client.chat.completions.create(**kwargs)
            return {
                "replies": [response.choices[0].message.content],
                "meta": [response.usage.model_dump() if response.usage else {}],
            }

        try:
            logger.debug(
                f"Executing OpenAI generation with retry config: "
                f"max_attempts={self.retry_config.max_attempts}, "
                f"initial_wait={self.retry_config.initial_wait}"
            )
            return _run_with_retry()
        except Exception as e:
            logger.error(f"OpenAI generation failed after retries: {e}")
            # エラー時のフォールバック応答
            return {
                "replies": [],
                "meta": [
                    {"error": str(e), "retry_config": self.retry_config.model_dump()}
                ],
            }


class OpenAIVisionGenerator:
    def __init__(
        self, model, api_key: str | None = None, retry_config: RetryConfig | None = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.retry_config = retry_config or RetryConfig()

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

    def run(
        self,
        base64_image: str,
        mime_type: str,
        model_config: GPTConfig,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """OpenAI Vision APIを使用して画像またはPDFを解析する

        Args:
            base64_image (str): Base64エンコードされた画像データ
            mime_type (str): 画像のMIMEタイプ
            model_config (GPTConfig): モデル設定
            system_prompt (str, optional): 分析指示プロンプト
            generation_kwargs (dict, optional): 生成用の追加パラメータ

        Returns:
            dict[str, Any]: レスポンス辞書
        """

        # Build messages with optional system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ],
        })

        # Use retry_config from constructor if not provided in model_config
        retry_config_to_use = (
            model_config.retry_config
            if hasattr(model_config, "retry_config")
            else self.retry_config
        )

        # Merge generation_kwargs with model config
        generation_params = model_config.generation_config.model_dump()
        if generation_kwargs:
            generation_params.update(generation_kwargs)

        return {
            "replies": self._chat_completion(
                messages=messages,
                api_key=self.api_key,
                retry_config=retry_config_to_use,
                **generation_params,
            )
        }
