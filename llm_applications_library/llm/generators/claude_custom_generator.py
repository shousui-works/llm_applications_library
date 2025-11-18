"""Claude API client for document analysis."""

import base64
import logging
import os
from typing import Any

from ..generators.schema import RetryConfig, ClaudeGenerationConfig
from ...utilities.claude_retry import claude_retry

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None


class RetryClaudeGenerator:
    """
    Retry機能付きのClaudeGeneratorコンポーネント

    tenacityベースのretry機能を追加。
    Claude APIの一時的なエラー（レート制限、タイムアウト等）に対して
    指数バックオフでリトライを実行する。
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",  # Default to verified working model
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        RetryClaudeGeneratorを初期化

        Args:
            model: Claude model name
            api_key: Anthropic API key
            retry_config: リトライ設定（RetryConfigオブジェクト）
        """
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for Claude functionality. "
                "Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        if not self.api_key:
            raise ValueError(
                "API key is required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

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

        @claude_retry(
            max_attempts=self.retry_config.max_attempts,
            initial_wait=self.retry_config.initial_wait,
            max_wait=self.retry_config.max_wait,
            multiplier=self.retry_config.multiplier,
        )
        def _run_with_retry():
            """retry機能付きのrun実行"""
            client = anthropic.Anthropic(api_key=self.api_key)

            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if generation_kwargs:
                # Validate using Pydantic model directly
                validated_config = ClaudeGenerationConfig.model_validate(
                    generation_kwargs
                )
                kwargs.update(validated_config.model_dump(exclude_none=True))

            response = client.messages.create(**kwargs)

            # Claude APIのレスポンス構造に基づく
            content = response.content[0].text if response.content else ""

            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
            }

            return {
                "replies": [content],
                "meta": [usage],
            }

        try:
            logger.debug(
                f"Executing Claude generation with retry config: "
                f"max_attempts={self.retry_config.max_attempts}, "
                f"initial_wait={self.retry_config.initial_wait}"
            )
            return _run_with_retry()
        except Exception as e:
            # Validation errors should be raised immediately (not retried)
            if "validation error" in str(e) or "Extra inputs are not permitted" in str(
                e
            ):
                logger.error(f"Claude parameter validation failed: {e}")
                raise
            else:
                logger.error(f"Claude generation failed after retries: {e}")
                # エラー時のフォールバック応答
                return {
                    "replies": [],
                    "meta": [
                        {
                            "error": str(e),
                            "retry_config": self.retry_config.model_dump(),
                        }
                    ],
                }


class ClaudeVisionGenerator:
    """Claude Vision APIを使用した画像解析ジェネレータ"""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",  # Default to verified working model
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        ClaudeVisionGeneratorを初期化

        Args:
            model: Claude model name (Vision対応モデルのみ)
            api_key: Anthropic API key
            retry_config: Retry configuration
        """
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for Claude functionality. "
                "Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        if not self.api_key:
            raise ValueError(
                "API key is required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

    def _chat_completion(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        retry_config: RetryConfig | None = None,
    ) -> dict[str, Any]:
        """Direct chat completion with tenacity retry for robust error handling"""

        @claude_retry(
            max_attempts=retry_config.max_attempts if retry_config else 3,
            initial_wait=retry_config.initial_wait if retry_config else 1.0,
            max_wait=retry_config.max_wait if retry_config else 60.0,
            multiplier=retry_config.multiplier if retry_config else 2.0,
        )
        def _make_api_call():
            client = anthropic.Anthropic(api_key=self.api_key)

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
            }

            if top_k is not None:
                kwargs["top_k"] = top_k

            if stop_sequences is not None:
                kwargs["stop_sequences"] = stop_sequences

            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)

            content = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
            }

            return {
                "success": True,
                "content": content,
                "usage": usage,
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
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Claude Vision APIを使用して画像を解析する

        Args:
            base64_image: Base64エンコードされた画像データ
            mime_type: 画像のMIMEタイプ（例: "image/jpeg", "image/png"）
            system_prompt: 分析指示プロンプト（オプション）
            generation_kwargs: 生成用パラメータ（temperature, max_tokens等）

        Returns:
            dict[str, Any]: レスポンス辞書
        """

        # Claude APIの画像メッセージ形式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_image,
                        },
                    },
                ],
            }
        ]

        # Use retry_config from constructor
        retry_config_to_use = self.retry_config

        # Validate and set default generation parameters
        if generation_kwargs:
            validated_config = ClaudeGenerationConfig.model_validate(generation_kwargs)
            generation_params = validated_config.model_dump(exclude_none=True)
        else:
            generation_params = {}

        # Set sensible defaults if not specified
        if "temperature" not in generation_params:
            generation_params["temperature"] = 0.1
        if "max_tokens" not in generation_params:
            generation_params["max_tokens"] = 4096

        response = self._chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            retry_config=retry_config_to_use,
            **generation_params,
        )

        return {"replies": [response]}

    def run_from_file(
        self,
        image_path: str,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """ファイルパスから画像を読み込んでClaude Vision APIで解析

        Args:
            image_path: 画像ファイルのパス
            system_prompt: 分析指示プロンプト（オプション）
            generation_kwargs: 生成用パラメータ（temperature, max_tokens等）

        Returns:
            dict[str, Any]: レスポンス辞書
        """
        import mimetypes

        # ファイルの存在確認
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # MIMEタイプの推定
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported file type: {mime_type}")

        # 画像ファイルをBase64エンコード
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        return self.run(
            base64_image=base64_image,
            mime_type=mime_type,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
        )
