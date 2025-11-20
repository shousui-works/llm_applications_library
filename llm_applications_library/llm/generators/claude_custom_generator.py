"""Claude API client for document analysis."""

import base64
import logging
import os
from typing import Any

from ..generators.schema import (
    RetryConfig,
    ClaudeGenerationConfig,
    GeneratorResponse,
)
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
    ) -> GeneratorResponse:
        """
        retry機能付きでテキスト生成を実行

        Args:
            prompt: テキスト生成用のプロンプト
            system_prompt: システムプロンプト（実行時設定）
            generation_kwargs: 生成用の追加パラメータ

        Returns:
            TextGeneratorResponse: 統一されたテキスト生成レスポンス
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

            return content, usage

        try:
            logger.debug(
                f"Executing Claude generation with retry config: "
                f"max_attempts={self.retry_config.max_attempts}, "
                f"initial_wait={self.retry_config.initial_wait}"
            )
            content, usage = _run_with_retry()
            return GeneratorResponse.create_success(content=content, usage=usage)
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
                return GeneratorResponse.create_error(error=str(e))


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
        temperature: float | None = None,
        max_tokens: int = 4096,
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
            }

            # Set temperature if provided, otherwise use default
            if temperature is not None:
                kwargs["temperature"] = temperature
            else:
                kwargs["temperature"] = 0.1

            if top_k is not None:
                kwargs["top_k"] = top_k

            if stop_sequences is not None:
                kwargs["stop_sequences"] = stop_sequences

            if system_prompt:
                kwargs["system"] = system_prompt

            # Debug: Log request details
            logger.debug(f"Claude API request kwargs: {kwargs}")
            # Log message content types safely
            try:
                first_message = kwargs["messages"][0]
                if isinstance(first_message.get("content"), list):
                    content_types = [
                        item.get("type", "unknown") for item in first_message["content"]
                    ]
                    logger.debug(f"Message content types: {content_types}")
                else:
                    logger.debug(
                        f"Message content is string: {type(first_message.get('content'))}"
                    )
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Could not log message content types: {e}")

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
            # Handle all errors with generic error handling
            error_msg = str(e)
            logger.error(f"Claude Vision API error: {error_msg}")

            # Check error message content for specific error types
            error_str = str(e).lower()
            if "bad request" in error_str or "400" in error_str:
                error_msg = f"Bad Request (400): {str(e)}"
                logger.error(f"Claude Vision API Bad Request: {error_msg}")

                # Common 400 error causes for Vision API
                if "maximum context length" in error_str or "too long" in error_str:
                    error_msg += " | Likely cause: Prompt too long. Consider reducing prompt length."
                elif "invalid" in error_str and "image" in error_str:
                    error_msg += " | Likely cause: Invalid image format or data."

            # Try to extract additional error details if available
            detailed_error = error_msg
            if hasattr(e, "response"):
                try:
                    response = e.response  # type: ignore
                    status_code = getattr(response, "status_code", "unknown")
                    if hasattr(response, "text"):
                        error_body = response.text
                        logger.error(f"Response body: {error_body}")
                        detailed_error += (
                            f" | Status: {status_code} | Body: {error_body}"
                        )
                except Exception as extract_error:
                    logger.error(f"Error extracting response details: {extract_error}")

            return {
                "success": False,
                "content": None,
                "usage": None,
                "error": detailed_error,
            }

    def run(
        self,
        base64_images: list[str],
        mime_types: list[str],
        prompt: str = "これらの画像を詳細に分析してください。",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """Claude Vision APIを使用して画像を解析する（単一または複数画像対応）

        Args:
            base64_images: Base64エンコードされた画像データのリスト
            mime_types: 画像のMIMEタイプのリスト（例: ["image/jpeg", "image/png"]）
            prompt: 画像に対する分析指示（デフォルト: "これらの画像を詳細に分析してください。"）
            system_prompt: システムプロンプト（オプション）
            generation_kwargs: 生成用パラメータ（temperature, max_tokens等）

        Returns:
            GeneratorResponse: 統一されたVision分析レスポンス

        Note:
            単一画像の場合: base64_images=["image_data"], mime_types=["image/jpeg"]
            複数画像の場合: base64_images=["img1", "img2", ...], mime_types=["image/jpeg", "image/png", ...]
            リストの長さは一致している必要があります
        """

        # Validate input lengths
        if len(base64_images) != len(mime_types):
            raise ValueError("Length of base64_images and mime_types must match")

        if not base64_images:
            raise ValueError("At least one image must be provided")

        # Validate each image
        valid_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        import base64 as b64_module

        for i, (base64_image, mime_type) in enumerate(zip(base64_images, mime_types)):
            # Validate inputs
            if not base64_image or not base64_image.strip():
                raise ValueError(f"base64_image {i} cannot be empty")

            if not mime_type:
                raise ValueError(f"mime_type {i} cannot be empty")

            # Validate base64 data
            try:
                # Test if it's valid base64
                b64_module.b64decode(base64_image, validate=True)
            except Exception as e:
                raise ValueError(f"Invalid base64 image data for image {i}: {e}")

            # Ensure mime_type is valid for Claude
            if mime_type not in valid_mime_types:
                logger.warning(
                    f"Potentially unsupported mime_type for image {i}: {mime_type}. Supported: {valid_mime_types}"
                )

        # Check prompt length and warn if too long
        if len(prompt) > 10000:  # Warn for very long prompts
            logger.warning(
                f"Very long prompt detected: {len(prompt)} characters. This may cause API errors."
            )

        # Limit prompt length to prevent API errors
        MAX_PROMPT_LENGTH = 50000  # Conservative limit
        if len(prompt) > MAX_PROMPT_LENGTH:
            truncated_prompt = (
                prompt[:MAX_PROMPT_LENGTH] + "...[truncated due to length]"
            )
            logger.warning(
                f"Prompt truncated from {len(prompt)} to {len(truncated_prompt)} characters"
            )
            prompt = truncated_prompt

        # Claude APIの画像メッセージ形式（複数画像とテキスト対応）
        content = []

        # Add all images
        for base64_image, mime_type in zip(base64_images, mime_types):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )

        # Add text prompt
        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Debug logging
        logger.debug(
            f"Claude Vision API request - {len(base64_images)} images, prompt length: {len(prompt)}, mime_types: {mime_types}"
        )
        logger.debug(f"Messages structure: {len(messages[0]['content'])} content items")

        # Use retry_config from constructor
        retry_config_to_use = self.retry_config

        # Validate and set generation parameters
        if generation_kwargs:
            validated_config = ClaudeGenerationConfig.model_validate(generation_kwargs)
            generation_params = validated_config.model_dump(exclude_none=True)
        else:
            generation_params = {}

        # Adjust max_tokens based on prompt length for very long prompts
        if "max_tokens" not in generation_params:
            if len(prompt) > 20000:
                # For very long prompts, use larger max_tokens but within limits
                generation_params["max_tokens"] = 8192  # Claude 3.5 Sonnet max
                logger.debug(
                    f"Increased max_tokens to {generation_params['max_tokens']} for long prompt"
                )
            else:
                generation_params["max_tokens"] = 4096

        response = self._chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            retry_config=retry_config_to_use,
            **generation_params,
        )

        # 新しい共通クラスで返り値を統一
        if response["success"]:
            return GeneratorResponse.create_success(
                content=response["content"], usage=response["usage"]
            )
        else:
            return GeneratorResponse.create_error(
                error=response["error"], usage=response["usage"]
            )

    def run_from_file(
        self,
        image_paths: list[str],
        prompt: str = "この画像を詳細に分析してください。",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """ファイルパスから画像を読み込んでClaude Vision APIで解析（単一または複数画像対応）

        Args:
            image_paths: 画像ファイルのパスのリスト
            prompt: 画像に対する分析指示（デフォルト: "この画像を詳細に分析してください。"）
            system_prompt: システムプロンプト（オプション）
            generation_kwargs: 生成用パラメータ（temperature, max_tokens等）

        Returns:
            GeneratorResponse: 統一されたVision分析レスポンス
        """
        import mimetypes

        if not image_paths:
            raise ValueError("At least one image path must be provided")

        base64_images = []
        mime_types = []

        # 各ファイルを処理
        for image_path in image_paths:
            # ファイルの存在確認
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # MIMEタイプの推定
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                raise ValueError(
                    f"Unsupported file type: {mime_type} for file: {image_path}"
                )

            # 画像ファイルをBase64エンコード
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            base64_images.append(base64_image)
            mime_types.append(mime_type)

        return self.run(
            base64_images=base64_images,
            mime_types=mime_types,
            prompt=prompt,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
        )
