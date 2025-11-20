"""OpenAI API client for document analysis."""

import logging
import os
from typing import Any
import openai
from .schema import (
    RetryConfig,
    OpenAIGenerationConfig,
    GeneratorResponse,
)

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
                # Validate using Pydantic model directly
                validated_config = OpenAIGenerationConfig.model_validate(
                    generation_kwargs
                )
                config_dict = validated_config.model_dump(exclude_none=True)

                # Handle max_tokens vs max_completion_tokens compatibility
                # For GPT-5 models, always convert max_tokens to max_completion_tokens
                if self.model.startswith("gpt-5") or "gpt-5" in self.model.lower():
                    if "max_tokens" in config_dict:
                        if "max_completion_tokens" not in config_dict:
                            # Convert max_tokens to max_completion_tokens
                            config_dict["max_completion_tokens"] = config_dict["max_tokens"]
                        # Remove max_tokens for GPT-5 models
                        config_dict.pop("max_tokens")

                kwargs.update(config_dict)

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}
            return content, usage

        try:
            logger.debug(
                f"Executing OpenAI generation with retry config: "
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
                logger.error(f"OpenAI parameter validation failed: {e}")
                raise
            else:
                logger.error(f"OpenAI generation failed after retries: {e}")
                # エラー時のフォールバック応答
                return GeneratorResponse.create_error(error=str(e))


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
                # Use max_completion_tokens for GPT-5 models, max_tokens for others
                if self.model.startswith("gpt-5") or "gpt-5" in self.model.lower():
                    kwargs["max_completion_tokens"] = max_tokens
                else:
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
        prompt: str = "Please analyze this image in detail.",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """OpenAI Vision APIを使用して画像またはPDFを解析する

        Args:
            base64_image (str): Base64エンコードされた画像データ
            mime_type (str): 画像のMIMEタイプ
            prompt (str): 画像に対する分析指示（デフォルト: "Please analyze this image in detail."）
            system_prompt (str, optional): システムプロンプト
            generation_kwargs (dict, optional): 生成用パラメータ（temperature, max_tokens等）

        Returns:
            VisionGeneratorResponse: 統一されたVision分析レスポンス
        """

        # Build messages with optional system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        )

        # Use retry_config from constructor
        retry_config_to_use = self.retry_config

        # Validate and set default generation parameters
        if generation_kwargs:
            validated_config = OpenAIGenerationConfig.model_validate(generation_kwargs)
            generation_params = validated_config.model_dump(exclude_none=True)

            # For GPT-5 models, convert max_tokens to max_completion_tokens
            if self.model.startswith("gpt-5") or "gpt-5" in self.model.lower():
                if "max_tokens" in generation_params:
                    if "max_completion_tokens" not in generation_params:
                        generation_params["max_completion_tokens"] = generation_params["max_tokens"]
                    generation_params.pop("max_tokens")
        else:
            generation_params = {}

        # Set sensible defaults if not specified
        if "temperature" not in generation_params:
            generation_params["temperature"] = 0.1
        if "max_tokens" not in generation_params and "max_completion_tokens" not in generation_params:
            # Use max_completion_tokens for GPT-5 models, max_tokens for others
            if self.model.startswith("gpt-5") or "gpt-5" in self.model.lower():
                generation_params["max_completion_tokens"] = 4096
            else:
                generation_params["max_tokens"] = 4096

        response = self._chat_completion(
            messages=messages,
            api_key=self.api_key,
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
        image_path: str,
        prompt: str = "Please analyze this image in detail.",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """ファイルパスから画像を読み込んでOpenAI Vision APIで解析

        Args:
            image_path: 画像ファイルのパス
            prompt: 画像に対する分析指示（デフォルト: "Please analyze this image in detail."）
            system_prompt: システムプロンプト（オプション）
            generation_kwargs: 生成用パラメータ（temperature, max_tokens等）

        Returns:
            VisionGeneratorResponse: 統一されたVision分析レスポンス
        """
        import base64
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
            prompt=prompt,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
        )
