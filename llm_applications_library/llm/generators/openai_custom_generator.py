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


def _extract_responses_content(response: Any) -> str | None:
    """Safely extract text content from Responses API response."""
    try:
        if hasattr(response, "output_text"):
            content = response.output_text
            if isinstance(content, list):
                return "".join(str(part) for part in content)
            return content

        if hasattr(response, "output") and response.output:
            first_output = response.output[0]
            if hasattr(first_output, "content") and first_output.content:
                first_content = first_output.content[0]
                text_value = getattr(first_content, "text", None)
                if text_value is not None:
                    return text_value
    except Exception as e:
        logger.debug(f"Failed to extract Responses API content: {e}")
    return None


def _extract_usage(usage_obj: Any) -> dict[str, Any]:
    """Normalize usage information from OpenAI SDK objects."""
    if not usage_obj:
        return {}
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    if isinstance(usage_obj, dict):
        return usage_obj
    return {}


class RetryOpenAIGenerator:
    """
    Retry機能付きのOpenAIGeneratorコンポーネント

    tenacityベースのretry機能を追加。
    OpenAI APIの一時的なエラー（レート制限、タイムアウト等）に対して
    指数バックオフでリトライを実行する。
    Azure OpenAIもサポート。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        retry_config: RetryConfig | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ):
        """
        RetryOpenAIGeneratorを初期化

        Args:
            api_key: OpenAI API key (Azure使用時はAzure APIキー)
            model: OpenAI model name (Azure使用時はデプロイメント名)
            retry_config: リトライ設定(RetryConfigオブジェクト)
            azure_endpoint: Azure OpenAIエンドポイント(指定時はAzureモード)
            azure_api_version: Azure OpenAI APIバージョン
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = (
            azure_api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2024-12-01-preview"
        )
        self._use_azure = self.azure_endpoint is not None

        if self._use_azure:
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.model = model
        self.retry_config = retry_config or RetryConfig()

    def _create_client(self) -> openai.OpenAI | openai.AzureOpenAI:
        """OpenAIまたはAzure OpenAIクライアントを作成"""
        if self._use_azure and self.azure_endpoint:
            return openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        return openai.OpenAI(api_key=self.api_key)

    def _build_text_input(self, prompt: str, system_prompt: str | None = None):
        """Build Responses API input payload for text-only requests."""
        if system_prompt:
            return [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ]
        return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]

    def _prepare_responses_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Convert generation params to Responses API compatible fields."""
        responses_params = config.copy()

        # Responses API uses max_output_tokens; map legacy aliases to it
        max_output_tokens = responses_params.pop("max_output_tokens", None)
        if "max_completion_tokens" in responses_params:
            max_output_tokens = responses_params.pop("max_completion_tokens")
        if max_output_tokens is None and "max_tokens" in responses_params:
            max_output_tokens = responses_params.pop("max_tokens")
        if max_output_tokens is not None:
            responses_params["max_output_tokens"] = max_output_tokens

        # Streaming is handled via responses.stream, not via a flag on create
        if "stream" in responses_params:
            stream_flag = responses_params.pop("stream")
            if stream_flag:
                logger.warning(
                    "stream=True is not supported in responses.create; ignoring stream flag."
                )

        return responses_params

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
            generation_kwargs: 生成用の追加パラメータ（temperature, max_output_tokens等）

        Returns:
            TextGeneratorResponse: 統一されたテキスト生成レスポンス
        """

        @openai_retry(self.retry_config)
        def _run_with_retry():
            """retry機能付きのrun実行"""
            client = self._create_client()
            input_payload = self._build_text_input(prompt, system_prompt)

            config_dict: dict[str, Any] = {}
            if generation_kwargs:
                # Validate using Pydantic model directly
                validated_config = OpenAIGenerationConfig.model_validate(
                    generation_kwargs
                )
                config_dict = validated_config.model_dump(exclude_none=True)

            responses_params = self._prepare_responses_params(config_dict)
            response = client.responses.create(
                model=self.model,
                input=input_payload,
                **responses_params,
            )
            content = _extract_responses_content(response)
            usage = _extract_usage(response.usage)

            return content, usage

        try:
            logger.debug(
                f"Executing OpenAI generation with retry config: "
                f"max_attempts={self.retry_config.max_attempts}, "
                f"initial_wait={self.retry_config.initial_wait}"
            )
            content, usage = _run_with_retry()
            return GeneratorResponse.create_success(content=content or "", usage=usage)
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
    """Vision Generator with Azure OpenAI support."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ):
        self.model = model
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = (
            azure_api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2024-12-01-preview"
        )
        self._use_azure = self.azure_endpoint is not None

        if self._use_azure:
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.retry_config = retry_config or RetryConfig()

    def _create_client(self) -> openai.OpenAI | openai.AzureOpenAI:
        """OpenAIまたはAzure OpenAIクライアントを作成"""
        if self._use_azure and self.azure_endpoint:
            return openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
                max_retries=0,
                timeout=1800,
            )
        return openai.OpenAI(
            api_key=self.api_key,
            max_retries=0,
            timeout=1800,
        )

    def _chat_completion(
        self,
        messages: list[dict],
        retry_config: RetryConfig | None = None,
        **generation_params: Any,
    ) -> dict[str, Any]:
        """Responses API call with tenacity retry for robust error handling.

        Args:
            messages: Input messages for the API
            retry_config: Retry configuration
            **generation_params: Generation parameters (temperature, max_output_tokens, text, etc.)
        """

        @openai_retry(retry_config)
        def _make_api_call():
            client = self._create_client()

            # Prepare Responses API parameters
            params = generation_params.copy()

            # Handle max_output_tokens / max_completion_tokens / max_tokens
            max_output_tokens = params.pop("max_output_tokens", None)
            max_completion_tokens = params.pop("max_completion_tokens", None)
            max_tokens = params.pop("max_tokens", None)
            token_limit = (
                max_output_tokens
                if max_output_tokens is not None
                else max_completion_tokens
            )
            if token_limit is None:
                token_limit = max_tokens
            if token_limit is not None:
                params["max_output_tokens"] = token_limit

            response = client.responses.create(
                model=self.model,
                input=messages,
                **params,
            )

            return {
                "success": True,
                "content": _extract_responses_content(response),
                "usage": _extract_usage(response.usage),
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
        base64_images: list[str],
        mime_types: list[str],
        prompt: str = "Please analyze these images in detail.",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """OpenAI Vision APIを使用して画像またはPDFを解析する（単一または複数画像対応）

        Args:
            base64_images (list[str]): Base64エンコードされた画像データのリスト
            mime_types (list[str]): 画像のMIMEタイプのリスト
            prompt (str): 画像に対する分析指示（デフォルト: "Please analyze these images in detail."）
            system_prompt (str, optional): システムプロンプト
            generation_kwargs (dict, optional): 生成用パラメータ（temperature, max_output_tokens等）

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

        # Build messages with optional system prompt
        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )

        # Build content array with multiple images and text prompt
        content = []

        # Add all images
        for base64_image, mime_type in zip(base64_images, mime_types):
            content.append(
                {
                    "type": "input_image",
                    # Responses API expects a string URL for images
                    "image_url": f"data:{mime_type};base64,{base64_image}",
                }
            )

        # Add text prompt
        content.append(
            {
                "type": "input_text",
                "text": prompt,
            }
        )

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        # Use retry_config from constructor
        retry_config_to_use = self.retry_config

        # Validate and set default generation parameters
        if generation_kwargs:
            validated_config = OpenAIGenerationConfig.model_validate(generation_kwargs)
            generation_params = validated_config.model_dump(exclude_none=True)

        else:
            generation_params = {}

        # Set sensible defaults if not specified
        # Note: Reasoning models (o1, o3, o3-mini, gpt-5) don't support temperature
        model_lower = self.model.lower()
        is_reasoning_model = (
            model_lower.startswith(("o1", "o3"))
            or "-o1" in model_lower
            or "-o3" in model_lower
            or "gpt-5" in model_lower
        )
        if "temperature" not in generation_params and not is_reasoning_model:
            generation_params["temperature"] = 0.1
        if (
            "max_output_tokens" not in generation_params
            and "max_completion_tokens" not in generation_params
        ):
            generation_params["max_output_tokens"] = 4096

        # Filter to only supported parameters for Responses API
        # See: https://platform.openai.com/docs/api-reference/responses/create
        supported_params = {
            # Basic generation parameters
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_completion_tokens",  # Legacy alias
            "max_tokens",  # Legacy alias
            "stop",
            # Responses API specific
            "text",  # Structured outputs: {"format": {"type": "json_schema", ...}}
            "tools",
            "reasoning",  # {"effort": "high", "summary": "auto"}
            "instructions",
            "metadata",
            "store",
            "include",
            "background",
            "service_tier",
        }
        filtered_params = {}
        excluded_params = []
        for k, v in generation_params.items():
            if k in supported_params:
                filtered_params[k] = v
            else:
                excluded_params.append(k)

        if excluded_params:
            logger.debug(
                f"Filtered out unsupported parameters for Responses API: {excluded_params}"
            )

        response = self._chat_completion(
            messages=messages,
            retry_config=retry_config_to_use,
            **filtered_params,
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
        prompt: str = "Please analyze these images in detail.",
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """ファイルパスから画像を読み込んでOpenAI Vision APIで解析（単一または複数画像対応）

        Args:
            image_paths: 画像ファイルのパスのリスト
            prompt: 画像に対する分析指示（デフォルト: "Please analyze these images in detail."）
            system_prompt: システムプロンプト（オプション）
            generation_kwargs: 生成用パラメータ（temperature, max_output_tokens等）

        Returns:
            GeneratorResponse: 統一されたVision分析レスポンス
        """
        import base64
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
