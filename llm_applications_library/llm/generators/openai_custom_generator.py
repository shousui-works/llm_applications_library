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

        def _contains_web_search_tool(tools: list[dict] | None) -> bool:
            """Detect whether the tools array includes the web_search tool."""
            if not tools:
                return False
            return any(
                isinstance(tool, dict) and tool.get("type") == "web_search"
                for tool in tools
            )

        def _normalize_usage(usage_obj: Any) -> dict[str, Any]:
            """Handle both dict and OpenAI object usages."""
            if not usage_obj:
                return {}
            if hasattr(usage_obj, "model_dump"):
                return usage_obj.model_dump()
            if isinstance(usage_obj, dict):
                return usage_obj
            return {}

        def _convert_content_for_responses(content: Any) -> list[dict]:
            """
            Responses API expects typed content entries (e.g., input_text).
            Convert plain strings or simple text objects into the expected shape.
            """
            if isinstance(content, str):
                return [{"type": "input_text", "text": content}]

            if isinstance(content, list):
                converted = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            converted.append(
                                {"type": "input_text", "text": item.get("text", "")}
                            )
                        else:
                            converted.append(item)
                    else:
                        converted.append({"type": "input_text", "text": str(item)})
                return converted

            # Fallback for unexpected shapes
            return [{"type": "input_text", "text": str(content)}]

        def _build_responses_kwargs(
            messages: list[dict], base_config: dict[str, Any]
        ) -> dict[str, Any]:
            responses_kwargs = {
                "model": self.model,
                "input": [
                    {
                        "role": msg["role"],
                        "content": _convert_content_for_responses(msg["content"]),
                    }
                    for msg in messages
                ],
            }

            config_copy = base_config.copy()
            max_output_tokens = config_copy.pop("max_completion_tokens", None)
            if max_output_tokens is None:
                max_output_tokens = config_copy.pop("max_tokens", None)
            if max_output_tokens is not None:
                responses_kwargs["max_output_tokens"] = max_output_tokens

            responses_kwargs.update(config_copy)
            return responses_kwargs

        def _extract_response_text(response: Any) -> str | None:
            """Pull the primary text output from Responses API result."""
            if response is None:
                return None

            if hasattr(response, "output_text"):
                text = response.output_text
                if text:
                    return text

            output = getattr(response, "output", None)
            try:
                if output:
                    first_item = output[0]
                    content = getattr(first_item, "content", None) or first_item.get(
                        "content"
                    )
                    if content:
                        first_content = content[0]
                        if hasattr(first_content, "text"):
                            return first_content.text
                        if isinstance(first_content, dict) and "text" in first_content:
                            return first_content["text"]
                        return str(first_content)
            except Exception:
                # If the response structure is unexpected, fall back later
                pass

            return None

        @openai_retry(self.retry_config)
        def _run_with_retry():
            """retry機能付きのrun実行"""
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {"model": self.model, "messages": messages}
            config_dict: dict[str, Any] = {}
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
                            config_dict["max_completion_tokens"] = config_dict[
                                "max_tokens"
                            ]
                    # Remove max_tokens for GPT-5 models
                    config_dict.pop("max_tokens")

            use_responses_api = _contains_web_search_tool(config_dict.get("tools"))

            if use_responses_api:
                responses_kwargs = _build_responses_kwargs(messages, config_dict)
                response = client.responses.create(**responses_kwargs)
                content = _extract_response_text(response) or ""
                usage = _normalize_usage(getattr(response, "usage", None))
            else:
                kwargs.update(config_dict)
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                usage = _normalize_usage(getattr(response, "usage", None))

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
        max_completion_tokens: int | None = None,
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

            # Handle token limits - prefer max_completion_tokens if provided
            if max_completion_tokens:
                kwargs["max_completion_tokens"] = max_completion_tokens
            elif max_tokens:
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
            generation_kwargs (dict, optional): 生成用パラメータ（temperature, max_tokens等）

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
            messages.append({"role": "system", "content": system_prompt})

        # Build content array with multiple images and text prompt
        content = []

        # Add all images
        for base64_image, mime_type in zip(base64_images, mime_types):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        # Add text prompt
        content.append(
            {
                "type": "text",
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

            # For GPT-5 models, convert max_tokens to max_completion_tokens
            if self.model.startswith("gpt-5") or "gpt-5" in self.model.lower():
                if "max_tokens" in generation_params:
                    if "max_completion_tokens" not in generation_params:
                        generation_params["max_completion_tokens"] = generation_params[
                            "max_tokens"
                        ]
                    generation_params.pop("max_tokens")
        else:
            generation_params = {}

        # Set sensible defaults if not specified
        if "temperature" not in generation_params:
            generation_params["temperature"] = 0.1
        if (
            "max_tokens" not in generation_params
            and "max_completion_tokens" not in generation_params
        ):
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
            generation_kwargs: 生成用パラメータ（temperature, max_tokens等）

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
