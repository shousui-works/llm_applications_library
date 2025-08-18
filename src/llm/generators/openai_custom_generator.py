"""OpenAI API client for document analysis."""

import logging
import os
from typing import Any, Callable, Union, Awaitable
import openai
from haystack import component
from llm.generators.schema import GPTConfig, RetryConfig

from llm.generators.retry_util import openai_retry

from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses.streaming_chunk import StreamingChunk
from haystack.utils.auth import Secret


logger = logging.getLogger(__name__)


@component
class RetryOpenAIGenerator(OpenAIGenerator):
    """
    Retry機能付きのOpenAIGeneratorコンポーネント

    HaystackのOpenAIGeneratorを継承し、tenacityベースのretry機能を追加。
    OpenAI APIの一時的なエラー（レート制限、タイムアウト等）に対して
    指数バックオフでリトライを実行する。
    """

    def __init__(
        self,
        api_key: Secret | None = None,
        model: str = "gpt-4o-mini",
        streaming_callback: Union[
            Callable[[StreamingChunk], None],
            Callable[[StreamingChunk], Awaitable[None]],
            None,
        ] = None,
        api_base_url: str | None = None,
        organization: str | None = None,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        RetryOpenAIGeneratorを初期化

        Args:
            retry_config: リトライ設定（RetryConfigオブジェクト）
            その他のパラメータはOpenAIGeneratorと同じ
        """
        # 親クラスの初期化
        super().__init__(
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=organization,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

        # リトライ設定の保存
        self.retry_config = retry_config or RetryConfig()

    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: Union[
            Callable[[StreamingChunk], None],
            Callable[[StreamingChunk], Awaitable[None]],
            None,
        ] = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        retry機能付きでテキスト生成を実行

        Args:
            prompt: テキスト生成用のプロンプト
            system_prompt: システムプロンプト（実行時設定）
            streaming_callback: ストリーミングコールバック
            generation_kwargs: 生成用の追加パラメータ

        Returns:
            生成されたレスポンスのリストとメタデータのリストを含む辞書
        """

        @openai_retry(self.retry_config)
        def _run_with_retry():
            """retry機能付きのrun実行"""
            return super(RetryOpenAIGenerator, self).run(
                prompt=prompt,
                system_prompt=system_prompt,
                streaming_callback=streaming_callback,
                generation_kwargs=generation_kwargs,
            )

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
