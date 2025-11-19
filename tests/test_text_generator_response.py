"""Tests for Text Generator Response classes."""

from llm_applications_library.llm.generators.schema import (
    TextGenerationUsage,
    TextGenerationMeta,
    TextGeneratorResponse,
)


class TestTextGenerationUsage:
    """TextGenerationUsageクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        usage = TextGenerationUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None

    def test_custom_values(self):
        """カスタム値のテスト"""
        usage = TextGenerationUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50

    def test_from_claude_dict(self):
        """Claude形式の辞書からの作成テスト"""
        usage_dict = {"input_tokens": 75, "output_tokens": 25, "total_tokens": 100}
        usage = TextGenerationUsage.model_validate(usage_dict)
        assert usage.input_tokens == 75
        assert usage.output_tokens == 25
        assert usage.total_tokens == 100

    def test_from_openai_dict(self):
        """OpenAI形式の辞書からの作成テスト"""
        usage_dict = {
            "prompt_tokens": 80,
            "completion_tokens": 30,
            "total_tokens": 110,
            "extra_field": "ignored",  # extra fieldsは無視される
        }
        usage = TextGenerationUsage.model_validate(usage_dict)
        assert usage.prompt_tokens == 80
        assert usage.completion_tokens == 30
        # OpenAI形式の場合、input/outputはデフォルト値
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestTextGenerationMeta:
    """TextGenerationMetaクラスのテスト"""

    def test_usage_meta(self):
        """使用統計メタデータのテスト"""
        meta = TextGenerationMeta(input_tokens=100, output_tokens=50, total_tokens=150)
        assert meta.input_tokens == 100
        assert meta.output_tokens == 50
        assert meta.total_tokens == 150
        assert meta.error is None
        assert meta.retry_config is None

    def test_error_meta(self):
        """エラーメタデータのテスト"""
        retry_config = {"max_attempts": 3, "initial_wait": 1.0}
        meta = TextGenerationMeta(error="API Error", retry_config=retry_config)
        assert meta.error == "API Error"
        assert meta.retry_config == retry_config
        assert meta.input_tokens is None
        assert meta.output_tokens is None

    def test_mixed_meta(self):
        """混合メタデータのテスト（OpenAI形式）"""
        meta = TextGenerationMeta(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        assert meta.prompt_tokens == 100
        assert meta.completion_tokens == 50
        assert meta.total_tokens == 150


class TestTextGeneratorResponse:
    """TextGeneratorResponseクラスのテスト"""

    def test_create_success_with_claude_usage(self):
        """Claude形式のusageでの成功レスポンス作成テスト"""
        usage_dict = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        response = TextGeneratorResponse.create_success(
            content="Generated text", usage=usage_dict
        )

        assert len(response.replies) == 1
        assert response.replies[0] == "Generated text"
        assert len(response.meta) == 1

        meta = response.meta[0]
        assert meta.input_tokens == 100
        assert meta.output_tokens == 50
        assert meta.total_tokens == 150
        assert meta.error is None

    def test_create_success_with_openai_usage(self):
        """OpenAI形式のusageでの成功レスポンス作成テスト"""
        usage_dict = {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
        response = TextGeneratorResponse.create_success(
            content="OpenAI response", usage=usage_dict
        )

        assert response.replies[0] == "OpenAI response"
        meta = response.meta[0]
        assert meta.prompt_tokens == 80
        assert meta.completion_tokens == 40
        assert meta.total_tokens == 120

    def test_create_success_no_usage(self):
        """Usage情報なしでの成功レスポンス作成テスト"""
        response = TextGeneratorResponse.create_success(content="Simple response")

        assert response.replies[0] == "Simple response"
        meta = response.meta[0]
        assert meta.input_tokens is None
        assert meta.output_tokens is None
        assert meta.error is None

    def test_create_error(self):
        """エラーレスポンス作成テスト"""
        retry_config = {
            "max_attempts": 3,
            "initial_wait": 1.0,
            "max_wait": 60.0,
            "multiplier": 2.0,
        }
        response = TextGeneratorResponse.create_error(
            error="Rate limit exceeded", retry_config=retry_config
        )

        assert len(response.replies) == 0
        assert len(response.meta) == 1

        meta = response.meta[0]
        assert meta.error == "Rate limit exceeded"
        assert meta.retry_config == retry_config
        assert meta.input_tokens is None

    def test_is_success(self):
        """成功判定テスト"""
        # 成功レスポンス
        success_response = TextGeneratorResponse.create_success("Success text")
        assert success_response.is_success() is True

        # エラーレスポンス
        error_response = TextGeneratorResponse.create_error("Error occurred")
        assert error_response.is_success() is False

        # 空のコンテンツでも成功扱い（空文字列の場合）
        empty_success = TextGeneratorResponse(replies=[""], meta=[TextGenerationMeta()])
        assert empty_success.is_success() is True

    def test_get_content(self):
        """コンテンツ取得テスト"""
        # 成功レスポンス
        success_response = TextGeneratorResponse.create_success("Test content")
        assert success_response.get_content() == "Test content"

        # エラーレスポンス
        error_response = TextGeneratorResponse.create_error("Error")
        assert error_response.get_content() is None

        # 複数のreplies（通常はないが、対応）
        multi_response = TextGeneratorResponse(
            replies=["First", "Second"],
            meta=[TextGenerationMeta(), TextGenerationMeta()],
        )
        assert multi_response.get_content() == "First"

    def test_get_error(self):
        """エラー取得テスト"""
        # 成功レスポンス
        success_response = TextGeneratorResponse.create_success("Success")
        assert success_response.get_error() is None

        # エラーレスポンス
        error_response = TextGeneratorResponse.create_error("API Error")
        assert error_response.get_error() == "API Error"

    def test_get_usage(self):
        """使用統計取得テスト"""
        # Claude形式のusage
        claude_response = TextGeneratorResponse.create_success(
            content="Claude text",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        usage = claude_response.get_usage()
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

        # OpenAI形式のusage
        openai_response = TextGeneratorResponse.create_success(
            content="OpenAI text",
            usage={"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
        )
        usage = openai_response.get_usage()
        assert usage is not None
        assert usage.prompt_tokens == 80
        assert usage.completion_tokens == 40
        assert usage.total_tokens == 120

        # Usage情報なし
        no_usage_response = TextGeneratorResponse.create_success("No usage")
        usage = no_usage_response.get_usage()
        assert usage is None

        # エラーレスポンス
        error_response = TextGeneratorResponse.create_error("Error")
        usage = error_response.get_usage()
        assert usage is None

    def test_backward_compatibility_claude(self):
        """Claude形式の後方互換性テスト"""
        # 従来の形式をシミュレート
        legacy_data = {
            "replies": ["Generated text from Claude"],
            "meta": [{"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}],
        }

        response = TextGeneratorResponse.model_validate(legacy_data)
        assert response.is_success() is True
        assert response.get_content() == "Generated text from Claude"
        usage = response.get_usage()
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_backward_compatibility_openai(self):
        """OpenAI形式の後方互換性テスト"""
        # 従来のOpenAI形式をシミュレート
        legacy_data = {
            "replies": ["Generated text from OpenAI"],
            "meta": [
                {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
            ],
        }

        response = TextGeneratorResponse.model_validate(legacy_data)
        assert response.is_success() is True
        assert response.get_content() == "Generated text from OpenAI"
        usage = response.get_usage()
        assert usage.prompt_tokens == 80
        assert usage.completion_tokens == 40

    def test_backward_compatibility_error(self):
        """エラー形式の後方互換性テスト"""
        # 従来のエラー形式をシミュレート
        legacy_data = {
            "replies": [],
            "meta": [
                {
                    "error": "Rate limit exceeded",
                    "retry_config": {
                        "max_attempts": 3,
                        "initial_wait": 1.0,
                        "max_wait": 60.0,
                        "multiplier": 2.0,
                    },
                }
            ],
        }

        response = TextGeneratorResponse.model_validate(legacy_data)
        assert response.is_success() is False
        assert response.get_content() is None
        assert response.get_error() == "Rate limit exceeded"
        assert response.meta[0].retry_config["max_attempts"] == 3
