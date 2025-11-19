"""Tests for Text Generator Response classes."""

from llm_applications_library.llm.generators.schema import (
    GeneratorUsage,
    GeneratorResponse,
)


class TestGeneratorUsage:
    """GeneratorUsageクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        usage = GeneratorUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None

    def test_custom_values(self):
        """カスタム値のテスト"""
        usage = GeneratorUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            prompt_tokens=80,
            completion_tokens=40,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.prompt_tokens == 80
        assert usage.completion_tokens == 40

    def test_from_claude_dict(self):
        """Claude形式辞書からの作成テスト"""
        usage_dict = {"input_tokens": 200, "output_tokens": 100}
        usage = GeneratorUsage.model_validate(usage_dict)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.total_tokens == 0  # デフォルト値

    def test_from_openai_dict(self):
        """OpenAI形式辞書からの作成テスト"""
        usage_dict = {
            "prompt_tokens": 150,
            "completion_tokens": 75,
            "total_tokens": 225,
        }
        usage = GeneratorUsage.model_validate(usage_dict)
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 75
        assert usage.total_tokens == 225
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestGeneratorResponse:
    """GeneratorResponseクラスのテスト"""

    def test_create_success_with_claude_usage(self):
        """Claude形式のusageでの成功レスポンス作成テスト"""
        usage_dict = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        response = GeneratorResponse.create_success(
            content="Generated text", usage=usage_dict
        )

        assert response.status == "success"
        assert response.content == "Generated text"
        assert response.usage is not None
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.error is None

    def test_create_success_with_openai_usage(self):
        """OpenAI形式のusageでの成功レスポンス作成テスト"""
        usage_dict = {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
        response = GeneratorResponse.create_success(
            content="OpenAI response", usage=usage_dict
        )

        assert response.status == "success"
        assert response.content == "OpenAI response"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 80
        assert response.usage.completion_tokens == 40
        assert response.usage.total_tokens == 120

    def test_create_success_no_usage(self):
        """usageなしでの成功レスポンス作成テスト"""
        response = GeneratorResponse.create_success(content="No usage text")

        assert response.status == "success"
        assert response.content == "No usage text"
        assert response.usage is None
        assert response.error is None

    def test_create_error(self):
        """エラーレスポンス作成テスト"""
        response = GeneratorResponse.create_error(error="API Error occurred")

        assert response.status == "error"
        assert response.content is None
        assert response.usage is None
        assert response.error == "API Error occurred"

    def test_is_success(self):
        """成功判定テスト"""
        success_response = GeneratorResponse.create_success(content="Success")
        error_response = GeneratorResponse.create_error(error="Error")

        assert success_response.is_success() is True
        assert error_response.is_success() is False

    def test_content_access(self):
        """コンテンツ直接アクセステスト"""
        response = GeneratorResponse.create_success(content="Test content")
        error_response = GeneratorResponse.create_error(error="Error")

        assert response.content == "Test content"
        assert error_response.content is None

    def test_error_access(self):
        """エラー直接アクセステスト"""
        success_response = GeneratorResponse.create_success(content="Success")
        error_response = GeneratorResponse.create_error(error="Test error")

        assert success_response.error is None
        assert error_response.error == "Test error"

    def test_usage_access(self):
        """使用統計直接アクセステスト"""
        usage_dict = {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75}
        response = GeneratorResponse.create_success(
            content="Usage test", usage=usage_dict
        )
        no_usage_response = GeneratorResponse.create_success(content="No usage")

        usage = response.usage
        assert usage is not None
        assert usage.input_tokens == 50
        assert usage.output_tokens == 25
        assert usage.total_tokens == 75

        assert no_usage_response.usage is None
