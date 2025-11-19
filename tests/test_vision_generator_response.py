"""Tests for Vision Generator Response classes."""

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

    def test_custom_values(self):
        """カスタム値のテスト"""
        usage = GeneratorUsage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
        )
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.total_tokens == 300

    def test_from_dict(self):
        """辞書からの作成テスト"""
        usage_dict = {"input_tokens": 150, "output_tokens": 75, "total_tokens": 225}
        usage = GeneratorUsage.model_validate(usage_dict)
        assert usage.input_tokens == 150
        assert usage.output_tokens == 75
        assert usage.total_tokens == 225


class TestVisionGeneratorResponse:
    """GeneratorResponseクラスのテスト（Vision用）"""

    def test_create_success_with_usage(self):
        """usageありでの成功レスポンス作成テスト"""
        usage_dict = {"input_tokens": 150, "output_tokens": 75, "total_tokens": 225}
        response = GeneratorResponse.create_success(
            content="Image analysis result", usage=usage_dict
        )

        assert response.status == "success"
        assert response.content == "Image analysis result"
        assert response.usage is not None
        assert response.usage.input_tokens == 150
        assert response.usage.output_tokens == 75
        assert response.usage.total_tokens == 225
        assert response.error is None

    def test_create_success_with_usage_object(self):
        """GeneratorUsageオブジェクトでの成功レスポンス作成テスト"""
        usage_obj = GeneratorUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        response = GeneratorResponse.create_success(
            content="Vision result", usage=usage_obj
        )

        assert response.status == "success"
        assert response.content == "Vision result"
        assert response.usage == usage_obj
        assert response.error is None

    def test_create_success_no_usage(self):
        """usageなしでの成功レスポンス作成テスト"""
        response = GeneratorResponse.create_success(content="No usage analysis")

        assert response.status == "success"
        assert response.content == "No usage analysis"
        assert response.usage is None
        assert response.error is None

    def test_create_error(self):
        """エラーレスポンス作成テスト"""
        response = GeneratorResponse.create_error(error="Vision API Error")

        assert response.status == "error"
        assert response.content is None
        assert response.usage is None
        assert response.error == "Vision API Error"

    def test_create_error_with_usage(self):
        """エラーレスポンス（usageあり）作成テスト"""
        usage_dict = {"input_tokens": 50, "output_tokens": 0, "total_tokens": 50}
        response = GeneratorResponse.create_error(
            error="Partial failure", usage=usage_dict
        )

        assert response.status == "error"
        assert response.content is None
        assert response.usage is not None
        assert response.usage.input_tokens == 50
        assert response.usage.output_tokens == 0
        assert response.error == "Partial failure"

    def test_is_success(self):
        """成功判定テスト"""
        success_response = GeneratorResponse.create_success(content="Success")
        error_response = GeneratorResponse.create_error(error="Error")

        assert success_response.is_success() is True
        assert error_response.is_success() is False

    def test_content_access(self):
        """コンテンツ直接アクセステスト"""
        response = GeneratorResponse.create_success(content="Analysis result")
        error_response = GeneratorResponse.create_error(error="Error")

        assert response.content == "Analysis result"
        assert error_response.content is None

    def test_error_access(self):
        """エラー直接アクセステスト"""
        success_response = GeneratorResponse.create_success(content="Success")
        error_response = GeneratorResponse.create_error(error="Vision error")

        assert success_response.error is None
        assert error_response.error == "Vision error"

    def test_usage_access(self):
        """使用統計直接アクセステスト"""
        usage_dict = {"input_tokens": 75, "output_tokens": 25, "total_tokens": 100}
        response = GeneratorResponse.create_success(
            content="Usage test", usage=usage_dict
        )
        no_usage_response = GeneratorResponse.create_success(content="No usage")

        usage = response.usage
        assert usage is not None
        assert usage.input_tokens == 75
        assert usage.output_tokens == 25
        assert usage.total_tokens == 100

        assert no_usage_response.usage is None
