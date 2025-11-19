"""Tests for Vision Generator Response classes."""

from llm_applications_library.llm.generators.schema import (
    VisionAnalysisUsage,
    VisionAnalysisResult,
    VisionGeneratorResponse,
)


class TestVisionAnalysisUsage:
    """VisionAnalysisUsageクラスのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        usage = VisionAnalysisUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        """カスタム値のテスト"""
        usage = VisionAnalysisUsage(
            input_tokens=100, output_tokens=50, total_tokens=150
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_from_dict(self):
        """辞書からの作成テスト"""
        usage_dict = {
            "input_tokens": 75,
            "output_tokens": 25,
            "total_tokens": 100,
            "extra_field": "ignored",  # extra fieldsは無視される
        }
        usage = VisionAnalysisUsage.model_validate(usage_dict)
        assert usage.input_tokens == 75
        assert usage.output_tokens == 25
        assert usage.total_tokens == 100


class TestVisionAnalysisResult:
    """VisionAnalysisResultクラスのテスト"""

    def test_success_result(self):
        """成功結果のテスト"""
        usage = VisionAnalysisUsage(input_tokens=50, output_tokens=30, total_tokens=80)
        result = VisionAnalysisResult(
            success=True, content="This is a success response", usage=usage, error=None
        )

        assert result.success is True
        assert result.content == "This is a success response"
        assert result.usage.input_tokens == 50
        assert result.error is None

    def test_error_result(self):
        """エラー結果のテスト"""
        result = VisionAnalysisResult(
            success=False, content=None, usage=None, error="API Error occurred"
        )

        assert result.success is False
        assert result.content is None
        assert result.usage is None
        assert result.error == "API Error occurred"

    def test_minimal_success(self):
        """最小限の成功結果テスト"""
        result = VisionAnalysisResult(success=True)
        assert result.success is True
        assert result.content is None
        assert result.usage is None
        assert result.error is None


class TestVisionGeneratorResponse:
    """VisionGeneratorResponseクラスのテスト"""

    def test_create_success(self):
        """成功レスポンスの作成テスト"""
        usage_dict = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        response = VisionGeneratorResponse.create_success(
            content="Analysis complete", usage=usage_dict
        )

        assert len(response.replies) == 1
        result = response.replies[0]
        assert result.success is True
        assert result.content == "Analysis complete"
        assert result.usage.input_tokens == 100
        assert result.error is None

    def test_create_success_with_usage_object(self):
        """UsageオブジェクトでのレスポンスSuccessテスト"""
        usage = VisionAnalysisUsage(input_tokens=75, output_tokens=25, total_tokens=100)
        response = VisionGeneratorResponse.create_success(
            content="Success with object", usage=usage
        )

        assert len(response.replies) == 1
        result = response.replies[0]
        assert result.success is True
        assert result.content == "Success with object"
        assert result.usage.input_tokens == 75

    def test_create_success_no_usage(self):
        """Usage情報なしでのSuccessテスト"""
        response = VisionGeneratorResponse.create_success(
            content="Success without usage"
        )

        assert len(response.replies) == 1
        result = response.replies[0]
        assert result.success is True
        assert result.content == "Success without usage"
        assert result.usage is None

    def test_create_error(self):
        """エラーレスポンスの作成テスト"""
        response = VisionGeneratorResponse.create_error(
            error="Vision API failed",
            usage={"input_tokens": 25, "output_tokens": 0, "total_tokens": 25},
        )

        assert len(response.replies) == 1
        result = response.replies[0]
        assert result.success is False
        assert result.content is None
        assert result.error == "Vision API failed"
        assert result.usage.input_tokens == 25

    def test_is_success(self):
        """成功判定テスト"""
        # 成功レスポンス
        success_response = VisionGeneratorResponse.create_success("Success")
        assert success_response.is_success() is True

        # エラーレスポンス
        error_response = VisionGeneratorResponse.create_error("Error")
        assert error_response.is_success() is False

    def test_get_content(self):
        """コンテンツ取得テスト"""
        # 成功レスポンス
        success_response = VisionGeneratorResponse.create_success("Test content")
        assert success_response.get_content() == "Test content"

        # エラーレスポンス
        error_response = VisionGeneratorResponse.create_error("Error")
        assert error_response.get_content() is None

    def test_get_error(self):
        """エラー取得テスト"""
        # 成功レスポンス
        success_response = VisionGeneratorResponse.create_success("Success")
        assert success_response.get_error() is None

        # エラーレスポンス
        error_response = VisionGeneratorResponse.create_error("Test error")
        assert error_response.get_error() == "Test error"

    def test_get_total_usage(self):
        """総使用統計取得テスト"""
        # 複数の結果を持つレスポンスを手動作成
        result1 = VisionAnalysisResult(
            success=True,
            content="Result 1",
            usage=VisionAnalysisUsage(
                input_tokens=50, output_tokens=25, total_tokens=75
            ),
        )
        result2 = VisionAnalysisResult(
            success=True,
            content="Result 2",
            usage=VisionAnalysisUsage(
                input_tokens=30, output_tokens=15, total_tokens=45
            ),
        )

        response = VisionGeneratorResponse(replies=[result1, result2])
        total_usage = response.get_total_usage()

        assert total_usage.input_tokens == 80  # 50 + 30
        assert total_usage.output_tokens == 40  # 25 + 15
        assert total_usage.total_tokens == 120  # 80 + 40

    def test_get_total_usage_no_usage(self):
        """Usage情報なしでの総使用統計テスト"""
        response = VisionGeneratorResponse.create_success("No usage")
        total_usage = response.get_total_usage()

        assert total_usage.input_tokens == 0
        assert total_usage.output_tokens == 0
        assert total_usage.total_tokens == 0

    def test_multiple_replies(self):
        """複数の結果を持つレスポンステスト"""
        result1 = VisionAnalysisResult(success=True, content="First result")
        result2 = VisionAnalysisResult(success=False, error="Second result failed")

        response = VisionGeneratorResponse(replies=[result1, result2])

        assert len(response.replies) == 2
        assert response.is_success() is False  # 1つでもfalseがあれば全体はfalse
        assert response.get_content() == "First result"  # 最初の成功コンテンツ
        assert response.get_error() == "Second result failed"  # 最初のエラー

    def test_backward_compatibility(self):
        """後方互換性テスト（従来の形式をシミュレート）"""
        # 従来の {"replies": [{"success": True, ...}]} 形式をシミュレート
        legacy_data = {
            "replies": [
                {
                    "success": True,
                    "content": "Legacy response",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                    "error": None,
                }
            ]
        }

        response = VisionGeneratorResponse.model_validate(legacy_data)
        assert response.is_success() is True
        assert response.get_content() == "Legacy response"
        assert response.get_total_usage().input_tokens == 100
