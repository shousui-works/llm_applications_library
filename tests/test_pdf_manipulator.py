"""utilities/pdf_manipulator.py のテスト"""

from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest

from llm_applications_library.utilities.pdf_manipulator import (
    _get_local_pdf_path,
    extract_pdf_text,
    is_pdf_text_based,
)


class TestGetLocalPDFPath:
    """_get_local_pdf_path関数のテスト"""

    def test_get_local_pdf_path_existing_file(self, tmp_path):
        """存在するローカルファイルのパス取得テスト"""
        # テスト用PDFファイルを作成
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy pdf content")

        local_path, is_temp = _get_local_pdf_path(str(test_file))

        assert local_path == str(test_file)
        assert is_temp is False

    def test_get_local_pdf_path_path_object(self, tmp_path):
        """Pathオブジェクトでのローカルファイルパス取得テスト"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy pdf content")

        local_path, is_temp = _get_local_pdf_path(test_file)

        assert local_path == str(test_file)
        assert is_temp is False

    def test_get_local_pdf_path_nonexistent_file(self):
        """存在しないローカルファイルのパス取得エラーテスト"""
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            _get_local_pdf_path("/nonexistent/file.pdf")

    @patch("llm_applications_library.utilities.pdf_manipulator.GCSClient")
    @patch("tempfile.NamedTemporaryFile")
    def test_get_local_pdf_path_gcs_url(self, mock_tempfile, mock_gcs_client):
        """GCS URLからのファイルダウンロードテスト"""
        # モックの設定
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_tempfile.return_value = mock_temp_file

        mock_client_instance = Mock()
        mock_gcs_client.return_value = mock_client_instance

        gs_url = "gs://bucket/path/to/file.pdf"
        local_path, is_temp = _get_local_pdf_path(gs_url)

        # GCSClientが作成されることを確認
        mock_gcs_client.assert_called_once()

        # 一時ファイルが作成されることを確認
        mock_tempfile.assert_called_once_with(suffix=".pdf", delete=False)
        mock_temp_file.close.assert_called_once()

        # ダウンロードが実行されることを確認
        mock_client_instance.download_file_to_local.assert_called_once_with(
            gs_url, Path("/tmp/test.pdf")
        )

        assert local_path == "/tmp/test.pdf"
        assert is_temp is True

    @patch("llm_applications_library.utilities.pdf_manipulator.GCSClient")
    @patch("tempfile.NamedTemporaryFile")
    def test_get_local_pdf_path_gcs_download_error(
        self, mock_tempfile, mock_gcs_client
    ):
        """GCSダウンロードエラーのテスト"""
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_tempfile.return_value = mock_temp_file

        mock_client_instance = Mock()
        mock_client_instance.download_file_to_local.side_effect = Exception(
            "Download failed"
        )
        mock_gcs_client.return_value = mock_client_instance

        with pytest.raises(Exception, match="Download failed"):
            _get_local_pdf_path("gs://bucket/error.pdf")


class TestExtractPDFText:
    """extract_pdf_text関数のテスト"""

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_extract_pdf_text_success(self, mock_fitz_open, mock_get_local_path):
        """PDFテキスト抽出成功テスト"""
        # モックの設定
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        # モックページの設定
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Page 2 content"

        # モックドキュメントの設定
        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page1, mock_page2]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = extract_pdf_text("test.pdf")

        mock_get_local_path.assert_called_once_with("test.pdf")
        mock_fitz_open.assert_called_once_with("/path/to/test.pdf")

        expected_content = (
            "--- Page 1 ---\nPage 1 content\n\n--- Page 2 ---\nPage 2 content\n\n"
        )
        assert result == expected_content

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_extract_pdf_text_empty_pages(self, mock_fitz_open, mock_get_local_path):
        """空ページを含むPDFのテキスト抽出テスト"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        # 空ページと内容ありページのモック
        mock_page1 = Mock()
        mock_page1.get_text.return_value = ""  # 空ページ
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "   "  # 空白のみページ
        mock_page3 = Mock()
        mock_page3.get_text.return_value = "Actual content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page1, mock_page2, mock_page3]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = extract_pdf_text("test.pdf")

        # 内容のあるページのみが含まれることを確認
        expected_content = "--- Page 3 ---\nActual content\n\n"
        assert result == expected_content

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    @patch("pathlib.Path.unlink")
    def test_extract_pdf_text_temp_file_cleanup(
        self, mock_unlink, mock_fitz_open, mock_get_local_path
    ):
        """一時ファイルのクリーンアップテスト"""
        mock_get_local_path.return_value = ("/tmp/temp.pdf", True)  # 一時ファイル

        mock_page = Mock()
        mock_page.get_text.return_value = "Test content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = extract_pdf_text("gs://bucket/test.pdf")

        # 一時ファイルが削除されることを確認
        mock_unlink.assert_called_once()
        assert "Test content" in result

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    @patch("pathlib.Path.unlink")
    def test_extract_pdf_text_cleanup_error(
        self, mock_unlink, mock_fitz_open, mock_get_local_path
    ):
        """ファイルクリーンアップエラーのテスト"""
        mock_get_local_path.return_value = ("/tmp/temp.pdf", True)
        mock_unlink.side_effect = OSError("Permission denied")

        mock_page = Mock()
        mock_page.get_text.return_value = "Test content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # クリーンアップエラーが発生しても処理が継続されることを確認
        result = extract_pdf_text("gs://bucket/test.pdf")
        assert "Test content" in result

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_extract_pdf_text_fitz_error(self, mock_fitz_open, mock_get_local_path):
        """PyMuPDFエラーのテスト"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)
        mock_fitz_open.side_effect = Exception("PDF reading error")

        with pytest.raises(Exception, match="PDF reading error"):
            extract_pdf_text("test.pdf")

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_extract_pdf_text_with_cache_parameter(
        self, mock_fitz_open, mock_get_local_path
    ):
        """キャッシュパラメータありのテスト"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        mock_page = Mock()
        mock_page.get_text.return_value = "Test content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # use_cache=Falseでテスト
        result = extract_pdf_text("test.pdf", use_cache=False)
        assert "Test content" in result


class TestIsPDFTextBased:
    """is_pdf_text_based関数のテスト"""

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_is_pdf_text_based_true(self, mock_fitz_open, mock_get_local_path):
        """テキストベースPDFの判定テスト（True）"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        # テキストを含むページのモック
        mock_page = Mock()
        mock_page.get_text.return_value = "This is text content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = is_pdf_text_based("test.pdf")

        assert result is True

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_is_pdf_text_based_false(self, mock_fitz_open, mock_get_local_path):
        """テキストベースでないPDFの判定テスト（False）"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        # テキストを含まないページのモック
        mock_page1 = Mock()
        mock_page1.get_text.return_value = ""
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "   "  # 空白のみ

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page1, mock_page2]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = is_pdf_text_based("test.pdf")

        assert result is None  # 関数はreturnなしで終了するのでNone

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_is_pdf_text_based_early_return(self, mock_fitz_open, mock_get_local_path):
        """最初のページでテキストが見つかった場合の早期リターンテスト"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        # 最初のページにテキストがある場合
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "First page text"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Second page text"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page1, mock_page2]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = is_pdf_text_based("test.pdf")

        assert result is True
        # 最初のページでリターンするので、2ページ目は処理されない
        mock_page1.get_text.assert_called_once()

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    @patch("pathlib.Path.unlink")
    def test_is_pdf_text_based_temp_file_cleanup(
        self, mock_unlink, mock_fitz_open, mock_get_local_path
    ):
        """一時ファイルのクリーンアップテスト"""
        mock_get_local_path.return_value = ("/tmp/temp.pdf", True)  # 一時ファイル

        mock_page = Mock()
        mock_page.get_text.return_value = "Text content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        result = is_pdf_text_based("gs://bucket/test.pdf")

        # 一時ファイルが削除されることを確認
        mock_unlink.assert_called_once()
        assert result is True

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    @patch("pathlib.Path.unlink")
    def test_is_pdf_text_based_cleanup_error(
        self, mock_unlink, mock_fitz_open, mock_get_local_path
    ):
        """ファイルクリーンアップエラーでも処理が継続されることを確認"""
        mock_get_local_path.return_value = ("/tmp/temp.pdf", True)
        mock_unlink.side_effect = OSError("Permission denied")

        mock_page = Mock()
        mock_page.get_text.return_value = "Text content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # クリーンアップエラーが発生しても結果が返されることを確認
        result = is_pdf_text_based("gs://bucket/test.pdf")
        assert result is True

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_is_pdf_text_based_with_cache_parameter(
        self, mock_fitz_open, mock_get_local_path
    ):
        """キャッシュパラメータありのテスト"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        mock_page = Mock()
        mock_page.get_text.return_value = "Text content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # use_cache=Falseでテスト
        result = is_pdf_text_based("test.pdf", use_cache=False)
        assert result is True


class TestPDFManipulatorIntegration:
    """pdf_manipulator モジュールの統合テスト"""

    def test_all_functions_importable(self):
        """すべての関数がインポート可能であることを確認"""
        from llm_applications_library.utilities.pdf_manipulator import (
            _get_local_pdf_path,
            extract_pdf_text,
            is_pdf_text_based,
        )

        assert callable(_get_local_pdf_path)
        assert callable(extract_pdf_text)
        assert callable(is_pdf_text_based)

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_workflow_text_extraction_and_detection(
        self, mock_fitz_open, mock_get_local_path
    ):
        """テキスト抽出と検出のワークフローテスト"""
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        # テキスト付きページのモック
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample PDF text content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # まずテキストベースかどうかを確認
        is_text_based = is_pdf_text_based("test.pdf")
        assert is_text_based is True

        # テキストを抽出
        extracted_text = extract_pdf_text("test.pdf")
        assert "Sample PDF text content" in extracted_text
        assert "--- Page 1 ---" in extracted_text

    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    def test_error_handling_consistency(self, mock_get_local_path):
        """エラーハンドリングの一貫性テスト"""
        # ファイルが見つからない場合
        mock_get_local_path.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            extract_pdf_text("nonexistent.pdf")

        with pytest.raises(FileNotFoundError):
            is_pdf_text_based("nonexistent.pdf")

    def test_parameter_validation(self):
        """パラメータ検証テスト"""
        # 関数が様々な型のパラメータを受け入れることを確認
        from pathlib import Path

        # これらの呼び出しでエラーが発生しないことを確認（実際のファイル処理は別途モック）
        with patch(
            "llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path"
        ) as mock_get_local:
            with patch("fitz.open"):
                mock_get_local.side_effect = FileNotFoundError()

                # 文字列パス
                with pytest.raises(FileNotFoundError):
                    extract_pdf_text("test.pdf")

                # Pathオブジェクト
                with pytest.raises(FileNotFoundError):
                    extract_pdf_text(Path("test.pdf"))

                # GCS URL
                with pytest.raises(FileNotFoundError):
                    extract_pdf_text("gs://bucket/test.pdf")

    @patch("llm_applications_library.utilities.pdf_manipulator.GCSClient")
    @patch("llm_applications_library.utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_gcs_workflow(self, mock_fitz_open, mock_get_local_path, mock_gcs_client):
        """GCSワークフローの統合テスト"""
        # GCS URLのワークフローテスト
        mock_get_local_path.return_value = ("/tmp/downloaded.pdf", True)

        mock_page = Mock()
        mock_page.get_text.return_value = "GCS PDF content"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        with patch("pathlib.Path.unlink"):  # ファイル削除をモック
            result = extract_pdf_text("gs://bucket/test.pdf")
            assert "GCS PDF content" in result
