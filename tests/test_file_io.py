"""utilities/file_io.py のテスト"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from llm_applications_library.utilities.file_io import load_yaml, load_text, save_text


class TestLoadYaml:
    """load_yaml関数のテスト"""

    def test_load_yaml_success_with_path_object(self, tmp_path):
        """Pathオブジェクトを使ったYAML読み込み成功テスト"""
        # テストYAMLファイルを作成
        yaml_content = {
            "name": "test",
            "version": 1.0,
            "items": ["a", "b", "c"],
            "config": {"debug": True, "timeout": 30},
        }

        yaml_file = tmp_path / "test.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        # YAML読み込みテスト
        result = load_yaml(yaml_file)

        assert result == yaml_content
        assert result["name"] == "test"
        assert result["version"] == 1.0
        assert result["items"] == ["a", "b", "c"]
        assert result["config"]["debug"] is True

    def test_load_yaml_success_with_string_path(self, tmp_path):
        """文字列パスを使ったYAML読み込み成功テスト"""
        yaml_content = {"key": "value", "number": 42}

        yaml_file = tmp_path / "test.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        # 文字列パスでテスト
        result = load_yaml(str(yaml_file))

        assert result == yaml_content

    def test_load_yaml_japanese_content(self, tmp_path):
        """日本語を含むYAMLファイルの読み込みテスト"""
        yaml_content = {
            "名前": "テスト",
            "説明": "これは日本語のテストです",
            "項目": ["項目1", "項目2", "項目3"],
        }

        yaml_file = tmp_path / "japanese.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        result = load_yaml(yaml_file)

        assert result == yaml_content
        assert result["名前"] == "テスト"

    def test_load_yaml_empty_file(self, tmp_path):
        """空のYAMLファイルの読み込みテスト"""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")

        result = load_yaml(yaml_file)

        # 空のYAMLファイルはNoneを返す
        assert result is None

    def test_load_yaml_file_not_found(self):
        """存在しないファイルの読み込みエラーテスト"""
        non_existent_file = Path("/non/existent/file.yaml")

        with pytest.raises(FileNotFoundError):
            load_yaml(non_existent_file)

    def test_load_yaml_invalid_yaml_format(self, tmp_path):
        """不正なYAML形式のファイル読み込みエラーテスト"""
        yaml_file = tmp_path / "invalid.yaml"
        # 不正なYAML内容を作成
        yaml_file.write_text("key: value\n  invalid_indent: bad", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            load_yaml(yaml_file)

    def test_load_yaml_complex_structure(self, tmp_path):
        """複雑なYAML構造の読み込みテスト"""
        yaml_content = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"username": "admin", "password": "secret"},
            },
            "services": [
                {"name": "web", "port": 8080, "enabled": True},
                {"name": "api", "port": 3000, "enabled": False},
            ],
            "features": {
                "logging": True,
                "monitoring": False,
                "caching": {"enabled": True, "ttl": 3600},
            },
        }

        yaml_file = tmp_path / "complex.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        result = load_yaml(yaml_file)

        assert result == yaml_content
        assert result["database"]["port"] == 5432
        assert len(result["services"]) == 2
        assert result["services"][0]["name"] == "web"


class TestLoadText:
    """load_text関数のテスト"""

    def test_load_text_success_utf8(self, tmp_path):
        """UTF-8テキストファイルの読み込み成功テスト"""
        content = "Hello, World!\nこんにちは、世界！\n"
        text_file = tmp_path / "test.txt"
        text_file.write_text(content, encoding="utf-8")

        result = load_text(text_file)

        assert result == content

    def test_load_text_success_with_string_path(self, tmp_path):
        """文字列パスを使ったテキスト読み込み成功テスト"""
        content = "Test content"
        text_file = tmp_path / "test.txt"
        text_file.write_text(content, encoding="utf-8")

        result = load_text(str(text_file))

        assert result == content

    def test_load_text_different_encoding(self, tmp_path):
        """異なるエンコーディングでのテキスト読み込みテスト"""
        content = "Test with different encoding"
        text_file = tmp_path / "test_latin1.txt"

        # latin-1エンコーディングで保存
        with open(text_file, "w", encoding="latin-1") as f:
            f.write(content)

        # latin-1エンコーディングで読み込み
        result = load_text(text_file, encoding="latin-1")

        assert result == content

    def test_load_text_empty_file(self, tmp_path):
        """空のテキストファイルの読み込みテスト"""
        text_file = tmp_path / "empty.txt"
        text_file.write_text("", encoding="utf-8")

        result = load_text(text_file)

        assert result == ""

    def test_load_text_multiline_content(self, tmp_path):
        """複数行テキストの読み込みテスト"""
        content = """Line 1
Line 2
Line 3
日本語の行
最後の行"""

        text_file = tmp_path / "multiline.txt"
        text_file.write_text(content, encoding="utf-8")

        result = load_text(text_file)

        assert result == content
        assert "Line 1" in result
        assert "日本語の行" in result

    def test_load_text_file_not_found(self):
        """存在しないファイルの読み込みエラーテスト"""
        non_existent_file = Path("/non/existent/file.txt")

        with pytest.raises(FileNotFoundError):
            load_text(non_existent_file)

    def test_load_text_encoding_error(self, tmp_path):
        """エンコーディングエラーのテスト"""
        # UTF-8でファイルを作成
        content = "日本語テキスト"
        text_file = tmp_path / "utf8_file.txt"
        text_file.write_text(content, encoding="utf-8")

        # ASCII エンコーディングで読み込もうとする（エラーが発生する）
        with pytest.raises(UnicodeDecodeError):
            load_text(text_file, encoding="ascii")


class TestSaveText:
    """save_text関数のテスト"""

    def test_save_text_success(self, tmp_path):
        """テキストファイル保存成功テスト"""
        content = "Hello, World!\nこんにちは、世界！"
        text_file = tmp_path / "output.txt"

        save_text(text_file, content)

        # ファイルが作成されていることを確認
        assert text_file.exists()

        # 内容を確認
        saved_content = text_file.read_text(encoding="utf-8")
        assert saved_content == content

    def test_save_text_with_string_path(self, tmp_path):
        """文字列パスを使ったテキスト保存テスト"""
        content = "Test content"
        text_file = tmp_path / "output.txt"

        save_text(str(text_file), content)

        assert text_file.exists()
        assert text_file.read_text(encoding="utf-8") == content

    def test_save_text_different_encoding(self, tmp_path):
        """異なるエンコーディングでのテキスト保存テスト"""
        content = "Test content"
        text_file = tmp_path / "output_latin1.txt"

        save_text(text_file, content, encoding="latin-1")

        # latin-1エンコーディングで読み込んで確認
        with open(text_file, "r", encoding="latin-1") as f:
            saved_content = f.read()

        assert saved_content == content

    def test_save_text_creates_directory(self, tmp_path):
        """ディレクトリが存在しない場合の自動作成テスト"""
        content = "Test content"
        nested_file = tmp_path / "deep" / "nested" / "path" / "file.txt"

        save_text(nested_file, content)

        # ディレクトリが作成されていることを確認
        assert nested_file.parent.exists()
        assert nested_file.exists()
        assert nested_file.read_text(encoding="utf-8") == content

    def test_save_text_overwrites_existing_file(self, tmp_path):
        """既存ファイルの上書きテスト"""
        text_file = tmp_path / "existing.txt"

        # 最初の内容を保存
        original_content = "Original content"
        save_text(text_file, original_content)
        assert text_file.read_text(encoding="utf-8") == original_content

        # 内容を上書き
        new_content = "New content"
        save_text(text_file, new_content)
        assert text_file.read_text(encoding="utf-8") == new_content

    def test_save_text_empty_content(self, tmp_path):
        """空の内容の保存テスト"""
        text_file = tmp_path / "empty.txt"

        save_text(text_file, "")

        assert text_file.exists()
        assert text_file.read_text(encoding="utf-8") == ""

    def test_save_text_multiline_content(self, tmp_path):
        """複数行テキストの保存テスト"""
        content = """Line 1
Line 2
Line 3
日本語の行
最後の行"""

        text_file = tmp_path / "multiline.txt"

        save_text(text_file, content)

        saved_content = text_file.read_text(encoding="utf-8")
        assert saved_content == content

    def test_save_text_japanese_content(self, tmp_path):
        """日本語コンテンツの保存テスト"""
        content = (
            "これは日本語のテストです。\n漢字、ひらがな、カタカナが含まれています。"
        )
        text_file = tmp_path / "japanese.txt"

        save_text(text_file, content)

        saved_content = text_file.read_text(encoding="utf-8")
        assert saved_content == content

    @patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))
    def test_save_text_permission_error(self, mock_mkdir):
        """ファイル書き込み権限エラーのテスト"""
        with pytest.raises(OSError, match="Permission denied"):
            save_text("/restricted/path/file.txt", "content")


class TestFileIOIntegration:
    """file_io関数の統合テスト"""

    def test_save_and_load_text_roundtrip(self, tmp_path):
        """テキスト保存→読み込みの往復テスト"""
        content = "Hello, World!\n日本語テキスト\n特殊文字: @#$%^&*()"
        text_file = tmp_path / "roundtrip.txt"

        # 保存
        save_text(text_file, content)

        # 読み込み
        loaded_content = load_text(text_file)

        assert loaded_content == content

    def test_yaml_and_text_operations(self, tmp_path):
        """YAMLとテキスト操作の組み合わせテスト"""
        # YAMLデータを準備
        yaml_data = {
            "app_name": "TestApp",
            "description": "テストアプリケーション",
            "version": "1.0.0",
        }

        yaml_file = tmp_path / "config.yaml"

        # YAMLファイルを作成
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f)

        # YAMLを読み込み
        loaded_yaml = load_yaml(yaml_file)
        assert loaded_yaml == yaml_data

        # YAMLデータをテキストとして保存
        text_content = (
            f"App: {loaded_yaml['app_name']}\nDescription: {loaded_yaml['description']}"
        )
        text_file = tmp_path / "summary.txt"
        save_text(text_file, text_content)

        # テキストを読み込んで確認
        loaded_text = load_text(text_file)
        assert "TestApp" in loaded_text
        assert "テストアプリケーション" in loaded_text
