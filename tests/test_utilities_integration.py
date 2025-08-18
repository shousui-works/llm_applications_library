"""utilities モジュール全体の統合テスト"""

import pytest
from unittest.mock import patch, Mock, MagicMock

from utilities import (
    # File I/O
    load_yaml,
    load_text,
    save_text,
    # Logging
    configure_openai_logging,
    setup_debug_logging_without_openai_http,
    # Token utilities
    get_encoding_for_model,
    count_tokens,
    count_tokens_for_messages,
    split_text_by_tokens,
    estimate_prompt_tokens,
    # PDF utilities
    extract_pdf_text,
    is_pdf_text_based,
)


class TestUtilitiesModuleImports:
    """utilities モジュールのインポートテスト"""

    def test_all_functions_importable(self):
        """すべての公開関数がインポート可能であることを確認"""
        # File I/O functions
        assert callable(load_yaml)
        assert callable(load_text)
        assert callable(save_text)

        # Logging functions
        assert callable(configure_openai_logging)
        assert callable(setup_debug_logging_without_openai_http)

        # Token utility functions
        assert callable(get_encoding_for_model)
        assert callable(count_tokens)
        assert callable(count_tokens_for_messages)
        assert callable(split_text_by_tokens)
        assert callable(estimate_prompt_tokens)

        # PDF utility functions
        assert callable(extract_pdf_text)
        assert callable(is_pdf_text_based)

    def test_module_all_attribute(self):
        """__all__属性に正しい関数が含まれていることを確認"""
        import utilities

        expected_functions = [
            # File I/O
            "load_yaml",
            "load_text",
            "save_text",
            # Logging
            "configure_openai_logging",
            "setup_debug_logging_without_openai_http",
            # Token utilities
            "get_encoding_for_model",
            "count_tokens",
            "count_tokens_for_messages",
            "split_text_by_tokens",
            "estimate_prompt_tokens",
            # PDF utilities
            "extract_pdf_text",
            "is_pdf_text_based",
        ]

        assert hasattr(utilities, "__all__")
        assert set(utilities.__all__) == set(expected_functions)

    def test_direct_module_import(self):
        """モジュール直接インポートのテスト"""
        import utilities

        # 各カテゴリの関数が利用可能であることを確認
        assert hasattr(utilities, "load_yaml")
        assert hasattr(utilities, "configure_openai_logging")
        assert hasattr(utilities, "count_tokens")
        assert hasattr(utilities, "extract_pdf_text")

    def test_specific_function_imports(self):
        """特定関数のインポートテスト"""
        from utilities import load_yaml, count_tokens, configure_openai_logging

        assert callable(load_yaml)
        assert callable(count_tokens)
        assert callable(configure_openai_logging)


class TestCrossModuleFunctionality:
    """モジュール間の機能連携テスト"""

    def test_file_io_and_token_counting_integration(self, tmp_path):
        """ファイルI/Oとトークン計算の統合テスト"""
        # テストファイルを作成
        test_file = tmp_path / "test.txt"
        content = "This is a test document for token counting."

        # ファイルに保存
        save_text(test_file, content)

        # ファイルから読み込み
        loaded_content = load_text(test_file)
        assert loaded_content == content

        # トークン数を計算
        token_count = count_tokens(loaded_content, "gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_yaml_file_and_template_estimation(self, tmp_path):
        """YAMLファイルとテンプレート推定の統合テスト"""
        # YAMLデータを準備
        yaml_data = {
            "template": "Hello, {name}! Your {item} is ready.",
            "variables": {"name": "Alice", "item": "order"},
        }

        # YAMLファイルに保存
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(yaml_data, f)

        # YAMLファイルを読み込み
        loaded_data = load_yaml(yaml_file)
        assert loaded_data == yaml_data

        # テンプレートのトークン数を推定
        template = loaded_data["template"]
        variables = loaded_data["variables"]
        estimated_tokens = estimate_prompt_tokens(template, variables, "gpt-4")

        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0

    @patch("utilities.logging_config.logging.getLogger")
    def test_logging_configuration_integration(self, mock_get_logger):
        """ログ設定の統合テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # OpenAIログを設定
        configure_openai_logging()

        # デバッグログ設定（OpenAI HTTPログ無効化）
        with patch(
            "utilities.logging_config.configure_openai_logging"
        ) as mock_configure:
            setup_debug_logging_without_openai_http()
            mock_configure.assert_called_once()

    def test_text_splitting_and_message_tokens(self):
        """テキスト分割とメッセージトークンの統合テスト"""
        # 長いテキストを分割
        long_text = "This is a long document. " * 50
        chunks = split_text_by_tokens(long_text, 100, "gpt-4")

        assert len(chunks) > 1

        # 各チャンクをメッセージ形式にしてトークン数を計算
        for chunk in chunks[:3]:  # 最初の3チャンクのみテスト
            messages = [{"role": "user", "content": f"Analyze this text: {chunk}"}]
            message_tokens = count_tokens_for_messages(messages, "gpt-4")
            assert isinstance(message_tokens, int)
            assert message_tokens > 0


class TestUtilitiesWorkflows:
    """utilities モジュールの実用的なワークフローテスト"""

    def test_document_processing_workflow(self, tmp_path):
        """文書処理ワークフローのテスト"""
        # 1. 設定ファイルを作成
        config_data = {
            "processing": {"max_tokens": 500, "model": "gpt-4", "overlap_tokens": 50},
            "document_template": "Process this document: {content}",
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(config_data, f)

        # 2. 設定を読み込み
        config = load_yaml(config_file)
        assert config["processing"]["max_tokens"] == 500

        # 3. 文書ファイルを作成
        document_content = "This is a sample document for processing. " * 20
        doc_file = tmp_path / "document.txt"
        save_text(doc_file, document_content)

        # 4. 文書を読み込み
        content = load_text(doc_file)

        # 5. トークン数をチェック
        total_tokens = count_tokens(content, config["processing"]["model"])

        # 6. 必要に応じてテキストを分割
        if total_tokens > config["processing"]["max_tokens"]:
            chunks = split_text_by_tokens(
                content,
                config["processing"]["max_tokens"],
                config["processing"]["model"],
                config["processing"]["overlap_tokens"],
            )
            assert len(chunks) > 1
        else:
            chunks = [content]

        # 7. 各チャンクのプロンプトトークン数を推定
        for chunk in chunks[:2]:  # 最初の2チャンクのみテスト
            template = config["document_template"]
            variables = {"content": chunk}
            prompt_tokens = estimate_prompt_tokens(
                template, variables, config["processing"]["model"]
            )
            assert isinstance(prompt_tokens, int)
            assert prompt_tokens > 0

    @patch("utilities.pdf_manipulator._get_local_pdf_path")
    @patch("fitz.open")
    def test_pdf_processing_workflow(self, mock_fitz_open, mock_get_local_path):
        """PDF処理ワークフローのテスト"""
        # モックの設定
        mock_get_local_path.return_value = ("/path/to/test.pdf", False)

        mock_page = Mock()
        mock_page.get_text.return_value = "Sample PDF content for token analysis."

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page]
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # 1. PDFがテキストベースかチェック
        is_text_based = is_pdf_text_based("test.pdf")
        assert is_text_based is True

        # 2. テキストベースの場合、テキストを抽出
        if is_text_based:
            extracted_text = extract_pdf_text("test.pdf")
            assert "Sample PDF content" in extracted_text

            # 3. 抽出したテキストのトークン数を計算
            token_count = count_tokens(extracted_text, "gpt-4")
            assert isinstance(token_count, int)
            assert token_count > 0

            # 4. 必要に応じてテキストを分割
            if token_count > 100:
                chunks = split_text_by_tokens(extracted_text, 100, "gpt-4")
                assert len(chunks) >= 1

    @patch("utilities.logging_config.logging.getLogger")
    def test_logging_setup_workflow(self, mock_get_logger):
        """ログ設定ワークフローのテスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # 1. アプリケーション開始時のログ設定
        setup_debug_logging_without_openai_http()

        # 2. 特定の処理でOpenAIログレベルを変更
        import logging

        configure_openai_logging(logging.ERROR)

        # 3. ログ設定が正しく呼び出されることを確認
        assert mock_get_logger.called
        assert mock_logger.setLevel.called


class TestUtilitiesErrorHandling:
    """utilities モジュールのエラーハンドリングテスト"""

    def test_file_operations_error_consistency(self):
        """ファイル操作のエラー一貫性テスト"""
        nonexistent_file = "/nonexistent/path/file.txt"

        # 存在しないファイルに対して一貫したエラーが発生することを確認
        with pytest.raises(FileNotFoundError):
            load_text(nonexistent_file)

        with pytest.raises(FileNotFoundError):
            load_yaml(nonexistent_file)

    def test_token_utilities_error_handling(self):
        """トークンユーティリティのエラーハンドリングテスト"""
        # 空文字列や不正な入力に対する適切な処理
        assert count_tokens("", "gpt-4") == 0
        assert count_tokens_for_messages([], "gpt-4") == 3  # プライミングトークンのみ
        assert split_text_by_tokens("", 100, "gpt-4") == []

    def test_cross_module_error_propagation(self):
        """モジュール間のエラー伝播テスト"""
        # ファイル読み込みエラーがトークン計算まで適切に伝播することを確認
        with pytest.raises(FileNotFoundError):
            content = load_text("/nonexistent/file.txt")
            count_tokens(content, "gpt-4")  # ここには到達しない


class TestUtilitiesPerformance:
    """utilities モジュールのパフォーマンステスト"""

    def test_large_text_handling(self):
        """大きなテキストの処理テスト"""
        # 大きなテキストを生成
        large_text = "This is a large text for performance testing. " * 1000

        # トークン計算が妥当な時間で完了することを確認
        token_count = count_tokens(large_text, "gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 1000

        # テキスト分割が適切に動作することを確認
        chunks = split_text_by_tokens(large_text, 500, "gpt-4")
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_multiple_file_operations(self, tmp_path):
        """複数ファイル操作のテスト"""
        files_data = {
            "file1.txt": "Content of file 1",
            "file2.txt": "Content of file 2",
            "file3.txt": "Content of file 3",
        }

        # 複数ファイルの保存
        for filename, content in files_data.items():
            file_path = tmp_path / filename
            save_text(file_path, content)

        # 複数ファイルの読み込みとトークン計算
        total_tokens = 0
        for filename in files_data.keys():
            file_path = tmp_path / filename
            content = load_text(file_path)
            tokens = count_tokens(content, "gpt-4")
            total_tokens += tokens

        assert total_tokens > 0
        assert isinstance(total_tokens, int)
