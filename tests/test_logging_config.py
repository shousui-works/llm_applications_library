"""utilities/logging_config.py のテスト"""

import logging
from unittest.mock import patch, Mock

from utilities.logging_config import (
    configure_openai_logging,
    setup_debug_logging_without_openai_http,
)


class TestConfigureOpenAILogging:
    """configure_openai_logging関数のテスト"""

    @patch("logging.getLogger")
    def test_configure_openai_logging_default_level(self, mock_get_logger):
        """デフォルトレベル(INFO)でのOpenAIログ設定テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        configure_openai_logging()

        # getLoggerが正しく呼び出されることを確認
        expected_calls = [
            (("openai._base_client",), {}),
            (("openai",), {}),
            (("httpx",), {}),
        ]

        assert mock_get_logger.call_count == 3
        for i, (args, kwargs) in enumerate(expected_calls):
            assert mock_get_logger.call_args_list[i] == (args, kwargs)

        # setLevelがINFOレベルで呼び出されることを確認
        assert mock_logger.setLevel.call_count == 3
        for call in mock_logger.setLevel.call_args_list:
            assert call[0][0] == logging.INFO

    @patch("logging.getLogger")
    def test_configure_openai_logging_custom_level(self, mock_get_logger):
        """カスタムレベルでのOpenAIログ設定テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        configure_openai_logging(logging.WARNING)

        # setLevelがWARNINGレベルで呼び出されることを確認
        assert mock_logger.setLevel.call_count == 3
        for call in mock_logger.setLevel.call_args_list:
            assert call[0][0] == logging.WARNING

    @patch("logging.getLogger")
    def test_configure_openai_logging_debug_level(self, mock_get_logger):
        """DEBUGレベルでのOpenAIログ設定テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        configure_openai_logging(logging.DEBUG)

        # setLevelがDEBUGレベルで呼び出されることを確認
        assert mock_logger.setLevel.call_count == 3
        for call in mock_logger.setLevel.call_args_list:
            assert call[0][0] == logging.DEBUG

    @patch("logging.getLogger")
    def test_configure_openai_logging_error_level(self, mock_get_logger):
        """ERRORレベルでのOpenAIログ設定テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        configure_openai_logging(logging.ERROR)

        # setLevelがERRORレベルで呼び出されることを確認
        assert mock_logger.setLevel.call_count == 3
        for call in mock_logger.setLevel.call_args_list:
            assert call[0][0] == logging.ERROR

    @patch("logging.getLogger")
    def test_configure_openai_logging_all_loggers_configured(self, mock_get_logger):
        """すべての必要なロガーが設定されることを確認"""
        mock_loggers = {}

        def get_logger_side_effect(name):
            if name not in mock_loggers:
                mock_loggers[name] = Mock()
            return mock_loggers[name]

        mock_get_logger.side_effect = get_logger_side_effect

        configure_openai_logging(logging.WARNING)

        # 期待されるロガー名が全て呼び出されたことを確認
        expected_logger_names = ["openai._base_client", "openai", "httpx"]

        actual_logger_names = [call[0][0] for call in mock_get_logger.call_args_list]
        assert set(actual_logger_names) == set(expected_logger_names)

        # 各ロガーのsetLevelが呼び出されたことを確認
        for logger_name in expected_logger_names:
            mock_loggers[logger_name].setLevel.assert_called_once_with(logging.WARNING)


class TestSetupDebugLoggingWithoutOpenAIHTTP:
    """setup_debug_logging_without_openai_http関数のテスト"""

    @patch("logging.getLogger")
    @patch("utilities.logging_config.configure_openai_logging")
    def test_setup_debug_logging_without_openai_http(
        self, mock_configure, mock_get_logger
    ):
        """デバッグログ設定のテスト"""
        mock_loggers = {}

        def get_logger_side_effect(name):
            if name not in mock_loggers:
                mock_loggers[name] = Mock()
            return mock_loggers[name]

        mock_get_logger.side_effect = get_logger_side_effect

        setup_debug_logging_without_openai_http()

        # configure_openai_loggingがWARNINGレベルで呼び出されることを確認
        mock_configure.assert_called_once_with(logging.WARNING)

        # デバッグ用ロガーがDEBUGレベルに設定されることを確認
        debug_logger_names = [
            "services.llm.components",
            "services.llm.generators.openai",
        ]

        for logger_name in debug_logger_names:
            mock_loggers[logger_name].setLevel.assert_called_with(logging.DEBUG)

        # その他のロガーがWARNINGレベルに設定されることを確認
        warning_logger_names = ["urllib3.connectionpool", "posthog"]

        for logger_name in warning_logger_names:
            mock_loggers[logger_name].setLevel.assert_called_with(logging.WARNING)

    @patch("logging.getLogger")
    @patch("utilities.logging_config.configure_openai_logging")
    def test_setup_debug_logging_all_loggers_configured(
        self, mock_configure, mock_get_logger
    ):
        """すべての必要なロガーが設定されることを詳細確認"""
        mock_loggers = {}

        def get_logger_side_effect(name):
            if name not in mock_loggers:
                mock_loggers[name] = Mock()
            return mock_loggers[name]

        mock_get_logger.side_effect = get_logger_side_effect

        setup_debug_logging_without_openai_http()

        # 期待されるロガー設定
        expected_logger_configs = {
            "services.llm.components": logging.DEBUG,
            "services.llm.generators.openai": logging.DEBUG,
            "urllib3.connectionpool": logging.WARNING,
            "posthog": logging.WARNING,
        }

        # 各ロガーが正しいレベルで設定されたことを確認
        for logger_name, expected_level in expected_logger_configs.items():
            mock_loggers[logger_name].setLevel.assert_called_with(expected_level)

        # configure_openai_loggingが呼び出されたことを確認
        mock_configure.assert_called_once_with(logging.WARNING)

    @patch("logging.getLogger")
    def test_setup_debug_logging_logger_call_count(self, mock_get_logger):
        """ロガー取得の呼び出し回数テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch("utilities.logging_config.configure_openai_logging"):
            setup_debug_logging_without_openai_http()

        # 4つのロガーが取得されることを確認
        assert mock_get_logger.call_count == 4

        expected_logger_names = [
            "services.llm.components",
            "services.llm.generators.openai",
            "urllib3.connectionpool",
            "posthog",
        ]

        actual_logger_names = [call[0][0] for call in mock_get_logger.call_args_list]
        assert set(actual_logger_names) == set(expected_logger_names)


class TestLoggingConfigIntegration:
    """logging_config モジュールの統合テスト"""

    def test_import_functions(self):
        """関数のインポートテスト"""
        from utilities.logging_config import (
            configure_openai_logging,
            setup_debug_logging_without_openai_http,
        )

        assert callable(configure_openai_logging)
        assert callable(setup_debug_logging_without_openai_http)

    def test_logging_levels_constants(self):
        """ログレベル定数の利用可能性テスト"""
        # ログレベル定数が利用可能であることを確認
        assert hasattr(logging, "DEBUG")
        assert hasattr(logging, "INFO")
        assert hasattr(logging, "WARNING")
        assert hasattr(logging, "ERROR")
        assert hasattr(logging, "CRITICAL")

        # ログレベル値の確認
        assert (
            logging.DEBUG
            < logging.INFO
            < logging.WARNING
            < logging.ERROR
            < logging.CRITICAL
        )

    @patch("logging.getLogger")
    def test_function_parameters_validation(self, mock_get_logger):
        """関数パラメータの検証テスト"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # 各種ログレベルでテスト
        valid_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        for level in valid_levels:
            configure_openai_logging(level)
            # エラーが発生しないことを確認
            assert mock_logger.setLevel.called

    def test_module_docstring_examples(self):
        """モジュールのdocstring例の基本構文確認"""
        # インポート文が正しく動作することを確認
        from utilities.logging_config import setup_debug_logging_without_openai_http
        from utilities.logging_config import configure_openai_logging

        # 関数が存在することを確認
        assert callable(setup_debug_logging_without_openai_http)
        assert callable(configure_openai_logging)
