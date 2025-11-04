"""
ログ設定ユーティリティ
"""

import logging


def configure_openai_logging(level: int = logging.INFO):
    """OpenAIライブラリのログレベルを設定"""
    # OpenAI内部のHTTPリクエストログを制御
    logging.getLogger("openai._base_client").setLevel(level)
    logging.getLogger("openai").setLevel(level)

    # httpxライブラリ (OpenAIが使用) のログも制御
    logging.getLogger("httpx").setLevel(level)


def setup_debug_logging_without_openai_http():
    """デバッグログを有効にしつつ、OpenAIのHTTPログは無効化"""
    # アプリケーションのデバッグログを有効化
    logging.getLogger("services.llm.components").setLevel(logging.DEBUG)
    logging.getLogger("services.llm.generators.openai").setLevel(logging.DEBUG)

    # OpenAIのHTTPリクエストログは無効化
    configure_openai_logging(logging.WARNING)

    # urllib3とPostHogのログを無効化
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("posthog").setLevel(logging.WARNING)


# 使用例をコメントで記載
"""
使用方法:

# デバッグログは有効、OpenAI HTTPログは無効
from utils.logging_config import setup_debug_logging_without_openai_http
setup_debug_logging_without_openai_http()

# または個別に制御
from utils.logging_config import configure_openai_logging
configure_openai_logging(logging.WARNING)  # HTTPログを無効化
"""
