"""
OpenAI APIエラー時のキャッシュ削除機能を提供するユーティリティ
"""

import functools
import logging
from collections.abc import Callable

from services.cache.manager import get_cache_manager

logger = logging.getLogger(__name__)


def with_cache_error_cleanup(
    cache_key_func: Callable[..., str] | None = None, delete_on_any_error: bool = False
):
    """
    OpenAI APIエラー発生時にキャッシュエントリを削除するデコレータ

    Args:
        cache_key_func: キャッシュキーを生成する関数（引数から）
        step_type: キャッシュのステップタイプ
        delete_on_any_error: 全てのエラーでキャッシュ削除するか（デフォルト: OpenAIエラーのみ）
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            cache_key = None

            if cache_key_func:
                try:
                    cache_key = cache_key_func(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Failed to generate cache key for cleanup: {e}")

            try:
                # 元の関数を実行
                return func(*args, **kwargs)

            except Exception as error:
                # エラー発生時のキャッシュクリーンアップ
                if cache_key:
                    if delete_on_any_error:
                        # 全てのエラーでキャッシュ削除
                        deleted = cache_manager.delete(cache_key)
                        if deleted:
                            logger.info(
                                f"Deleted cache entry due to error: {cache_key[:16]}..."
                            )
                    else:
                        # OpenAIエラーのみでキャッシュ削除
                        cache_manager.delete_on_openai_error(cache_key, error)

                # 元のエラーを再発生
                raise

        return wrapper

    return decorator


def cleanup_cache_on_openai_error(
    cache_key: str, error: Exception, step_type: str | None = None
) -> bool:
    """
    OpenAI APIエラー発生時に特定のキャッシュエントリを削除

    Args:
        cache_key: 削除対象のキャッシュキー
        error: 発生したエラー
        step_type: キャッシュのステップタイプ（ログ用）

    Returns:
        削除が実行されたかどうか
    """
    if not cache_key:
        return False

    cache_manager = get_cache_manager()
    deleted = cache_manager.delete_on_openai_error(cache_key, error)

    if deleted and step_type:
        logger.info(
            f"Cleaned up {step_type} cache due to OpenAI error: {type(error).__name__}"
        )

    return deleted


def cleanup_related_cache_entries(
    component: str, step_type: str | None = None, reason: str = "API error"
) -> int:
    """
    特定のコンポーネントに関連するキャッシュエントリを削除

    Args:
        component: キャッシュキーに含まれるコンポーネント文字列
        step_type: 削除対象のステップタイプ
        reason: 削除理由（ログ用）

    Returns:
        削除されたエントリ数
    """
    cache_manager = get_cache_manager()
    deleted_count = cache_manager.delete_by_component(component, step_type)

    if deleted_count > 0:
        logger.info(
            f"Cleaned up {deleted_count} cache entries related to {component} due to {reason}"
        )

    return deleted_count


class CacheCleanupContext:
    """
    withブロック内でのエラー発生時にキャッシュクリーンアップを実行するコンテキストマネージャー
    """

    def __init__(
        self,
        cache_key: str | None = None,
        step_type: str | None = None,
        cleanup_component: str | None = None,
    ):
        self.cache_key = cache_key
        self.step_type = step_type
        self.cleanup_component = cleanup_component
        self.cache_manager = get_cache_manager()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # エラーが発生した場合
            cleaned_up = False

            # 特定のキャッシュキーを削除
            if self.cache_key:
                deleted = self.cache_manager.delete_on_openai_error(
                    self.cache_key, exc_val
                )
                if deleted:
                    cleaned_up = True
                    logger.info(
                        f"Cache cleanup: deleted key {self.cache_key[:16]}... due to {exc_type.__name__}"
                    )

            # コンポーネント関連のキャッシュを削除
            if self.cleanup_component:
                deleted_count = self.cache_manager.delete_by_component(
                    self.cleanup_component, self.step_type
                )
                if deleted_count > 0:
                    cleaned_up = True
                    logger.info(
                        f"Cache cleanup: deleted {deleted_count} entries for component {self.cleanup_component}"
                    )

            if not cleaned_up and (self.cache_key or self.cleanup_component):
                logger.debug(f"No cache cleanup needed for error: {exc_type.__name__}")

        return False
