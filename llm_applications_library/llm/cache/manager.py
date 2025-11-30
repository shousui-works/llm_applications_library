import logging
import time
from pathlib import Path
from typing import Any

import diskcache
from pydantic import BaseModel

from .utils import validate_cache_key

logger = logging.getLogger(__name__)


class CacheStats(BaseModel):
    """キャッシュ統計情報"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_mb: float = 0.0
    total_keys: int = 0

    @property
    def hit_rate(self) -> float:
        """ヒット率を計算"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class CacheEntry(BaseModel):
    """キャッシュエントリ"""

    data: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    step_type: str = ''


class CacheManager:
    """ドキュメント解析ワークフロー用のキャッシュ管理システム"""

    def __init__(
        self,
        cache_dir: str = '.cache',
        max_size_mb: int = 1000,
        default_ttl: int = 3600 * 24,  # 24時間
        step_ttls: dict[str, int] | None = None,
    ):
        """
        キャッシュマネージャーを初期化

        Args:
            cache_dir: キャッシュディレクトリ
            max_size_mb: 最大キャッシュサイズ(MB)
            default_ttl: デフォルトTTL(秒)
            step_ttls: ステップ別TTL設定
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # diskcacheインスタンスを作成
        self.cache = diskcache.Cache(
            str(self.cache_dir),
            size_limit=max_size_mb * 1024 * 1024,  # バイトに変換
            eviction_policy='least-recently-used',
        )

        self.default_ttl = default_ttl
        self.step_ttls = step_ttls or {
            'file_analysis': 3600 * 24 * 7,  # 7日間
            'file_selection': 3600 * 24 * 3,  # 3日間
            'info_extraction': 3600 * 24 * 7,  # 7日間
            'report_generation': 3600 * 24,  # 1日間
            'pdf_text_extraction': 3600 * 24 * 30,  # 30日間(PDFテキストは変わらないため長期)
            'pdf_text_check': 3600 * 24 * 30,  # 30日間(PDFテキスト有無も変わらないため長期)
            'gcs_file_listing': 3600 * 24 * 7,  # 7日間(ファイルリストは変わる可能性があるため中期)
            'gcs_text_read': 3600 * 24 * 30,  # 30日間(GCSテキストファイルは変わらないため長期)
        }

        # 統計情報
        self.stats = CacheStats()

    def _get_ttl(self, step_type: str) -> int:
        """ステップタイプに応じたTTLを取得"""
        return self.step_ttls.get(step_type, self.default_ttl)

    def _create_cache_entry(self, data: Any, step_type: str = '') -> CacheEntry:
        """キャッシュエントリを作成"""
        now = time.time()
        return CacheEntry(
            data=data,
            created_at=now,
            last_accessed=now,
            access_count=1,
            step_type=step_type,
        )

    def get(self, key: str, step_type: str = '') -> tuple[Any | None, bool]:
        """
        キャッシュからデータを取得

        Args:
            key: キャッシュキー
            step_type: ステップタイプ

        Returns:
            (データ, キャッシュヒットフラグ)
        """
        if not validate_cache_key(key):
            logger.warning(f'Invalid cache key: {key}')
            self.stats.misses += 1
            return None, False

        try:
            entry_data = self.cache.get(key)
            if entry_data is None:
                logger.debug(f'Cache miss for key: {key[:16]}... (step: {step_type})')
                self.stats.misses += 1
                return None, False

            # エントリを復元
            entry = CacheEntry.model_validate(entry_data)

            # TTLチェック
            ttl = self._get_ttl(step_type or entry.step_type)
            if time.time() - entry.created_at > ttl:
                logger.debug(f'Cache entry expired for key: {key[:16]}...')
                self.cache.delete(key)
                self.stats.misses += 1
                return None, False

            # アクセス情報を更新
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.cache.set(key, entry.model_dump())

            logger.debug(f'Cache hit for key: {key[:16]}... (step: {step_type})')
            self.stats.hits += 1

        except Exception:
            logger.exception('Error getting cache entry')
            self.stats.misses += 1
            return None, False
        else:
            return entry.data, True

    def set(self, key: str, data: Any, step_type: str = '') -> bool:
        """
        データをキャッシュに保存

        Args:
            key: キャッシュキー
            data: 保存するデータ
            step_type: ステップタイプ

        Returns:
            保存成功フラグ
        """
        if not validate_cache_key(key):
            logger.warning(f'Invalid cache key: {key}')
            return False

        try:
            entry = self._create_cache_entry(data, step_type)
            self.cache.set(key, entry.model_dump())

            logger.debug(f'Cache set for key: {key[:16]}... (step: {step_type})')

        except Exception:
            logger.exception('Error setting cache entry')
            return False
        else:
            return True

    def delete(self, key: str) -> bool:
        """
        キャッシュエントリを削除

        Args:
            key: キャッシュキー

        Returns:
            削除成功フラグ
        """
        if not validate_cache_key(key):
            return False

        try:
            result = self.cache.delete(key)
            if result:
                logger.debug(f'Cache deleted for key: {key[:16]}...')
        except Exception:
            logger.exception('Error deleting cache entry')
            return False
        else:
            return result

    def clear(self, step_type: str | None = None) -> int:
        """
        キャッシュをクリア

        Args:
            step_type: 特定のステップタイプのみクリアする場合

        Returns:
            削除されたエントリ数
        """
        try:
            if step_type is None:
                # 全てクリア
                count = len(self.cache)
                self.cache.clear()
                logger.info(f'Cleared all cache entries: {count}')
                return count
            # 特定のステップタイプのみクリア
            count = 0
            keys_to_delete = []

            for key in self.cache:
                try:
                    entry_data = self.cache.get(key)
                    if entry_data:
                        entry = CacheEntry.model_validate(entry_data)
                        if entry.step_type == step_type:
                            keys_to_delete.append(key)
                except Exception:
                    logger.debug(f'Failed to process cache entry {key}')
                    continue

            for key in keys_to_delete:
                if self.cache.delete(key):
                    count += 1

            logger.info(f'Cleared {count} cache entries for step type: {step_type}')

        except Exception:
            logger.exception('Error clearing cache')
            return 0
        else:
            return count

    def get_stats(self) -> CacheStats:
        """キャッシュ統計情報を取得"""
        try:
            # diskcacheの統計を更新
            self.stats.total_keys = len(self.cache)
            self.stats.size_mb = self.cache.volume() / (1024 * 1024)
        except Exception:
            logger.exception('Error getting cache stats')
        return self.stats

    def cleanup_expired(self) -> int:
        """期限切れエントリをクリーンアップ"""
        try:
            count = 0
            keys_to_delete = []
            current_time = time.time()

            for key in self.cache:
                try:
                    entry_data = self.cache.get(key)
                    if entry_data:
                        entry = CacheEntry.model_validate(entry_data)
                        ttl = self._get_ttl(entry.step_type)
                        if current_time - entry.created_at > ttl:
                            keys_to_delete.append(key)
                except Exception:
                    # 破損したエントリも削除
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                if self.cache.delete(key):
                    count += 1

            if count > 0:
                logger.info(f'Cleaned up {count} expired cache entries')

        except Exception:
            logger.exception('Error during cache cleanup')
            return 0
        else:
            return count

    def delete_on_openai_error(self, cache_key: str, error: Exception) -> bool:
        """
        OpenAI APIエラー発生時にキャッシュエントリを削除

        Args:
            cache_key: 削除対象のキャッシュキー
            error: 発生したOpenAIエラー

        Returns:
            削除成功フラグ
        """
        if not cache_key:
            return False

        # OpenAI APIエラーの場合にキャッシュを削除
        openai_error_types = [
            'RateLimitError',
            'APITimeoutError',
            'InternalServerError',
            'APIConnectionError',
            'BadRequestError',
            'AuthenticationError',
            'APIError',
            'OpenAIError',
        ]

        error_type = type(error).__name__
        if error_type in openai_error_types or 'openai' in str(error).lower():
            deleted = self.delete(cache_key)
            if deleted:
                logger.info(f'Deleted cache entry due to OpenAI error ({error_type}): {cache_key[:16]}...')
            return deleted

        return False

    def delete_by_component(self, component_substring: str, step_type: str = None) -> int:
        """
        特定のコンポーネントを含むキャッシュエントリを削除

        Args:
            component_substring: キャッシュキーに含まれるべき文字列
            step_type: 削除対象のステップタイプ（Noneの場合は全ステップ）

        Returns:
            削除されたエントリ数
        """
        deleted_count = 0

        try:
            # 全キーを取得して条件に一致するものを削除
            all_keys = list(self.cache.keys())

            for key in all_keys:
                try:
                    entry = self.cache.get(key)
                    if entry and isinstance(entry, CacheEntry):
                        # ステップタイプフィルター
                        if step_type and entry.step_type != step_type:
                            continue

                        # コンポーネント文字列が含まれているかチェック
                        if component_substring in key and self.delete(key):
                            deleted_count += 1
                            logger.debug(f'Deleted cache entry matching component: {key[:16]}...')
                except Exception as e:
                    logger.warning(f'Error checking cache key {key[:16]}...: {e}')
                    continue

        except Exception as e:
            logger.error(f'Error during component-based cache deletion: {e}')

        if deleted_count > 0:
            logger.info(f'Deleted {deleted_count} cache entries matching component: {component_substring}')

        return deleted_count

    def warm_cache(self, keys_and_data: dict[str, tuple[Any, str]]) -> int:
        """
        キャッシュのウォーミング

        Args:
            keys_and_data: {キー: (データ, ステップタイプ)}の辞書

        Returns:
            ウォーミングされたエントリ数
        """
        count = 0
        for key, (data, step_type) in keys_and_data.items():
            if self.set(key, data, step_type):
                count += 1

        logger.info(f'Warmed cache with {count} entries')
        return count

    def close(self):
        """キャッシュを閉じる"""
        try:
            self.cache.close()
            logger.info('Cache closed successfully')
        except Exception:
            logger.exception('Error closing cache')


class CacheManagerSingleton:
    """キャッシュマネージャーのシングルトンパターン"""

    _instance: CacheManager | None = None

    @classmethod
    def get_instance(cls) -> CacheManager:
        """キャッシュマネージャーインスタンスを取得"""
        if cls._instance is None:
            cls._instance = CacheManager()
        return cls._instance

    @classmethod
    def initialize(cls, **kwargs) -> CacheManager:
        """キャッシュマネージャーを初期化"""
        cls._instance = CacheManager(**kwargs)
        return cls._instance


def get_cache_manager() -> CacheManager:
    """グローバルキャッシュマネージャーを取得"""
    return CacheManagerSingleton.get_instance()


def initialize_cache_manager(**kwargs) -> CacheManager:
    """キャッシュマネージャーを初期化"""
    return CacheManagerSingleton.initialize(**kwargs)
