import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

# get_cache_manager will be imported locally to avoid circular imports


def generate_file_hash(file_path: Path) -> str:
    """ファイルの内容とメタデータからハッシュを生成"""
    if not file_path.exists():
        return ""

    # ファイルの更新時刻とサイズを取得
    stat = file_path.stat()
    file_info = {
        "path": str(file_path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }

    # 小さなファイルの場合は内容も含める
    if stat.st_size < 1024 * 1024:  # 1MB未満
        try:
            with open(file_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()
                file_info["content_hash"] = content_hash
        except Exception as e:
            # ファイル読み込みエラーの場合はスキップ
            logging.getLogger(__name__).debug(f"Failed to read file for hashing: {e}")

    return hashlib.sha256(json.dumps(file_info, sort_keys=True).encode()).hexdigest()


def generate_directory_hash(directory_path: Path) -> str:
    """ディレクトリ内のファイル群からハッシュを生成"""
    if not directory_path.exists():
        return ""

    file_hashes = []
    for root, _, files in os.walk(directory_path):
        for file in sorted(files):
            file_path = Path(root) / file
            file_hash = generate_file_hash(file_path)
            file_hashes.append(f"{file}:{file_hash}")

    return hashlib.sha256("\n".join(file_hashes).encode()).hexdigest()


def generate_file_list_content_hash(file_list) -> str:
    """
    ファイルリストからコンテンツベースのハッシュを生成
    ファイル名のみを使用して、ローカル/GCSの状態に依存しない安定したハッシュを生成

    Args:
        file_list: ファイルオブジェクトのリスト

    Returns:
        str: ファイルリストのコンテンツハッシュ
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Generating stable content hash for {len(file_list)} files")

    try:
        file_signatures = []

        for file_obj in file_list:
            if hasattr(file_obj, "name"):
                file_name = file_obj.name

                # ファイル名のみを使用して一貫性を保証
                # サイズやメタデータはローカル/GCSの状態によって異なるため除外
                file_signature = {
                    "name": file_name,
                    "type": "file",  # 一定の構造を保持
                }

                file_signatures.append(file_signature)

        # ファイル署名をソートして一意性を保証
        file_signatures.sort(key=lambda x: x.get("name", ""))

        # ハッシュを生成
        signatures_json = json.dumps(file_signatures, sort_keys=True)
        content_hash = hashlib.sha256(signatures_json.encode()).hexdigest()
        logger.debug(
            f"Generated stable content hash: {content_hash[:16]}... from {len(file_signatures)} file signatures"
        )
        return content_hash

    except Exception as e:
        logger.error(f"Error generating file list content hash: {e}")
        import traceback

        logger.error(f"Stack trace: {traceback.format_exc()}")

        # エラーの場合はファイル名のみでフォールバック
        fallback_names = []
        for file_obj in file_list:
            if hasattr(file_obj, "name"):
                fallback_names.append(file_obj.name)
        fallback_names.sort()

        # フォールバックでも同じ構造を使用
        fallback_signatures = [
            {"name": name, "type": "file"} for name in fallback_names
        ]
        fallback_json = json.dumps(fallback_signatures, sort_keys=True)
        fallback_hash = hashlib.sha256(fallback_json.encode()).hexdigest()
        logger.warning(
            f"Using emergency fallback hash: {fallback_hash[:16]}... based on {len(fallback_names)} filenames"
        )
        return fallback_hash


def generate_gcs_directory_content_hash(gcs_directory_url: str) -> str:
    """
    GCSディレクトリの内容に基づいてハッシュを生成
    パスではなく、含まれるファイルの情報を使用

    Args:
        gcs_directory_url: GCSディレクトリURL

    Returns:
        str: ディレクトリ内容のハッシュ
    """
    try:
        from utils.gcs import get_gcs_client

        gcs_client = get_gcs_client()
        file_list = gcs_client.list_files_in_directory(
            gcs_directory_url, use_cache=False
        )
        return generate_file_list_content_hash(file_list)
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Error generating GCS directory content hash: {e}"
        )
        # エラーの場合はURLベースのハッシュにフォールバック
        return hashlib.sha256(gcs_directory_url.encode()).hexdigest()


def normalize_data_for_hash(data: Any) -> str:
    """データを正規化してハッシュ生成用の文字列に変換"""
    if isinstance(data, dict):
        # 辞書は キーをソートして正規化
        normalized = {}
        for key, value in data.items():
            normalized[key] = normalize_data_for_hash(value)
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    if isinstance(data, list):
        # リストは各要素を正規化
        return json.dumps(
            [normalize_data_for_hash(item) for item in data], ensure_ascii=False
        )
    if hasattr(data, "model_dump"):
        # Pydanticモデルの場合
        return normalize_data_for_hash(data.model_dump())
    # その他の型はそのままJSON化
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)


def generate_cache_key(*components: Any) -> str:
    """複数のコンポーネントからキャッシュキーを生成"""
    normalized_components = []

    for component in components:
        if isinstance(component, Path):
            # パスの場合はファイルハッシュを生成
            if component.is_file():
                normalized_components.append(generate_file_hash(component))
            elif component.is_dir():
                normalized_components.append(generate_directory_hash(component))
            else:
                normalized_components.append(str(component))
        elif isinstance(component, str) and component.startswith("gs://"):
            # GCS URLがディレクトリかファイルかを判定
            if component.endswith("/") or ("." not in component.split("/")[-1]):
                # ディレクトリの場合(最後が/で終わるか、最後の部分に拡張子がない)
                normalized_components.append(
                    generate_gcs_directory_content_hash(component)
                )
            else:
                # 個別ファイルの場合はそのままURL文字列を使用
                normalized_components.append(normalize_data_for_hash(component))
        elif isinstance(component, list) and len(component) > 0:
            # ファイルリストの可能性をチェック
            first_item = component[0]
            if hasattr(first_item, "name") and hasattr(first_item, "path"):
                # Fileオブジェクトのリストの場合はコンテンツハッシュを使用
                normalized_components.append(generate_file_list_content_hash(component))
            else:
                # その他のリストは通常の正規化
                normalized_components.append(normalize_data_for_hash(component))
        else:
            # その他のデータは正規化
            normalized_components.append(normalize_data_for_hash(component))

    # 全てのコンポーネントを結合してハッシュ化
    combined_data = "|".join(normalized_components)
    return hashlib.sha256(combined_data.encode()).hexdigest()


def validate_cache_key(key: str) -> bool:
    """キャッシュキーの有効性を検証"""
    if not key or not isinstance(key, str):
        return False

    # SHA256ハッシュの長さをチェック(64文字)
    if len(key) != 64:
        return False

    # 16進数文字のみかチェック
    try:
        int(key, 16)
    except ValueError:
        return False
    else:
        return True


class CacheKeyBuilder:
    """キャッシュキー生成のヘルパークラス"""

    def __init__(self):
        self.components = []

    def add_component(self, component: Any) -> "CacheKeyBuilder":
        """コンポーネントを追加"""
        self.components.append(component)
        return self

    def add_file_list(self, file_list) -> "CacheKeyBuilder":
        """ファイルリストを追加（コンテンツベースのハッシュを使用）"""
        if isinstance(file_list, list) and len(file_list) > 0:
            first_item = file_list[0]
            if hasattr(first_item, "name") and hasattr(first_item, "path"):
                # Fileオブジェクトのリストの場合は専用ハッシュを使用
                content_hash = generate_file_list_content_hash(file_list)
                self.components.append(content_hash)
            else:
                self.components.append(file_list)
        else:
            self.components.append(file_list)
        return self

    def add_file(self, file_path: Path) -> "CacheKeyBuilder":
        """ファイルを追加"""
        return self.add_component(file_path)

    def add_directory(self, dir_path: Path) -> "CacheKeyBuilder":
        """ディレクトリを追加"""
        return self.add_component(dir_path)

    def add_config(self, config: dict[str, Any]) -> "CacheKeyBuilder":
        """設定を追加"""
        return self.add_component(config)

    def add_list(self, items: list[Any]) -> "CacheKeyBuilder":
        """リストを追加"""
        return self.add_component(items)

    def build(self) -> str:
        """キャッシュキーを構築"""
        logger = logging.getLogger(__name__)
        cache_key = generate_cache_key(*self.components)
        logger.debug(
            f"Generated cache key: {cache_key[:16]}... from {len(self.components)} components"
        )
        return cache_key

    def clear(self) -> "CacheKeyBuilder":
        """コンポーネントをクリア"""
        self.components.clear()
        return self


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
        # Use lazy import to avoid circular dependency
        from services.cache.manager import get_cache_manager

        self.cache_manager = get_cache_manager()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # エラーが発生した場合
            logger = logging.getLogger(__name__)
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
