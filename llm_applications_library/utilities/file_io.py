from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    YAMLファイルを読み込んで辞書として返す

    Args:
        file_path (str | Path): YAMLファイルのパス

    Returns:
        Dict[str, Any]: YAMLファイルの内容

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        yaml.YAMLError: YAML形式が不正な場合
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError

    with open(file_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_text(file_path: str | Path, encoding: str = "utf-8") -> str:
    """
    テキストファイルを読み込んで文字列として返す

    Args:
        file_path (str | Path): テキストファイルのパス
        encoding (str): ファイルエンコーディング (default: utf-8)

    Returns:
        str: ファイルの内容

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        UnicodeDecodeError: エンコーディングエラーの場合
    """
    file_path = Path(file_path)

    with open(file_path, encoding=encoding) as file:
        return file.read()


def save_text(file_path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """
    テキストをファイルに保存する

    Args:
        file_path (str | Path): 保存先ファイルのパス
        content (str): 保存するテキスト内容
        encoding (str): ファイルエンコーディング (default: utf-8)

    Raises:
        OSError: ファイル書き込みエラーの場合
    """
    file_path = Path(file_path)

    # ディレクトリが存在しない場合は作成
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=encoding) as file:
        file.write(content)
