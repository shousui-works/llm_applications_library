import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from ..storage.gcs import GCSClient


def _get_local_pdf_path(pdf_path):
    """
    PDFファイルのローカルパスを取得する
    GCS URLの場合は一時ディレクトリにダウンロードする

    Args:
        pdf_path (str | Path): PDFファイルのパス（ローカルまたはGCS URL）

    Returns:
        str: ローカルファイルパス
        bool: 一時ファイルかどうか（削除が必要）
    """
    # ローカルファイルパスの場合は、そのまま使用
    if not (isinstance(pdf_path, str) and pdf_path.startswith("gs://")):
        local_path = str(pdf_path)
        # ファイルの存在確認
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        return local_path, False

    # GCS URLの場合、一時ファイルにダウンロード

    gcs_client = GCSClient()

    # 一時ファイルを作成
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    # GCSからダウンロード
    gcs_client.download_file_to_local(pdf_path, temp_path)

    return str(temp_path), True


def extract_pdf_text(pdf_path, use_cache=True):
    """
    PDFファイルからテキストを抽出する
    ローカルファイルとGCS URLの両方に対応

    Args:
        pdf_path (str | Path): PDFファイルのパス（ローカルまたはGCS URL）
        use_cache (bool): キャッシュを使用するかどうか

    Returns:
        str: 抽出されたテキスト
    """
    # キャッシュミスまたはキャッシュ無効の場合、PDFを処理
    local_path, is_temp = _get_local_pdf_path(pdf_path)

    try:
        text_content = ""
        with fitz.open(local_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"--- Page {page_num + 1} ---\n"
                    text_content += page_text + "\n\n"

        return text_content
    finally:
        # 一時ファイルの場合は削除
        if is_temp:
            try:
                Path(local_path).unlink()
            except OSError:
                pass  # ファイル削除に失敗しても処理を続行


def is_pdf_text_based(pdf_path, use_cache=True):
    """
    PDFがテキストベースかどうかを判定する
    ローカルファイルとGCS URLの両方に対応

    Args:
        pdf_path (str | Path): PDFファイルのパス（ローカルまたはGCS URL）
        use_cache (bool): キャッシュを使用するかどうか

    Returns:
        bool: テキストベースの場合True
    """
    # キャッシュミスまたはキャッシュ無効の場合、PDFを処理
    local_path, is_temp = _get_local_pdf_path(pdf_path)

    try:
        with fitz.open(local_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    return True  # テキストが存在する

    finally:
        # 一時ファイルの場合は削除
        if is_temp:
            try:
                Path(local_path).unlink()
            except OSError:
                pass  # ファイル削除に失敗しても処理を続行
