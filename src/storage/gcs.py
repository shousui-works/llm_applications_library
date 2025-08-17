"""
Google Cloud Storage utilities for Due Diligence
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from google.cloud import storage
from google.cloud.exceptions import NotFound


logger = logging.getLogger(__name__)


@dataclass
class File:
    """ファイル情報を表すデータクラス"""

    id: int
    name: str
    path: str


class GCSClient:
    """Google Cloud Storage client wrapper"""

    def __init__(self, project_id: str | None = None):
        """
        Initialize GCS client

        Args:
            project_id: Google Cloud project ID. If None, uses default from environment
        """
        self.client = storage.Client(project=project_id)
        logger.info(f"Initialized GCS client for project: {self.client.project}")

    def parse_gs_url(self, gs_url: str) -> tuple[str, str]:
        """
        Parse gs:// URL into bucket and blob name

        Args:
            gs_url: GCS URL like gs://bucket/path/to/file

        Returns:
            Tuple of (bucket_name, blob_name)
        """
        parsed = urlparse(gs_url)
        if parsed.scheme != "gs":
            raise ValueError(f"Invalid GCS URL: {gs_url}. Must start with gs://")

        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        return bucket_name, blob_name

    def download_file_to_memory(self, gs_url: str) -> bytes:
        """
        Download file from GCS to memory

        Args:
            gs_url: GCS URL like gs://bucket/path/to/file

        Returns:
            File content as bytes
        """
        bucket_name, blob_name = self.parse_gs_url(gs_url)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            logger.info(f"Downloading {gs_url}...")
            content = blob.download_as_bytes()
            logger.info(f"Downloaded {len(content)} bytes from {gs_url}")

            return content

        except NotFound:
            raise FileNotFoundError(f"File not found: {gs_url}")
        except Exception as e:
            logger.error(f"Error downloading {gs_url}: {e}")
            raise

    def download_file_to_local(self, gs_url: str, local_path: Path) -> None:
        """
        Download file from GCS to local path

        Args:
            gs_url: GCS URL like gs://bucket/path/to/file
            local_path: Local file path to save to
        """
        bucket_name, blob_name = self.parse_gs_url(gs_url)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {gs_url} to {local_path}...")
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded {gs_url} to {local_path}")

        except NotFound:
            raise FileNotFoundError(f"File not found: {gs_url}")
        except Exception as e:
            logger.error(f"Error downloading {gs_url}: {e}")
            raise

    def list_files_in_directory(self, gs_directory_url: str) -> list[File]:
        """
        List all files in a GCS directory

        Args:
            gs_directory_url: GCS directory URL like gs://bucket/path/to/directory/
            use_cache: キャッシュを使用するかどうか

        Returns:
            List of File objects
        """
        # キャッシュミスまたはキャッシュ無効の場合、GCSから取得
        bucket_name, prefix = self.parse_gs_url(gs_directory_url)

        # Ensure prefix ends with / for directory listing
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)

            file_list = []
            file_id = 0

            for blob in blobs:
                # Skip directories (blobs ending with /)
                if blob.name.endswith("/"):
                    continue

                # Create File object
                file_name = Path(blob.name).name
                gs_path = f"gs://{bucket_name}/{blob.name}"

                file_obj = File(
                    id=file_id,
                    name=file_name,
                    path=gs_path,  # Store GCS path instead of local path
                )

                file_list.append(file_obj)
                file_id += 1

            logger.info(f"Found {len(file_list)} files in {gs_directory_url}")

            return file_list

        except NotFound:
            logger.warning(f"Directory not found: {gs_directory_url}")
            return []
        except Exception as e:
            logger.error(f"Error listing files in {gs_directory_url}: {e}")
            raise

    def read_csv_from_gcs(self, gs_url: str) -> pd.DataFrame:
        """
        Read CSV file from GCS directly into pandas DataFrame

        Args:
            gs_url: GCS URL like gs://bucket/path/to/file.csv

        Returns:
            Pandas DataFrame
        """
        try:
            content = self.download_file_to_memory(gs_url)

            # Read CSV from bytes
            csv_buffer = io.StringIO(content.decode("utf-8"))
            df = pd.read_csv(csv_buffer)

            logger.info(f"Read CSV with {len(df)} rows from {gs_url}")
            return df

        except Exception as e:
            logger.error(f"Error reading CSV from {gs_url}: {e}")
            raise

    def upload_text_to_gcs(self, content: str, gs_url: str) -> str:
        """
        Upload text content to GCS

        Args:
            content: Text content to upload
            gs_url: GCS URL like gs://bucket/path/to/file.txt

        Returns:
            GCS URL of uploaded file
        """
        bucket_name, blob_name = self.parse_gs_url(gs_url)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            logger.info(f"Uploading text content to {gs_url}...")
            blob.upload_from_string(content, content_type="text/plain")
            logger.info(f"Uploaded {len(content)} characters to {gs_url}")

            return gs_url

        except Exception as e:
            logger.error(f"Error uploading to {gs_url}: {e}")
            raise

    def upload_file_to_gcs(self, local_file_path: str, gs_url: str) -> str:
        """
        Upload a local file to GCS

        Args:
            local_file_path: Path to local file
            gs_url: GCS URL like gs://bucket/path/to/file

        Returns:
            GCS URL of uploaded file
        """
        bucket_name, blob_name = self.parse_gs_url(gs_url)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Determine content type based on file extension
            file_extension = Path(local_file_path).suffix.lower()
            content_type_map = {
                ".csv": "text/csv",
                ".txt": "text/plain",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls": "application/vnd.ms-excel",
            }
            content_type = content_type_map.get(
                file_extension, "application/octet-stream"
            )

            logger.info(f"Uploading file {local_file_path} to {gs_url}...")
            blob.upload_from_filename(local_file_path, content_type=content_type)

            file_size = Path(local_file_path).stat().st_size
            logger.info(f"Uploaded {file_size} bytes to {gs_url}")

            return gs_url

        except Exception as e:
            logger.error(f"Error uploading file {local_file_path} to {gs_url}: {e}")
            raise

    def read_text_from_gcs(self, gs_url: str, use_cache: bool = True) -> str:  # noqa: ARG002
        """
        Read text content from GCS

        Args:
            gs_url: GCS URL like gs://bucket/path/to/file.txt
            use_cache: キャッシュを使用するかどうか

        Returns:
            Text content as string
        """

        # キャッシュミスまたはキャッシュ無効の場合、GCSから読み込み
        bucket_name, blob_name = self.parse_gs_url(gs_url)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            logger.info(f"Reading text content from {gs_url}...")
            content = blob.download_as_text()
            logger.info(f"Read {len(content)} characters from {gs_url}")

            return content

        except NotFound:
            raise FileNotFoundError(f"File not found: {gs_url}")
        except Exception as e:
            logger.error(f"Error reading from {gs_url}: {e}")
            raise


def get_gcs_client(project_id: str | None = None) -> GCSClient:
    """
    GCSClientのファクトリ関数

    Args:
        project_id: Google Cloud project ID. If None, uses default from environment

    Returns:
        GCSClientインスタンス
    """
    return GCSClient(project_id=project_id)
