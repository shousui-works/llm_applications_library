"""Google Cloud Storage utilities のテスト"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from google.cloud.exceptions import NotFound

from storage.gcs import GCSClient, File, get_gcs_client


class TestGCSClient:
    """GCSClientクラスのテスト"""

    @pytest.fixture
    def mock_storage_client(self):
        """google.cloud.storageのClientをモック化"""
        with patch("storage.gcs.storage.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def gcs_client(self, mock_storage_client):
        """テスト用GCSClientインスタンス"""
        return GCSClient(project_id="test-project")

    def test_initialization(self, mock_storage_client):
        """初期化テスト"""
        client = GCSClient(project_id="test-project")
        mock_storage_client.assert_called_once_with(project="test-project")
        assert client.client == mock_storage_client.return_value

    def test_initialization_without_project_id(self, mock_storage_client):
        """プロジェクトID指定なしの初期化テスト"""
        GCSClient()
        mock_storage_client.assert_called_once_with(project=None)

    @pytest.mark.parametrize(
        "gs_url,expected_bucket,expected_blob",
        [
            ("gs://bucket/file.txt", "bucket", "file.txt"),
            ("gs://my-bucket/path/to/file.csv", "my-bucket", "path/to/file.csv"),
            (
                "gs://bucket/folder/subfolder/document.pdf",
                "bucket",
                "folder/subfolder/document.pdf",
            ),
            ("gs://bucket/", "bucket", ""),
        ],
    )
    def test_parse_gs_url_success(
        self, gcs_client, gs_url, expected_bucket, expected_blob
    ):
        """gs:// URL解析の成功テスト"""
        bucket, blob = gcs_client.parse_gs_url(gs_url)
        assert bucket == expected_bucket
        assert blob == expected_blob

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "http://bucket/file.txt",
            "https://bucket/file.txt",
            "s3://bucket/file.txt",
            "bucket/file.txt",
            "",
        ],
    )
    def test_parse_gs_url_invalid(self, gcs_client, invalid_url):
        """gs:// URL解析の失敗テスト"""
        with pytest.raises(ValueError, match="Invalid GCS URL"):
            gcs_client.parse_gs_url(invalid_url)


class TestGCSClientDownload:
    """GCSClientのダウンロード機能テスト"""

    @pytest.fixture
    def mock_storage_client(self):
        with patch("storage.gcs.storage.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def gcs_client(self, mock_storage_client):
        return GCSClient(project_id="test-project")

    @pytest.fixture
    def mock_bucket_and_blob(self, mock_storage_client):
        """バケットとブロブのモック"""
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        return mock_bucket, mock_blob

    def test_download_file_to_memory_success(self, gcs_client, mock_bucket_and_blob):
        """メモリへのダウンロード成功テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        test_content = b"test file content"
        mock_blob.download_as_bytes.return_value = test_content

        result = gcs_client.download_file_to_memory("gs://bucket/file.txt")

        assert result == test_content
        mock_blob.download_as_bytes.assert_called_once()

    def test_download_file_to_memory_not_found(self, gcs_client, mock_bucket_and_blob):
        """メモリへのダウンロード - ファイル未存在テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        mock_blob.download_as_bytes.side_effect = NotFound("File not found")

        with pytest.raises(
            FileNotFoundError, match="File not found: gs://bucket/file.txt"
        ):
            gcs_client.download_file_to_memory("gs://bucket/file.txt")

    def test_download_file_to_local_success(
        self, gcs_client, mock_bucket_and_blob, tmp_path
    ):
        """ローカルファイルへのダウンロード成功テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        local_path = tmp_path / "downloaded_file.txt"

        gcs_client.download_file_to_local("gs://bucket/file.txt", local_path)

        mock_blob.download_to_filename.assert_called_once_with(str(local_path))
        assert local_path.parent.exists()  # 親ディレクトリが作成されていることを確認

    def test_download_file_to_local_creates_directory(
        self, gcs_client, mock_bucket_and_blob, tmp_path
    ):
        """ローカルダウンロード時のディレクトリ作成テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        local_path = tmp_path / "deep" / "nested" / "path" / "file.txt"

        gcs_client.download_file_to_local("gs://bucket/file.txt", local_path)

        assert local_path.parent.exists()
        mock_blob.download_to_filename.assert_called_once_with(str(local_path))


class TestGCSClientUpload:
    """GCSClientのアップロード機能テスト"""

    @pytest.fixture
    def mock_storage_client(self):
        with patch("storage.gcs.storage.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def gcs_client(self, mock_storage_client):
        return GCSClient(project_id="test-project")

    @pytest.fixture
    def mock_bucket_and_blob(self, mock_storage_client):
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        return mock_bucket, mock_blob

    def test_upload_text_to_gcs_success(self, gcs_client, mock_bucket_and_blob):
        """テキストのGCSアップロード成功テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        test_content = "Hello, World!"
        gs_url = "gs://bucket/file.txt"

        result = gcs_client.upload_text_to_gcs(test_content, gs_url)

        assert result == gs_url
        mock_blob.upload_from_string.assert_called_once_with(
            test_content, content_type="text/plain"
        )

    def test_upload_file_to_gcs_success(
        self, gcs_client, mock_bucket_and_blob, tmp_path
    ):
        """ファイルのGCSアップロード成功テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob

        # テストファイルを作成
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n")

        gs_url = "gs://bucket/data.csv"

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 100
            result = gcs_client.upload_file_to_gcs(str(test_file), gs_url)

        assert result == gs_url
        mock_blob.upload_from_filename.assert_called_once_with(
            str(test_file), content_type="text/csv"
        )

    @pytest.mark.parametrize(
        "file_extension,expected_content_type",
        [
            (".csv", "text/csv"),
            (".txt", "text/plain"),
            (
                ".xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            (".xls", "application/vnd.ms-excel"),
            (".unknown", "application/octet-stream"),
        ],
    )
    def test_upload_file_content_type_mapping(
        self,
        gcs_client,
        mock_bucket_and_blob,
        tmp_path,
        file_extension,
        expected_content_type,
    ):
        """ファイル拡張子による Content-Type マッピングテスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob

        test_file = tmp_path / f"test{file_extension}"
        test_file.write_text("content")

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 100
            gcs_client.upload_file_to_gcs(str(test_file), "gs://bucket/file")

        mock_blob.upload_from_filename.assert_called_once_with(
            str(test_file), content_type=expected_content_type
        )


class TestGCSClientRead:
    """GCSClientの読み取り機能テスト"""

    @pytest.fixture
    def mock_storage_client(self):
        with patch("storage.gcs.storage.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def gcs_client(self, mock_storage_client):
        return GCSClient(project_id="test-project")

    @pytest.fixture
    def mock_bucket_and_blob(self, mock_storage_client):
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        return mock_bucket, mock_blob

    def test_read_text_from_gcs_success(self, gcs_client, mock_bucket_and_blob):
        """GCSからのテキスト読み取り成功テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        test_content = "Hello, World!"
        mock_blob.download_as_text.return_value = test_content

        result = gcs_client.read_text_from_gcs("gs://bucket/file.txt")

        assert result == test_content
        mock_blob.download_as_text.assert_called_once()

    def test_read_csv_from_gcs_success(self, gcs_client, mock_bucket_and_blob):
        """GCSからのCSV読み取り成功テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        csv_content = b"name,age\nAlice,30\nBob,25"
        mock_blob.download_as_bytes.return_value = csv_content

        result = gcs_client.read_csv_from_gcs("gs://bucket/data.csv")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age"]
        assert result.iloc[0]["name"] == "Alice"

    def test_read_text_from_gcs_not_found(self, gcs_client, mock_bucket_and_blob):
        """GCSからのテキスト読み取り - ファイル未存在テスト"""
        mock_bucket, mock_blob = mock_bucket_and_blob
        mock_blob.download_as_text.side_effect = NotFound("File not found")

        with pytest.raises(
            FileNotFoundError, match="File not found: gs://bucket/file.txt"
        ):
            gcs_client.read_text_from_gcs("gs://bucket/file.txt")


class TestGCSClientListFiles:
    """GCSClientのファイル一覧取得機能テスト"""

    @pytest.fixture
    def mock_storage_client(self):
        with patch("storage.gcs.storage.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def gcs_client(self, mock_storage_client):
        return GCSClient(project_id="test-project")

    def test_list_files_in_directory_success(self, gcs_client, mock_storage_client):
        """ディレクトリ内ファイル一覧取得成功テスト"""
        # モックブロブを作成
        mock_blob1 = Mock()
        mock_blob1.name = "folder/file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "folder/file2.csv"
        mock_blob3 = Mock()
        mock_blob3.name = "folder/subfolder/"  # ディレクトリ（スキップされるべき）

        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        result = gcs_client.list_files_in_directory("gs://bucket/folder")

        assert len(result) == 2
        assert isinstance(result[0], File)
        assert result[0].name == "file1.txt"
        assert result[0].path == "gs://bucket/folder/file1.txt"
        assert result[1].name == "file2.csv"
        assert result[1].path == "gs://bucket/folder/file2.csv"

    def test_list_files_adds_trailing_slash(self, gcs_client, mock_storage_client):
        """ディレクトリURLに末尾スラッシュが追加されるテスト"""
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = []
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        gcs_client.list_files_in_directory("gs://bucket/folder")

        mock_bucket.list_blobs.assert_called_once_with(prefix="folder/")

    def test_list_files_directory_not_found(self, gcs_client, mock_storage_client):
        """ディレクトリ未存在時のテスト"""
        mock_bucket = Mock()
        mock_bucket.list_blobs.side_effect = NotFound("Directory not found")
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        result = gcs_client.list_files_in_directory("gs://bucket/nonexistent")

        assert result == []


class TestFactoryFunction:
    """ファクトリ関数のテスト"""

    @patch("storage.gcs.GCSClient")
    def test_get_gcs_client(self, mock_gcs_client):
        """get_gcs_client関数のテスト"""
        result = get_gcs_client(project_id="test-project")

        mock_gcs_client.assert_called_once_with(project_id="test-project")
        assert result == mock_gcs_client.return_value

    @patch("storage.gcs.GCSClient")
    def test_get_gcs_client_no_project_id(self, mock_gcs_client):
        """get_gcs_client関数（プロジェクトID指定なし）のテスト"""
        result = get_gcs_client()

        mock_gcs_client.assert_called_once_with(project_id=None)
        assert result == mock_gcs_client.return_value


class TestFileDataclass:
    """Fileデータクラスのテスト"""

    def test_file_creation(self):
        """Fileオブジェクト作成テスト"""
        file_obj = File(id=1, name="test.txt", path="gs://bucket/test.txt")

        assert file_obj.id == 1
        assert file_obj.name == "test.txt"
        assert file_obj.path == "gs://bucket/test.txt"

    def test_file_equality(self):
        """Fileオブジェクトの等価性テスト"""
        file1 = File(id=1, name="test.txt", path="gs://bucket/test.txt")
        file2 = File(id=1, name="test.txt", path="gs://bucket/test.txt")
        file3 = File(id=2, name="test.txt", path="gs://bucket/test.txt")

        assert file1 == file2
        assert file1 != file3
