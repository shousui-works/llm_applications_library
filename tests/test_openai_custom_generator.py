"""OpenAI Custom Generator の実用的なテスト"""

from unittest.mock import Mock, patch
from llm_applications_library.llm.generators.openai_custom_generator import (
    OpenAIVisionGenerator,
    RetryOpenAIGenerator,
)
from llm_applications_library.llm.generators.schema import RetryConfig


class TestOpenAIVisionGenerator:
    """OpenAIVisionGeneratorの実用的なテスト"""

    def test_initialization(self):
        """初期化テスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")
        assert generator.model == "gpt-4o"
        assert generator.api_key == "test-key"

    def test_initialization_with_env_var(self):
        """環境変数からのAPI key取得テスト"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            generator = OpenAIVisionGenerator(model="gpt-4o")
            assert generator.api_key == "env-key"

    def test_chat_completion_success(self):
        """_chat_completionメソッドの成功テスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        # OpenAI レスポンスのモック
        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_response.usage.model_dump.return_value = {"total_tokens": 100}

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            messages = [{"role": "user", "content": "test"}]
            result = generator._chat_completion(messages)

            assert result["success"] is True
            assert result["content"] == "Test response"
            assert result["usage"]["total_tokens"] == 100

    def test_chat_completion_failure(self):
        """_chat_completionメソッドの失敗テスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            messages = [{"role": "user", "content": "test"}]
            result = generator._chat_completion(messages)

            assert result["success"] is False
            assert result["content"] is None
            assert "API Error" in result["error"]

    def test_run_method(self):
        """runメソッドのテスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        expected_result = {
            "success": True,
            "content": "Image analysis result",
            "usage": {"total_tokens": 150},
            "error": None,
        }

        with patch.object(
            generator, "_chat_completion", return_value=expected_result
        ) as mock_chat:
            result = generator.run(
                base64_images=["test_base64_data"],
                mime_types=["image/png"],
                system_prompt="Analyze this image",
                generation_kwargs={"temperature": 0.1, "max_tokens": 1000},
            )

            # Check response structure - new unified GeneratorResponse
            assert result.is_success() is True
            assert result.content == "Image analysis result"

            # _chat_completionが呼ばれたことを確認
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs

            # メッセージ構造の確認 (system_prompt + user message with image)
            messages = call_kwargs["messages"]
            assert len(messages) == 2

            # System message
            assert messages[0]["role"] == "system"
            assert messages[0]["content"][0]["type"] == "input_text"
            assert messages[0]["content"][0]["text"] == "Analyze this image"

            # User message with image and text
            assert messages[1]["role"] == "user"
            assert len(messages[1]["content"]) == 2

            # 画像部分の確認
            assert messages[1]["content"][0]["type"] == "input_image"
            assert isinstance(messages[1]["content"][0]["image_url"], str)
            # テキスト部分の確認
            assert messages[1]["content"][1]["type"] == "input_text"
            assert (
                "data:image/png;base64,test_base64_data"
                in messages[1]["content"][0]["image_url"]
            )

    def test_component_functionality(self):
        """コンポーネントとしての基本機能テスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        # runメソッドが存在し、呼び出し可能であることを確認
        assert hasattr(generator, "run")
        assert callable(generator.run)

        # _chat_completionメソッドが存在することを確認
        assert hasattr(generator, "_chat_completion")
        assert callable(generator._chat_completion)

    def test_retry_configuration(self):
        """リトライ設定のテスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        retry_config = RetryConfig(max_attempts=5, initial_wait=2.0)

        with patch(
            "llm_applications_library.llm.generators.openai_custom_generator.openai_retry"
        ) as mock_retry:
            # リトライデコレータをモック
            mock_retry.return_value = lambda f: f

            with patch("openai.OpenAI"):
                messages = [{"role": "user", "content": "test"}]
                generator._chat_completion(messages, retry_config=retry_config)

                # リトライデコレータが正しい設定で呼ばれたことを確認
                mock_retry.assert_called_once_with(retry_config)

    def test_chat_completion_passes_generation_params(self):
        """_chat_completionがgeneration_paramsを正しくAPIに渡すテスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_response.usage.model_dump.return_value = {"total_tokens": 100}

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            messages = [{"role": "user", "content": "test"}]
            # text パラメータを含むgeneration_paramsを渡す
            generator._chat_completion(
                messages,
                temperature=0.5,
                max_output_tokens=1000,
                text={"format": {"type": "json_schema", "schema": {"foo": "bar"}}},
            )

            # responses.createが正しいパラメータで呼ばれたことを確認
            mock_client.responses.create.assert_called_once()
            call_kwargs = mock_client.responses.create.call_args.kwargs

            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_output_tokens"] == 1000
            assert call_kwargs["text"] == {
                "format": {"type": "json_schema", "schema": {"foo": "bar"}}
            }

    def test_run_with_text_format_parameter(self):
        """runメソッドでtext (Structured Outputs)パラメータが渡されるテスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        expected_result = {
            "success": True,
            "content": '{"result": "structured"}',
            "usage": {"total_tokens": 150},
            "error": None,
        }

        with patch.object(
            generator, "_chat_completion", return_value=expected_result
        ) as mock_chat:
            result = generator.run(
                base64_images=["test_base64_data"],
                mime_types=["image/png"],
                generation_kwargs={
                    "temperature": 0.1,
                    "text": {"format": {"type": "json_schema", "schema": {"test": 1}}},
                },
            )

            assert result.is_success() is True

            # _chat_completionに正しいパラメータが渡されたことを確認
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["text"] == {
                "format": {"type": "json_schema", "schema": {"test": 1}}
            }

    def test_chat_completion_normalizes_max_tokens(self):
        """_chat_completionがmax_tokensをmax_output_tokensに正規化するテスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_response.usage.model_dump.return_value = {"total_tokens": 100}

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            messages = [{"role": "user", "content": "test"}]
            # legacy max_tokens パラメータを渡す
            generator._chat_completion(
                messages,
                max_tokens=500,
            )

            # responses.createでmax_output_tokensに変換されることを確認
            mock_client.responses.create.assert_called_once()
            call_kwargs = mock_client.responses.create.call_args.kwargs

            assert call_kwargs["max_output_tokens"] == 500
            assert "max_tokens" not in call_kwargs


class TestAzureOpenAISupport:
    """Azure OpenAI対応のテスト"""

    def test_retry_generator_azure_initialization(self):
        """RetryOpenAIGeneratorのAzure初期化テスト"""
        generator = RetryOpenAIGenerator(
            api_key="azure-key",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com/",
            azure_api_version="2024-12-01-preview",
        )
        assert generator._use_azure is True
        assert generator.api_key == "azure-key"
        assert generator.azure_endpoint == "https://test.openai.azure.com/"
        assert generator.azure_api_version == "2024-12-01-preview"

    def test_retry_generator_standard_openai_initialization(self):
        """RetryOpenAIGeneratorの標準OpenAI初期化テスト"""
        generator = RetryOpenAIGenerator(
            api_key="openai-key",
            model="gpt-4o",
        )
        assert generator._use_azure is False
        assert generator.api_key == "openai-key"
        assert generator.azure_endpoint is None

    def test_retry_generator_azure_env_vars(self):
        """RetryOpenAIGeneratorのAzure環境変数テスト"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://env-test.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "env-azure-key",
                "AZURE_OPENAI_API_VERSION": "2024-06-01",
            },
            clear=False,
        ):
            generator = RetryOpenAIGenerator(model="gpt-4o")
            assert generator._use_azure is True
            assert generator.api_key == "env-azure-key"
            assert generator.azure_endpoint == "https://env-test.openai.azure.com/"
            assert generator.azure_api_version == "2024-06-01"

    def test_retry_generator_creates_azure_client(self):
        """RetryOpenAIGeneratorがAzureOpenAIクライアントを作成するテスト"""
        generator = RetryOpenAIGenerator(
            api_key="azure-key",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com/",
        )

        with patch("openai.AzureOpenAI") as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            client = generator._create_client()

            mock_azure.assert_called_once_with(
                api_key="azure-key",
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-12-01-preview",
            )
            assert client == mock_client

    def test_retry_generator_creates_openai_client(self):
        """RetryOpenAIGeneratorが標準OpenAIクライアントを作成するテスト"""
        generator = RetryOpenAIGenerator(
            api_key="openai-key",
            model="gpt-4o",
        )

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            client = generator._create_client()

            mock_openai.assert_called_once_with(api_key="openai-key")
            assert client == mock_client

    def test_vision_generator_azure_initialization(self):
        """OpenAIVisionGeneratorのAzure初期化テスト"""
        generator = OpenAIVisionGenerator(
            model="gpt-4o",
            api_key="azure-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_api_version="2024-12-01-preview",
        )
        assert generator._use_azure is True
        assert generator.api_key == "azure-key"
        assert generator.azure_endpoint == "https://test.openai.azure.com/"

    def test_vision_generator_creates_azure_client(self):
        """OpenAIVisionGeneratorがAzureOpenAIクライアントを作成するテスト"""
        generator = OpenAIVisionGenerator(
            model="gpt-4o",
            api_key="azure-key",
            azure_endpoint="https://test.openai.azure.com/",
        )

        with patch("openai.AzureOpenAI") as mock_azure:
            mock_client = Mock()
            mock_azure.return_value = mock_client

            client = generator._create_client()

            mock_azure.assert_called_once_with(
                api_key="azure-key",
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-12-01-preview",
                max_retries=0,
                timeout=1800,
            )
            assert client == mock_client

    def test_vision_generator_azure_env_vars(self):
        """OpenAIVisionGeneratorのAzure環境変数テスト"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://env-test.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "env-azure-key",
            },
            clear=False,
        ):
            generator = OpenAIVisionGenerator(model="gpt-4o")
            assert generator._use_azure is True
            assert generator.api_key == "env-azure-key"
            assert generator.azure_endpoint == "https://env-test.openai.azure.com/"
