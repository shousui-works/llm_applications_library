"""OpenAI Custom Generator の実用的なテスト"""

from unittest.mock import Mock, patch
from llm.generators.openai_custom_generator import OpenAIVisionGenerator
from llm.generators.schema import GPTConfig, OpenAIGenerationConfig, RetryConfig


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
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.model_dump.return_value = {"total_tokens": 100}

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
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
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            messages = [{"role": "user", "content": "test"}]
            result = generator._chat_completion(messages)

            assert result["success"] is False
            assert result["content"] is None
            assert "API Error" in result["error"]

    def test_run_method(self):
        """runメソッドのテスト"""
        generator = OpenAIVisionGenerator(model="gpt-4o", api_key="test-key")

        model_config = GPTConfig(
            model="gpt-4o",
            generation_config=OpenAIGenerationConfig(temperature=0.5),
            retry_config=RetryConfig(max_attempts=3),
        )

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
                base64_image="test_base64_data",
                mime_type="image/png",
                prompt="Analyze this image",
                model_config=model_config,
            )

            assert "replies" in result
            assert result["replies"] == expected_result

            # _chat_completionが呼ばれたことを確認
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs

            # メッセージ構造の確認
            messages = call_kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert len(messages[0]["content"]) == 2

            # テキスト部分の確認
            assert messages[0]["content"][0]["type"] == "text"
            assert messages[0]["content"][0]["text"] == "Analyze this image"

            # 画像部分の確認
            assert messages[0]["content"][1]["type"] == "image_url"
            assert (
                "data:image/png;base64,test_base64_data"
                in (messages[0]["content"][1]["image_url"]["url"])
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

        with patch("llm.generators.openai_custom_generator.openai_retry") as mock_retry:
            # リトライデコレータをモック
            mock_retry.return_value = lambda f: f

            with patch("openai.OpenAI"):
                messages = [{"role": "user", "content": "test"}]
                generator._chat_completion(messages, retry_config=retry_config)

                # リトライデコレータが正しい設定で呼ばれたことを確認
                mock_retry.assert_called_once_with(retry_config)
