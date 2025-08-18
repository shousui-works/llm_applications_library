"""utilities/token_utils.py のテスト"""

from unittest.mock import patch, Mock
import tiktoken

from utilities.token_utils import (
    get_encoding_for_model,
    count_tokens,
    count_tokens_for_messages,
    split_text_by_tokens,
    estimate_prompt_tokens,
    _get_text_tail_by_tokens,
    _split_paragraph_by_sentences,
    _force_split_by_chars,
)


class TestGetEncodingForModel:
    """get_encoding_for_model関数のテスト"""

    @patch("tiktoken.encoding_for_model")
    def test_get_encoding_for_model_known_model(self, mock_encoding_for_model):
        """既知のモデルに対するエンコーディング取得テスト"""
        mock_encoding = Mock()
        mock_encoding_for_model.return_value = mock_encoding

        result = get_encoding_for_model("gpt-4")

        mock_encoding_for_model.assert_called_once_with("gpt-4")
        assert result == mock_encoding

    @patch("tiktoken.get_encoding")
    @patch("tiktoken.encoding_for_model")
    def test_get_encoding_for_model_unknown_model(
        self, mock_encoding_for_model, mock_get_encoding
    ):
        """未知のモデルに対するデフォルトエンコーディング取得テスト"""
        mock_encoding_for_model.side_effect = KeyError("Unknown model")
        mock_default_encoding = Mock()
        mock_get_encoding.return_value = mock_default_encoding

        result = get_encoding_for_model("unknown-model")

        mock_encoding_for_model.assert_called_once_with("unknown-model")
        mock_get_encoding.assert_called_once_with("o200k_base")
        assert result == mock_default_encoding

    def test_get_encoding_for_model_real_models(self):
        """実際のモデルでのエンコーディング取得テスト"""
        # 実際のtiktokenを使用したテスト
        encoding = get_encoding_for_model("gpt-4")
        assert isinstance(encoding, tiktoken.Encoding)

        # 未知のモデルでもエンコーディングが取得できることを確認
        encoding = get_encoding_for_model("nonexistent-model")
        assert isinstance(encoding, tiktoken.Encoding)


class TestCountTokens:
    """count_tokens関数のテスト"""

    def test_count_tokens_empty_string(self):
        """空文字列のトークン数テスト"""
        result = count_tokens("", "gpt-4")
        assert result == 0

    def test_count_tokens_simple_text(self):
        """シンプルなテキストのトークン数テスト"""
        result = count_tokens("Hello, world!", "gpt-4")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_japanese_text(self):
        """日本語テキストのトークン数テスト"""
        result = count_tokens("こんにちは、世界！", "gpt-4")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_different_models(self):
        """異なるモデルでのトークン数テスト"""
        text = "This is a test sentence."

        result_gpt4 = count_tokens(text, "gpt-4")
        result_gpt35 = count_tokens(text, "gpt-3.5-turbo")

        assert isinstance(result_gpt4, int)
        assert isinstance(result_gpt35, int)
        assert result_gpt4 > 0
        assert result_gpt35 > 0

    @patch("utilities.token_utils.get_encoding_for_model")
    def test_count_tokens_with_mock_encoding(self, mock_get_encoding):
        """モックエンコーディングを使用したトークン数テスト"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_get_encoding.return_value = mock_encoding

        result = count_tokens("test text", "gpt-4")

        mock_get_encoding.assert_called_once_with("gpt-4")
        mock_encoding.encode.assert_called_once_with("test text")
        assert result == 5


class TestCountTokensForMessages:
    """count_tokens_for_messages関数のテスト"""

    def test_count_tokens_for_messages_empty_list(self):
        """空のメッセージリストのトークン数テスト"""
        result = count_tokens_for_messages([], "gpt-4")
        assert result == 3  # プライミングトークンのみ

    def test_count_tokens_for_messages_single_message(self):
        """単一メッセージのトークン数テスト"""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        result = count_tokens_for_messages(messages, "gpt-4")
        assert isinstance(result, int)
        assert result > 3  # プライミングトークン + メッセージトークン

    def test_count_tokens_for_messages_multiple_messages(self):
        """複数メッセージのトークン数テスト"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = count_tokens_for_messages(messages, "gpt-4")
        assert isinstance(result, int)
        assert result > 10  # 複数メッセージの合計

    def test_count_tokens_for_messages_with_name(self):
        """名前付きメッセージのトークン数テスト"""
        messages = [{"role": "user", "name": "Alice", "content": "Hello"}]
        result = count_tokens_for_messages(messages, "gpt-4")
        assert isinstance(result, int)
        assert result > 3

    def test_count_tokens_for_messages_gpt35_turbo(self):
        """GPT-3.5-turboモデルでのトークン数テスト"""
        messages = [{"role": "user", "content": "Test message"}]
        result = count_tokens_for_messages(messages, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_for_messages_specific_gpt35_model(self):
        """特定のGPT-3.5モデルでのトークン数テスト"""
        messages = [{"role": "user", "content": "Test message"}]
        result = count_tokens_for_messages(messages, "gpt-3.5-turbo-0613")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_for_messages_gpt4_specific(self):
        """特定のGPT-4モデルでのトークン数テスト"""
        messages = [{"role": "user", "content": "Test message"}]
        result = count_tokens_for_messages(messages, "gpt-4-0613")
        assert isinstance(result, int)
        assert result > 0

    @patch("utilities.token_utils.get_encoding_for_model")
    def test_count_tokens_for_messages_unknown_model(self, mock_get_encoding):
        """未知モデルでのトークン数テスト"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        messages = [{"role": "user", "content": "test"}]
        result = count_tokens_for_messages(messages, "unknown-model")

        assert isinstance(result, int)
        assert result > 0


class TestSplitTextByTokens:
    """split_text_by_tokens関数のテスト"""

    def test_split_text_by_tokens_empty_string(self):
        """空文字列の分割テスト"""
        result = split_text_by_tokens("", 100, "gpt-4")
        assert result == []

    def test_split_text_by_tokens_short_text(self):
        """短いテキストの分割テスト（分割不要）"""
        text = "This is a short text."
        result = split_text_by_tokens(text, 100, "gpt-4")
        assert len(result) == 1
        assert result[0] == text

    def test_split_text_by_tokens_long_text(self):
        """長いテキストの分割テスト"""
        # 十分長いテキストを作成
        text = "This is a test sentence. " * 100
        result = split_text_by_tokens(text, 50, "gpt-4")

        assert len(result) > 1
        assert all(isinstance(chunk, str) for chunk in result)
        assert all(len(chunk) > 0 for chunk in result)

    def test_split_text_by_tokens_paragraph_splitting(self):
        """段落単位での分割テスト"""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = split_text_by_tokens(text, 20, "gpt-4")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_text_by_tokens_with_overlap(self):
        """オーバーラップありの分割テスト"""
        text = "This is a test. " * 20
        result = split_text_by_tokens(text, 30, "gpt-4", overlap_tokens=5)

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_text_by_tokens_japanese_text(self):
        """日本語テキストの分割テスト"""
        text = "これはテストです。" * 20
        result = split_text_by_tokens(text, 30, "gpt-4")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    @patch("utilities.token_utils.count_tokens")
    def test_split_text_by_tokens_mock_count(self, mock_count_tokens):
        """モックを使用したトークン分割テスト"""
        # 最初の呼び出しで長いテキスト、その後は短いテキストとして扱う
        mock_count_tokens.side_effect = [
            100,
            10,
            10,
            10,
        ]  # total, chunk1, chunk2, chunk3

        text = "Test text that should be split"
        result = split_text_by_tokens(text, 50, "gpt-4")

        assert len(result) >= 1


class TestGetTextTailByTokens:
    """_get_text_tail_by_tokens関数のテスト"""

    @patch("utilities.token_utils.get_encoding_for_model")
    def test_get_text_tail_by_tokens_short_text(self, mock_get_encoding):
        """短いテキストの末尾取得テスト"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_encoding.decode.return_value = "short text"
        mock_get_encoding.return_value = mock_encoding

        result = _get_text_tail_by_tokens("short text", 10, "gpt-4")

        assert result == "short text"

    @patch("utilities.token_utils.get_encoding_for_model")
    def test_get_text_tail_by_tokens_long_text(self, mock_get_encoding):
        """長いテキストの末尾取得テスト"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
        mock_encoding.decode.return_value = "tail text"
        mock_get_encoding.return_value = mock_encoding

        result = _get_text_tail_by_tokens("long text", 5, "gpt-4")

        mock_encoding.decode.assert_called_once_with([4, 5, 6, 7, 8])
        assert result == "tail text"


class TestSplitParagraphBySentences:
    """_split_paragraph_by_sentences関数のテスト"""

    def test_split_paragraph_by_sentences_short_paragraph(self):
        """短い段落の文分割テスト"""
        paragraph = "This is a short paragraph."
        result = _split_paragraph_by_sentences(paragraph, 100, "gpt-4")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_paragraph_by_sentences_multiple_sentences(self):
        """複数文を含む段落の分割テスト"""
        paragraph = "First sentence. Second sentence! Third sentence?"
        result = _split_paragraph_by_sentences(paragraph, 20, "gpt-4")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_paragraph_by_sentences_japanese(self):
        """日本語文の分割テスト"""
        paragraph = "最初の文です。二番目の文です！三番目の文です？"
        result = _split_paragraph_by_sentences(paragraph, 20, "gpt-4")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_paragraph_by_sentences_with_overlap(self):
        """オーバーラップありの文分割テスト"""
        paragraph = "Sentence one. Sentence two. Sentence three."
        result = _split_paragraph_by_sentences(paragraph, 15, "gpt-4", overlap_tokens=3)

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)


class TestForceSplitByChars:
    """_force_split_by_chars関数のテスト"""

    @patch("utilities.token_utils.get_encoding_for_model")
    def test_force_split_by_chars(self, mock_get_encoding):
        """文字による強制分割テスト"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_encoding.decode.side_effect = ["chunk1", "chunk2", "chunk3", "chunk4"]
        mock_get_encoding.return_value = mock_encoding

        result = _force_split_by_chars("long text", 3, "gpt-4")

        assert len(result) == 4  # 10トークンを3トークンずつ分割すると4チャンク
        assert result == ["chunk1", "chunk2", "chunk3", "chunk4"]

    @patch("utilities.token_utils.get_encoding_for_model")
    def test_force_split_by_chars_exact_fit(self, mock_get_encoding):
        """ちょうどのサイズでの強制分割テスト"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]
        mock_encoding.decode.side_effect = ["part1", "part2"]
        mock_get_encoding.return_value = mock_encoding

        result = _force_split_by_chars("test", 2, "gpt-4")

        assert len(result) == 2
        assert result == ["part1", "part2"]


class TestEstimatePromptTokens:
    """estimate_prompt_tokens関数のテスト"""

    def test_estimate_prompt_tokens_simple_template(self):
        """シンプルなテンプレートのトークン推定テスト"""
        template = "Hello, {name}! How are you today?"
        variables = {"name": "Alice"}

        result = estimate_prompt_tokens(template, variables, "gpt-4")

        assert isinstance(result, int)
        assert result > 0

    def test_estimate_prompt_tokens_multiple_variables(self):
        """複数変数のテンプレートのトークン推定テスト"""
        template = "User {user} wants to {action} the {object}."
        variables = {"user": "John", "action": "analyze", "object": "document"}

        result = estimate_prompt_tokens(template, variables, "gpt-4")

        assert isinstance(result, int)
        assert result > 0

    def test_estimate_prompt_tokens_missing_variable(self):
        """不足している変数のあるテンプレートのトークン推定テスト"""
        template = "Hello, {name}! Your {item} is ready."
        variables = {"name": "Bob"}  # itemが不足

        # エラーが発生せずに概算が返されることを確認
        result = estimate_prompt_tokens(template, variables, "gpt-4")

        assert isinstance(result, int)
        assert result > 0

    def test_estimate_prompt_tokens_no_variables(self):
        """変数なしテンプレートのトークン推定テスト"""
        template = "This is a simple text with no variables."
        variables = {}

        result = estimate_prompt_tokens(template, variables, "gpt-4")

        assert isinstance(result, int)
        assert result > 0

    def test_estimate_prompt_tokens_japanese_template(self):
        """日本語テンプレートのトークン推定テスト"""
        template = "こんにちは、{name}さん！今日は{weather}ですね。"
        variables = {"name": "田中", "weather": "晴れ"}

        result = estimate_prompt_tokens(template, variables, "gpt-4")

        assert isinstance(result, int)
        assert result > 0


class TestTokenUtilsIntegration:
    """token_utils モジュールの統合テスト"""

    def test_all_functions_importable(self):
        """すべての関数がインポート可能であることを確認"""
        from utilities.token_utils import (
            get_encoding_for_model,
            count_tokens,
            count_tokens_for_messages,
            split_text_by_tokens,
            estimate_prompt_tokens,
        )

        assert callable(get_encoding_for_model)
        assert callable(count_tokens)
        assert callable(count_tokens_for_messages)
        assert callable(split_text_by_tokens)
        assert callable(estimate_prompt_tokens)

    def test_workflow_token_counting_and_splitting(self):
        """トークン計算と分割のワークフローテスト"""
        text = (
            "This is a sample text for testing token counting and splitting functionality. "
            * 10
        )

        # トークン数を計算
        token_count = count_tokens(text, "gpt-4")
        assert token_count > 0

        # テキストを分割
        chunks = split_text_by_tokens(text, 50, "gpt-4")
        assert len(chunks) > 1

        # 各チャンクのトークン数が制限内であることを確認
        for chunk in chunks:
            chunk_tokens = count_tokens(chunk, "gpt-4")
            assert chunk_tokens <= 55  # 多少の余裕を見る

    def test_workflow_message_tokens_and_estimation(self):
        """メッセージトークンとテンプレート推定のワークフローテスト"""
        # メッセージのトークン数を計算
        messages = [{"role": "user", "content": "Count tokens for this message"}]
        message_tokens = count_tokens_for_messages(messages, "gpt-4")
        assert message_tokens > 0

        # テンプレートのトークン数を推定
        template = "Response to: {message}"
        variables = {"message": "Count tokens for this message"}
        estimated_tokens = estimate_prompt_tokens(template, variables, "gpt-4")
        assert estimated_tokens > 0

    def test_different_models_consistency(self):
        """異なるモデル間での一貫性テスト"""
        text = "Test text for model consistency"

        models = ["gpt-4", "gpt-3.5-turbo"]
        results = []

        for model in models:
            tokens = count_tokens(text, model)
            results.append(tokens)
            assert tokens > 0

        # 結果が妥当な範囲内であることを確認（モデル間で多少の差はあり得る）
        assert all(isinstance(r, int) for r in results)
        assert all(r > 0 for r in results)
