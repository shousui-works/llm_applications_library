# テストガイド

このドキュメントでは、llm_applications_libraryのテスト実行方法について説明します。

## テストの種類

### 1. ユニットテスト（デフォルト）
外部依存性をモックした軽量なテスト
```bash
pytest
```

### 2. 統合テスト（Integration Tests）
実際のAPIを呼び出すテスト

## 統合テストの実行

### 前提条件

Claude API統合テストを実行するには、有効なAnthropic API キーが必要です：

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

### 統合テストの実行方法

#### すべての統合テストを実行
```bash
pytest -m integration
```

#### 特定のクラスの統合テストを実行
```bash
pytest tests/test_claude_integration.py::TestRetryClaudeGeneratorIntegration -m integration
```

#### Claude テキスト生成のテストのみ実行
```bash
pytest tests/test_claude_integration.py::TestRetryClaudeGeneratorIntegration::test_simple_text_generation -m integration
```

#### Claude Vision テストのみ実行
```bash
pytest tests/test_claude_integration.py::TestClaudeVisionGeneratorIntegration -m integration
```

#### 時間のかかるテストを含めて実行
```bash
pytest -m "integration and slow"
```

### テストマーカー

- `integration`: 実際のAPIを呼び出す統合テスト（API キーが必要）
- `slow`: 実行に時間がかかるテスト（レート制限テストなど）

## テスト例

### 基本的なテキスト生成テスト
```python
def test_simple_text_generation(api_key):
    generator = RetryClaudeGenerator(api_key=api_key)
    result = generator.run("Hello, please respond with just the word 'success'")

    assert "replies" in result
    assert len(result["replies"]) == 1
    assert isinstance(result["replies"][0], str)
```

### Vision APIテスト
```python
def test_vision_analysis_with_base64(api_key, claude_config, test_image_base64):
    generator = ClaudeVisionGenerator(api_key=api_key)

    result = generator.run(
        base64_image=test_image_base64,
        mime_type="image/png",
        prompt="What do you see in this image?",
        model_config=claude_config
    )

    assert "replies" in result
    assert result["replies"][0]["success"] is True
```

## API制限とベストプラクティス

### レート制限
- Claude APIには使用量制限があります
- 統合テストは実際のAPIクレジットを消費します
- 開発時はユニットテスト（モック）を主に使用することを推奨

### エラーハンドリング
統合テストでは以下の状況を想定しています：
- API キーが無効または期限切れ
- ネットワーク接続エラー
- レート制限エラー
- サービス一時停止

### テスト環境の分離

#### 開発環境
```bash
# ユニットテストのみ実行（推奨）
pytest tests/ -k "not integration"
```

#### CI/CD環境
```bash
# API キーが設定されている場合のみ統合テストを実行
if [ -n "$ANTHROPIC_API_KEY" ]; then
    pytest -m integration
else
    echo "Skipping integration tests - no API key"
    pytest -k "not integration"
fi
```

## トラブルシューティング

### よくあるエラー

#### 1. "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="your_key_here"
pytest -m integration
```

#### 2. "anthropic package not available"
```bash
pip install anthropic
```

#### 3. Rate limit エラー
テスト間隔を空けるか、リトライ設定を調整：
```python
retry_config = RetryConfig(
    max_attempts=3,
    initial_wait=2.0,
    max_wait=10.0,
    multiplier=2.0
)
```

### ログレベルの調整
```bash
pytest -m integration --log-level=DEBUG
```

## 新しい統合テストの追加

新しい統合テストを追加する場合：

1. `@pytest.mark.integration` デコレータを追加
2. API キーの存在確認を含める
3. 適切なエラーハンドリングを実装
4. 実際のAPI使用量を最小限に抑える

例：
```python
@pytest.mark.integration
def test_new_feature(api_key, claude_config):
    \"\"\"Test description.\"\"\"
    try:
        from module import Generator
    except ImportError:
        pytest.skip("Required package not available")

    generator = Generator(api_key=api_key)
    result = generator.some_method()

    assert result["success"] is True
```