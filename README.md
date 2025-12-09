# LLM Applications Library

OpenAIのビジョン機能とリトライ機能を備えた包括的なLLMアプリケーションライブラリです。Haystackフレームワークとの統合により、堅牢なAIアプリケーションの構築を支援します。

## 特徴

- **OpenAI Vision API統合**: 画像とテキストの統合処理
- **自動リトライ機能**: API呼び出しの堅牢性を向上
- **Haystackコンポーネント**: 標準的なAIパイプライン構築
- **設定可能なスキーマ**: Pydanticベースの型安全な設定管理

## インストール

### 基本インストール

```bash
pip install llm-applications-library
```

### 開発環境用インストール

```bash
pip install llm-applications-library[dev]
```

### Google Cloud Storage機能付き

```bash
pip install llm-applications-library[gcs]
```

### 全機能付きインストール

```bash
pip install llm-applications-library[all]
```

## 使用方法

### 基本的な使用例

```python
from llm.generators.openai_custom_generator import OpenAIVisionGenerator
from llm.generators.schema import RetryConfig

# ジェネレータの初期化
generator = OpenAIVisionGenerator(model="gpt-4o")

# リトライ設定
retry_config = RetryConfig(
    max_attempts=5,
    initial_wait=1.0,
    max_wait=60.0,
    multiplier=2.0
)

# メッセージの作成
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "この画像について説明してください"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}
            }
        ]
    }
]

# API呼び出し
response = generator._chat_completion(messages, retry_config)
print(response)
```

### 設定スキーマの使用

```python
from llm.generators.schema import RetryConfig, GPTConfig

# リトライ設定の作成
retry_config = RetryConfig(
    max_attempts=3,
    initial_wait=2.0,
    max_wait=30.0,
    multiplier=1.5
)

# GPT設定の作成
gpt_config = GPTConfig(
    model="gpt-4o",
    max_output_tokens=1000,
    temperature=0.7
)
```

## 開発

### 環境セットアップ

```bash
git clone https://github.com/shousui-works/llm_applications_library.git
cd llm_applications_library
pip install -e .[dev]
```

### テストの実行

```bash
# 全テストの実行
pytest

# カバレッジ付きテスト
pytest --cov=src/llm

# tox環境でのテスト
uv run tox
```

### コード品質チェック

```bash
# Ruffによるリンティング
ruff check src/ tests/

# Ruffによるフォーマット
ruff format src/ tests/
```

## 更新履歴

### v0.1.21（開発中）
- キャッシュシステムの設計改善
- コードクリーンアップとリファクタリング

### v0.1.20
- VisionGenerator複数画像対応を追加
- パイプライン実行結果でusage情報の取得を可能に
- OpenAIVisionGeneratorでmax_completion_tokensパラメータを正しく処理

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。詳細は[Issues](https://github.com/shousui-works/llm_applications_library/issues)をご確認ください。

## 要件

- Python 3.12以上
- OpenAI API キー（環境変数`OPENAI_API_KEY`として設定）

## 依存関係

- haystack>=0.42
- openai>=1.50.0
- pydantic>=2.11.7
- tenacity>=8.0.0
- pymupdf>=1.26.3
