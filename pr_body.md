## タイトル
OpenAI/Claudeモデル対応の拡充とトークン計算の共通化

## 概要
- Vision入力の `image_url` を Responses API 仕様に合わせて文字列URLに統一。
- 利用可能トークン計算を共通ユーティリティ `token_budget` として切り出し、`Model` Enum専用のAPIに整理。
- GPT-5/5.1 と Claude 4.5 系モデルを `Model` Enum とコンテキストウィンドウ定義に追加し、デフォルトウィンドウは schema に集約。

## 変更点
- `OpenAIVisionGenerator` の画像URL組み立てを文字列1フィールドに統一。
- `token_budget.py` を追加し、`calculate_available_tokens` が `Model` を受け取り共通のコンテキストウィンドウを参照するように変更。`MODEL_CONTEXT_WINDOWS` は `schema.py` に移動。
- `Model` Enum に GPT-5/5.1, Claude 4.5 系を追加。
- ユーティリティ公開 (`__all__`) とテストを更新 (`test_token_budget.py`, `test_utilities_integration.py`, `test_openai_custom_generator.py`).

## 動作確認
- `uv run tox`（py312, format, lint）
  - 326 passed, 13 skipped、format/lint OK
