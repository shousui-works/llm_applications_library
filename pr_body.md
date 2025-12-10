## タイトル
Vision入力画像の `image_url` をResponses API仕様に合わせて文字列化

## 概要
- Vision用メッセージで画像URLを文字列1フィールドに統一し、誤ってオブジェクトを送らないよう修正しました。
- 仕様に沿わない重複キーを排除し、OpenAI Responses APIに素直に通る形に揃えています。

## 変更点
- `OpenAIVisionGenerator.run` の画像コンテンツ組み立てで `image_url` を `"data:{mime};base64,..."` の文字列だけにし、重複キーを削除。
- コメントで仕様意図を明示。

## 動作確認
- ローカルテストは未実行（環境未セットアップ）。
- 実行推奨: `pytest tests/test_openai_custom_generator.py::TestOpenAIVisionGenerator::test_chat_completion_success`
