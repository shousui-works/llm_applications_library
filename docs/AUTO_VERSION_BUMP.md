# 🤖 自動バージョンアップシステム

このリポジトリでは、PRがマージされた際に自動でバージョンを更新し、リリースを作成する仕組みが導入されています。

## 🏷️ ラベルベースのバージョン管理

### バージョンアップの種類

| ラベル | バージョンアップ | 説明 | 例 |
|--------|-----------------|------|-----|
| `major` | MAJOR | 破壊的変更 | 1.0.0 → 2.0.0 |
| `minor` | MINOR | 新機能追加 | 1.0.0 → 1.1.0 |
| `patch` | PATCH | バグ修正・小改善 | 1.0.0 → 1.0.1 |

### 自動ラベリング

PRを作成すると、タイトルやブランチ名から自動でラベルが付与されます：

#### タイトルベースの判定
- `feat:` または `feature` → `minor` + `feature`
- `fix:` または `bugfix` → `patch` + `bugfix`
- `docs:` または `documentation` → `patch` + `documentation`
- `test:` または `testing` → `patch` + `test`
- `chore:` または `refactor` → `patch` + `maintenance`
- `breaking` または `major` → `major`

#### ブランチ名ベースの判定
- `feature/xxx` → `minor`
- `fix/xxx` または `bugfix/xxx` → `patch`
- `hotfix/xxx` → `patch` + `hotfix`
- `docs/xxx` → `patch` + `documentation`

## 🚀 使用方法

### 1. PRを作成
```bash
git checkout -b feature/new-amazing-feature
# 作業...
git commit -m "feat: 新しい素晴らしい機能を追加"
git push origin feature/new-amazing-feature
gh pr create --base develop --title "feat: 新しい素晴らしい機能を追加"
```

### 2. 自動ラベリングの確認
PRを作成すると自動で適切なラベルが付与されます。必要に応じて手動で調整してください。

### 3. PRをマージ
PRがマージされると：
1. ✅ バージョンが自動更新される
2. 🏷️ 新しいタグが作成される
3. 🚀 GitHubリリースが作成される
4. 📦 パッケージがビルドされる

## 📋 ラベル一覧

### バージョンバンプ用
- 🔴 `major`: メジャーバージョンアップ（破壊的変更）
- 🔵 `minor`: マイナーバージョンアップ（新機能追加）
- 🟣 `patch`: パッチバージョンアップ（バグ修正・小改善）

### 機能分類用
- 🟢 `feature`: 新機能の追加
- 🔴 `bugfix`: バグ修正
- 🔥 `hotfix`: 緊急修正
- 📚 `documentation`: ドキュメントの更新
- 🧪 `test`: テストの追加・修正
- 🔧 `maintenance`: メンテナンス・リファクタリング

### その他
- 🤖 `ci/cd`: CI/CDの改善
- 📦 `dependencies`: 依存関係の更新
- 💥 `breaking-change`: 破壊的変更を含む

## ⚙️ セットアップ手順

### 1. ラベルの作成
リポジトリに必要なラベルを作成：

```bash
# GitHubのActionsタブから「Setup Repository Labels」を手動実行
```

### 2. 現在のPRにラベルを追加
既存のPRまたは新しいPRに適切なラベルを追加してテスト。

## 🔧 カスタマイズ

### デフォルトのバージョンアップタイプを変更
`.github/workflows/auto-version-bump.yml` の以下の部分を修正：

```yaml
else
  # デフォルトはpatch（小さな改善や修正）
  echo "type=patch" >> $GITHUB_OUTPUT
```

### 自動ラベリングルールの調整
`.github/workflows/pr-labeler.yml` のラベリングロジックを修正。

## 📝 例

### 新機能の追加
```bash
# PR作成
gh pr create --title "feat: OpenAI retryユーティリティを追加" --label "minor,feature"

# マージ後 → 1.0.0 → 1.1.0
```

### バグ修正
```bash
# PR作成
gh pr create --title "fix: リトライロジックのバグを修正" --label "patch,bugfix"

# マージ後 → 1.1.0 → 1.1.1
```

### 破壊的変更
```bash
# PR作成
gh pr create --title "feat!: APIインターフェースを大幅変更" --label "major,breaking-change"

# マージ後 → 1.1.1 → 2.0.0
```

## 🚨 注意点

1. **手動でのバージョン更新は避ける**: システムが自動で管理するため、手動でのバージョンファイル変更は推奨されません。

2. **ラベルの正確性**: 適切なバージョンアップのため、PRには正しいラベルを付けてください。

3. **developとmainの運用**:
   - `develop`ブランチ: 開発版のバージョン管理
   - `main`ブランチ: 安定版のリリース

4. **緊急時の手動実行**: 必要に応じて手動でバージョン更新やリリース作成も可能です。

## 🔍 トラブルシューティング

### バージョンが更新されない
- PRに適切なラベル（`major`, `minor`, `patch`）が付いているか確認
- GitHub Actionsの実行ログを確認

### 重複するタグエラー
- 同じバージョンのタグが既に存在する場合は、手動でタグを削除してからリトライ

### 権限エラー
- GitHub Actionsに適切な権限（`contents: write`）があることを確認

---

🤖 このシステムにより、開発チームは効率的で一貫性のあるバージョン管理が可能になります！