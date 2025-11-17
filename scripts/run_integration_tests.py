#!/usr/bin/env python3
"""
統合テスト実行スクリプト

このスクリプトは Claude API の統合テストを実行します。
ANTHROPIC_API_KEY 環境変数が設定されている場合のみテストを実行します。
"""

import os
import subprocess
import sys
from pathlib import Path


def check_api_key():
    """API キーの存在を確認"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        print("\n統合テストを実行するには、有効な Anthropic API キーが必要です：")
        print("export ANTHROPIC_API_KEY='your_api_key_here'")
        return False
    print(f"✅ ANTHROPIC_API_KEY が設定されています (***{api_key[-4:]})")
    return True


def check_dependencies():
    """必要な依存関係を確認"""
    try:
        import pytest

        print("✅ pytest が利用可能です")
    except ImportError:
        print("❌ pytest がインストールされていません")
        print("pip install pytest を実行してください")
        return False

    try:
        import anthropic

        print("✅ anthropic パッケージが利用可能です")
    except ImportError:
        print("❌ anthropic パッケージがインストールされていません")
        print("pip install anthropic を実行してください")
        return False

    return True


def run_tests(test_type="all"):
    """テストを実行"""
    project_root = Path(__file__).parent.parent

    if test_type == "all":
        cmd = ["python", "-m", "pytest", "-m", "integration", "-v"]
    elif test_type == "text":
        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/test_claude_integration.py::TestRetryClaudeGeneratorIntegration",
            "-v",
        ]
    elif test_type == "vision":
        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/test_claude_integration.py::TestClaudeVisionGeneratorIntegration",
            "-v",
        ]
    elif test_type == "fast":
        cmd = ["python", "-m", "pytest", "-m", "integration and not slow", "-v"]
    else:
        print(f"❌ 不明なテストタイプ: {test_type}")
        return False

    print(f"\n🚀 統合テスト実行中: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n✅ すべてのテストが成功しました！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ テストが失敗しました（終了コード: {e.returncode}）")
        return False


def main():
    """メイン実行関数"""
    print("🔧 Claude API 統合テスト実行ツール")
    print("=" * 40)

    # コマンドライン引数の処理
    test_type = "all"
    if len(sys.argv) > 1:
        test_type = sys.argv[1]

    print(f"テストタイプ: {test_type}")

    # 前提条件の確認
    if not check_api_key():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    # テスト実行
    if run_tests(test_type):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
