#!/usr/bin/env python3
"""Test script for long prompt handling in Claude Vision API"""

import os
import base64
import logging
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from llm_applications_library.llm.generators.claude_custom_generator import (
    ClaudeVisionGenerator,
)

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)


def test_long_prompt():
    """Test Claude Vision API with very long prompt (similar to the error case)"""
    print("🧪 Testing Claude Vision API with Long Prompt")
    print("=" * 50)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return

    print(f"✅ API Key found: {api_key[:10]}...")

    try:
        generator = ClaudeVisionGenerator(
            model="claude-sonnet-4-5-20250929",  # Use the same model as error case
            api_key=api_key,
        )

        # Create test image
        test_image = base64.b64encode(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )
        ).decode("utf-8")

        # Create very long prompt similar to the error case
        long_prompt = (
            """---

# 法務デューデリジェンスレポート更新プロンプト

## 重要な注意事項
**レポートフォーマット内のAIへの指示（【プロンプト】タグ、命令形の文章、処理指示など）は、すべて実行対象であり、レポート本文には一切含めません。最終出力は弁護士が作成する法務デューデリジェンスレポートとして、専門性と正確性を備えた文章のみとします。**

## 必ず守るルール

- 本レポートは、対象会社に関する法務デューデリジェンスの結果を整理したものです。
"""
            * 100
        )  # Repeat to make it very long

        print(f"📋 Test details:")
        print(f"  - Model: claude-sonnet-4-5-20250929")
        print(f"  - Prompt length: {len(long_prompt):,} characters")
        print(f"  - Image data length: {len(test_image)}")

        print("\n📤 Testing API call with long prompt...")
        result = generator.run(
            base64_image=test_image,
            mime_type="image/png",
            prompt=long_prompt,
        )

        print("📝 Result:")
        if isinstance(result, dict) and result.get("replies"):
            reply = result["replies"][0]
            if isinstance(reply, dict):
                if reply.get("success", True):
                    print(f"✅ Success: {reply.get('content', '')[:200]}...")
                    print(f"📊 Usage: {reply.get('usage', {})}")
                else:
                    print(f"❌ Failed: {reply.get('error', 'Unknown error')}")
            else:
                print(f"✅ Success: {str(reply)[:200]}...")
        else:
            print(f"❓ Unexpected result: {str(result)[:200]}...")

    except Exception as e:
        print(f"❌ Exception: {e}")
        print(f"   Error type: {e.__class__.__name__}")

        # Additional error details
        if hasattr(e, "__dict__"):
            print(f"   Error attributes: {e.__dict__}")


if __name__ == "__main__":
    test_long_prompt()
