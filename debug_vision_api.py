#!/usr/bin/env python3
"""Debug script for Claude Vision API issues"""

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


def create_test_image():
    """Create a minimal valid PNG image in base64"""
    # 1x1 transparent PNG
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png_data).decode("utf-8")


def test_vision_api():
    """Test Claude Vision API with debugging"""
    print("🔍 Debugging Claude Vision API")
    print("=" * 50)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return

    print(f"✅ API Key found: {api_key[:10]}...")

    try:
        # Create generator
        generator = ClaudeVisionGenerator(
            model="claude-3-haiku-20240307", api_key=api_key
        )
        print("✅ Generator created successfully")

        # Test with minimal image
        test_image = create_test_image()
        print(f"✅ Test image created (length: {len(test_image)})")

        print("\n📤 Testing API call...")
        print(f"📋 Request details:")
        print(f"  - Image data length: {len(test_image)}")
        print(f"  - MIME type: image/png")
        print(f"  - Prompt: この画像を分析してください。")

        result = generator.run(
            base64_image=test_image,
            mime_type="image/png",
            prompt="この画像を分析してください。",
        )

        print("📝 Result:", result)

        # Check if error occurred
        if isinstance(result, dict) and result.get("replies"):
            reply = result["replies"][0]
            if isinstance(reply, dict) and not reply.get("success", True):
                print(f"❌ API call failed: {reply.get('error', 'Unknown error')}")
            else:
                print("✅ API call succeeded!")

    except Exception as e:
        print(f"❌ Error: {e}")

        # Additional error details
        if hasattr(e, "__dict__"):
            print(f"Error attributes: {e.__dict__}")


if __name__ == "__main__":
    test_vision_api()
