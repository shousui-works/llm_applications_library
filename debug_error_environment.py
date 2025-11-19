#!/usr/bin/env python3
"""Debug script to reproduce the specific error environment"""

import os
import base64
import json
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


def test_different_configurations():
    """Test different configurations to reproduce the error"""
    print("🔍 Testing Different Claude Vision Configurations")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return

    print(f"✅ API Key found: {api_key[:10]}...")

    # Test configurations that might cause 400 error
    test_cases = [
        {
            "name": "Empty prompt",
            "model": "claude-3-haiku-20240307",
            "prompt": "",
            "mime_type": "image/png",
        },
        {
            "name": "Different model",
            "model": "claude-3-5-sonnet-20241022",
            "prompt": "この画像を分析してください。",
            "mime_type": "image/png",
        },
        {
            "name": "Wrong MIME type",
            "model": "claude-3-haiku-20240307",
            "prompt": "この画像を分析してください。",
            "mime_type": "image/jpeg",  # Wrong MIME type for PNG data
        },
        {
            "name": "Very long prompt",
            "model": "claude-3-haiku-20240307",
            "prompt": "この画像を詳細に分析してください。" * 100,  # Very long prompt
            "mime_type": "image/png",
        },
        {
            "name": "Invalid base64 padding",
            "model": "claude-3-haiku-20240307",
            "prompt": "この画像を分析してください。",
            "mime_type": "image/png",
            "use_invalid_base64": True,
        },
        {
            "name": "Large image simulation",
            "model": "claude-3-haiku-20240307",
            "prompt": "この画像を分析してください。",
            "mime_type": "image/png",
            "use_large_image": True,
        },
    ]

    # Create test image
    test_image = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
    ).decode("utf-8")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print("=" * 30)

        try:
            generator = ClaudeVisionGenerator(model=test_case["model"], api_key=api_key)

            # Prepare image data
            image_data = test_image
            if test_case.get("use_invalid_base64"):
                image_data = test_image[:-2]  # Remove padding to make invalid
            elif test_case.get("use_large_image"):
                # Create artificially large base64 string
                image_data = test_image * 1000

            print(f"📋 Parameters:")
            print(f"  - Model: {test_case['model']}")
            print(
                f"  - Prompt: {test_case['prompt'][:50]}{'...' if len(test_case['prompt']) > 50 else ''}"
            )
            print(f"  - MIME: {test_case['mime_type']}")
            print(f"  - Image length: {len(image_data)}")

            result = generator.run(
                base64_image=image_data,
                mime_type=test_case["mime_type"],
                prompt=test_case["prompt"],
            )

            # Check result
            if isinstance(result, dict) and result.get("replies"):
                reply = result["replies"][0]
                if isinstance(reply, dict):
                    if reply.get("success", True):
                        print(f"✅ Success: {reply.get('content', '')[:100]}...")
                    else:
                        print(f"❌ Failed: {reply.get('error', 'Unknown error')}")
                else:
                    print(f"✅ Success: {str(reply)[:100]}...")
            else:
                print(f"❓ Unexpected result: {str(result)[:100]}...")

        except Exception as e:
            print(f"❌ Exception: {e}")

            # Try to get detailed error information
            if hasattr(e, "__class__"):
                print(f"   Error type: {e.__class__.__name__}")
            if hasattr(e, "response"):
                print(f"   Has response attribute: True")
                try:
                    if hasattr(e.response, "text"):
                        print(f"   Response text: {e.response.text}")
                    if hasattr(e.response, "status_code"):
                        print(f"   Status code: {e.response.status_code}")
                except:
                    print("   Could not extract response details")


if __name__ == "__main__":
    test_different_configurations()
