"""Example usage of vision pipeline with create_vision_pipeline."""

import base64
from llm_applications_library.llm.generators.pipeline_factory import (
    create_vision_pipeline,
)


def example_vision_pipeline():
    """Example of using create_vision_pipeline for image analysis."""

    # Create vision pipeline for GPT-4o
    pipeline = create_vision_pipeline(
        model="gpt-4o",
        generation_kwargs={"temperature": 0.7, "max_output_tokens": 1000},
    )

    # Example base64 image (1x1 red pixel PNG)
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    # Run pipeline with image data
    result = pipeline.run(
        {
            "VisionGenerator": {
                "base64_image": red_pixel_png,
                "mime_type": "image/png",
                "prompt": "この画像に何が写っていますか？色や形について説明してください。",
            }
        }
    )

    print("Vision Pipeline Result:")
    print(f"Keys: {list(result.keys())}")

    # Extract response
    if "ProviderSelectableInstructGenerator" in result:
        response = result["ProviderSelectableInstructGenerator"]["response"]
        print(f"Response: {response}")

    # Also check direct VisionGenerator output
    if "VisionGenerator" in result:
        replies = result["VisionGenerator"]["replies"]
        print(f"Direct Vision Output: {replies}")


def example_claude_vision():
    """Example using Claude for vision analysis."""

    # Create vision pipeline for Claude
    pipeline = create_vision_pipeline(
        model="claude-3-haiku-20240307",
        generation_kwargs={"temperature": 0.1, "max_tokens": 500},
    )

    # Same example image
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    print("\nClaude Vision Pipeline:")
    result = pipeline.run(
        {
            "VisionGenerator": {
                "base64_image": red_pixel_png,
                "mime_type": "image/png",
                "prompt": "この画像を分析して、色、形、サイズについて教えてください。",
            }
        }
    )

    print(f"Keys: {list(result.keys())}")
    if "VisionGenerator" in result:
        replies = result["VisionGenerator"]["replies"]
        print(f"Claude Vision Output: {replies}")


if __name__ == "__main__":
    print("Vision Pipeline Examples")
    print("=" * 40)

    try:
        example_vision_pipeline()
        example_claude_vision()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: This requires valid API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)")
