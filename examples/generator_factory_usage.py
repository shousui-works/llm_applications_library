#!/usr/bin/env python3
"""
Example usage of the GeneratorFactory for automatic model selection.

This example demonstrates how to use the new factory system to automatically
select between OpenAI and Claude generators based on model names.
"""

import os
from llm_applications_library.llm.generators import (
    GeneratorFactory,
    create_generator,
    detect_provider_from_model,
    ProviderType,
    RetryConfig,
    Model,
)


def main():
    """Demonstrate factory usage."""
    print("=== Generator Factory Usage Examples ===\n")

    # Example 1: Provider Detection
    print("1. Provider Detection:")
    models_to_test = [
        "gpt-4o",
        "gpt-3.5-turbo",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
    ]

    for model in models_to_test:
        provider = detect_provider_from_model(model)
        print(f"   {model} -> {provider}")

    print()

    # Example 2: Using Model enum
    print("2. Using Model Enum:")
    print(f"   {Model.GPT_4O} -> {detect_provider_from_model(Model.GPT_4O)}")
    print(
        f"   {Model.CLAUDE_3_HAIKU} -> {detect_provider_from_model(Model.CLAUDE_3_HAIKU)}"
    )
    print()

    # Example 3: Creating generators with factory
    print("3. Creating Generators (without API calls):")

    # Create OpenAI text generator
    try:
        openai_gen = GeneratorFactory.create_text_generator(
            model="gpt-4o-mini",
            api_key="demo-key",  # Using demo key for example
        )
        print(f"   ✓ OpenAI Text Generator: {type(openai_gen).__name__}")
    except Exception as e:
        print(f"   ✗ OpenAI Error: {e}")

    # Create Claude text generator
    try:
        claude_gen = GeneratorFactory.create_text_generator(
            model="claude-3-haiku-20240307",
            api_key="demo-key",  # Using demo key for example
        )
        print(f"   ✓ Claude Text Generator: {type(claude_gen).__name__}")
    except Exception as e:
        print(f"   ✗ Claude Error: {e}")

    # Create vision generators
    try:
        vision_gen = GeneratorFactory.create_vision_generator(
            model="gpt-4o", api_key="demo-key"
        )
        print(f"   ✓ OpenAI Vision Generator: {type(vision_gen).__name__}")
    except Exception as e:
        print(f"   ✗ Vision Error: {e}")

    print()

    # Example 4: Convenience function
    print("4. Using Convenience Function:")
    try:
        # Create text generator using convenience function
        gen = create_generator(
            model="gpt-4o", generator_type="text", api_key="demo-key"
        )
        print(f"   ✓ Convenience function: {type(gen).__name__}")
    except Exception as e:
        print(f"   ✗ Convenience Error: {e}")

    print()

    # Example 5: Custom retry configuration
    print("5. Custom Retry Configuration:")
    custom_retry = RetryConfig(
        max_attempts=5, initial_wait=2.0, max_wait=120.0, multiplier=3.0
    )

    try:
        gen = GeneratorFactory.create_text_generator(
            model="gpt-4o-mini", api_key="demo-key", retry_config=custom_retry
        )
        print(f"   ✓ Custom retry config applied: {type(gen).__name__}")
        print(f"   ✓ Retry config: max_attempts={custom_retry.max_attempts}")
    except Exception as e:
        print(f"   ✗ Custom retry error: {e}")

    print()

    # Example 6: Error handling for unknown models
    print("6. Error Handling:")
    try:
        unknown_gen = GeneratorFactory.create_text_generator("unknown-model-123")
    except ValueError as e:
        print(f"   ✓ Properly caught error for unknown model: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    print()

    # Example 7: Actual usage pattern (commented out - requires real API keys)
    print("7. Real Usage Pattern (commented - requires API keys):")
    print("""
    # Real usage with environment variables:
    # os.environ['OPENAI_API_KEY'] = 'your-openai-key'
    # os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key'

    # generator = GeneratorFactory.create_text_generator("gpt-4o")
    # result = generator.run("Hello, world!")
    # print(result["replies"][0])

    # claude_gen = GeneratorFactory.create_text_generator("claude-3-haiku-20240307")
    # result = claude_gen.run("Hello, Claude!")
    # print(result["replies"][0])
    """)

    print("=== Factory Examples Complete ===")


if __name__ == "__main__":
    main()
