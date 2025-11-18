#!/usr/bin/env python3
"""Demo script for testing pipeline factory functionality"""

import os
from llm_applications_library.llm.generators.pipeline_factory import (
    create_pipeline,
    PipelineCreationError,
)
from llm_applications_library.llm.generators.schema import RetryConfig


def test_pipeline_creation():
    """Test pipeline creation with different models"""
    print("🔍 Testing Pipeline Factory")
    print("=" * 50)

    # Test data
    prompt_template = "Answer this question: {question}"
    required_variables = ["question"]
    test_question = "What is the capital of Japan?"

    # Test 1: Auto-detected OpenAI pipeline
    print("\n🤖 Test 1: Auto-detected OpenAI pipeline")
    try:
        pipeline = create_pipeline(
            model="gpt-4o-mini",
            user_prompt_template=prompt_template,
            required_variables=required_variables,
            generation_kwargs={"temperature": 0.7, "max_tokens": 50},
        )
        print("✅ OpenAI pipeline created successfully")

        # Test run
        print("📤 Testing pipeline execution...")
        result = pipeline.run({"PromptBuilder": {"question": test_question}})
        response = result["ProviderSelectableInstructGenerator"]["response"]
        print(f"📝 Response: {response[:100]}...")

    except Exception as e:
        print(f"❌ OpenAI pipeline failed: {e}")

    # Test 2: Auto-detected Claude pipeline
    print("\n🧠 Test 2: Auto-detected Claude pipeline")
    try:
        pipeline = create_pipeline(
            model="claude-sonnet-4-5-20250929",
            user_prompt_template=prompt_template,
            required_variables=required_variables,
            generation_kwargs={"temperature": 0.5, "max_tokens": 50},
        )
        print("✅ Claude pipeline created successfully")

        # Test run
        print("📤 Testing pipeline execution...")
        result = pipeline.run({"PromptBuilder": {"question": test_question}})
        response = result["ProviderSelectableInstructGenerator"]["response"]
        print(f"📝 Response: {response[:100]}...")

    except Exception as e:
        print(f"❌ Claude pipeline failed: {e}")

    # Test 3: Custom retry config
    print("\n🔧 Test 3: Pipeline with custom retry config")
    try:
        retry_config = RetryConfig(max_attempts=2, initial_wait=0.5)
        pipeline = create_pipeline(
            model="gpt-4o-mini",
            user_prompt_template=prompt_template,
            required_variables=required_variables,
            generation_kwargs={"temperature": 0.3, "max_tokens": 30},
            retry_config=retry_config,
        )
        print("✅ Pipeline with custom retry config created successfully")

    except Exception as e:
        print(f"❌ Pipeline with retry config failed: {e}")


def test_error_handling():
    """Test error handling scenarios"""
    print("\n⚠️  Testing Error Handling")
    print("=" * 30)

    # Test 1: Invalid model
    print("\n❌ Test: Invalid model name")
    try:
        pipeline = create_pipeline(
            model="invalid-model-name",
            user_prompt_template="Test: {input}",
            required_variables=["input"],
        )
        print("❌ Should have failed with invalid model")
    except (PipelineCreationError, ValueError) as e:
        print(f"✅ Correctly caught error: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {type(e).__name__}: {e}")

    # Test 2: Missing API keys (if not set)
    print("\n🔑 API Key Status:")
    print(
        f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}"
    )
    print(
        f"   ANTHROPIC_API_KEY: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Not set'}"
    )


def demo_pipeline_features():
    """Demo pipeline advanced features"""
    print("\n🚀 Advanced Pipeline Features Demo")
    print("=" * 40)

    # Complex prompt template
    complex_template = """
    You are an AI assistant. Please answer the following question in the style of {style}.

    Question: {question}

    Additional context: {context}

    Please provide a {response_length} response.
    """

    required_vars = ["style", "question", "context", "response_length"]

    try:
        pipeline = create_pipeline(
            model="claude-3-5-haiku-20241022",
            user_prompt_template=complex_template,
            required_variables=required_vars,
            generation_kwargs={"temperature": 0.8, "max_tokens": 150},
        )

        print("✅ Complex pipeline created successfully")
        print("📋 Template variables:", required_vars)

        # Test with complex input
        test_input = {
            "PromptBuilder": {
                "style": "a friendly teacher",
                "question": "How does photosynthesis work?",
                "context": "This is for a 10-year-old student",
                "response_length": "brief and simple",
            }
        }

        print("📤 Testing complex pipeline...")
        result = pipeline.run(test_input)
        response = result["ProviderSelectableInstructGenerator"]["response"]
        print(f"📝 Complex response: {response[:200]}...")

    except Exception as e:
        print(f"❌ Complex pipeline failed: {e}")


if __name__ == "__main__":
    print("🧪 Pipeline Factory Demo")
    print("=" * 60)

    test_pipeline_creation()
    test_error_handling()
    demo_pipeline_features()

    print("\n🎉 Demo completed!")
