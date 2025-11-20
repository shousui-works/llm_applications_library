"""Pipeline factory for creating Haystack pipelines with LLM generators."""

import logging
from typing import Any

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from .factory import GeneratorFactory
from .schema import RetryConfig
from .converter import ProviderSelectableInstructGenerator


logger = logging.getLogger(__name__)


class PipelineCreationError(Exception):
    """Exception raised when pipeline creation fails."""

    pass


@component
class HaystackGeneratorWrapper:
    """Wrapper component to integrate GeneratorFactory output with Haystack pipeline."""

    def __init__(
        self,
        model: str,
        generation_kwargs: dict[str, Any] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.generator = GeneratorFactory.create_text_generator(
            model=model, retry_config=retry_config
        )
        self.generation_kwargs = generation_kwargs or {}

    @component.output_types(replies=list[str])
    def run(self, prompt: str):
        result = self.generator.run(
            prompt=prompt, generation_kwargs=self.generation_kwargs
        )
        # Handle new GeneratorResponse format
        if hasattr(result, "content"):
            content = result.content
            replies = [content] if content else []
        else:
            # Fallback for old format (backward compatibility) - should not happen with new generators
            replies = []
        return {"replies": [str(reply) for reply in replies]}


@component
class HaystackVisionGeneratorWrapper:
    """Wrapper component to integrate Vision GeneratorFactory output with Haystack pipeline."""

    def __init__(
        self,
        model: str,
        generation_kwargs: dict[str, Any] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.generator = GeneratorFactory.create_vision_generator(
            model=model, retry_config=retry_config
        )
        self.generation_kwargs = generation_kwargs or {}

    @component.output_types(replies=list[str])
    def run(
        self,
        base64_image: str,
        mime_type: str,
        prompt: str = "この画像を詳細に分析してください。",
    ):
        result = self.generator.run(
            base64_image=base64_image,
            mime_type=mime_type,
            prompt=prompt,
            generation_kwargs=self.generation_kwargs,
        )
        # Handle new GeneratorResponse format
        if hasattr(result, "content"):
            content = result.content
            replies = [content] if content else []
        else:
            # Fallback for old format (backward compatibility) - should not happen with new generators
            replies = []
        return {"replies": [str(reply) for reply in replies]}


def create_pipeline(
    model: str,
    user_prompt_template: str,
    required_variables: list[str],
    generation_kwargs: dict[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> Pipeline:
    """
    Create a pipeline using model name auto-detection with GeneratorFactory.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
        user_prompt_template: Template string for the prompt
        required_variables: List of required template variables
        generation_kwargs: Optional generation parameters
        retry_config: Optional retry configuration

    Returns:
        Configured Haystack pipeline

    Raises:
        PipelineCreationError: If pipeline creation fails

    Example:
        ```python
        pipeline = create_pipeline(
            model="gpt-4o",
            user_prompt_template="Answer this question: {question}",
            required_variables=["question"],
            generation_kwargs={"temperature": 0.7, "max_tokens": 100}
        )

        result = pipeline.run({"PromptBuilder": {"question": "What is AI?"}})
        response = result["ProviderSelectableInstructGenerator"]["response"]
        ```
    """
    try:
        logger.debug(f"Creating pipeline for model {model}")

        # Create default retry config if not provided
        if retry_config is None:
            retry_config = RetryConfig()

        pipeline = Pipeline()

        # Add PromptBuilder component
        pipeline.add_component(
            name="PromptBuilder",
            instance=PromptBuilder(
                template=user_prompt_template, required_variables=required_variables
            ),
        )

        # Add generator wrapper that uses GeneratorFactory
        generator_wrapper = HaystackGeneratorWrapper(
            model=model, generation_kwargs=generation_kwargs, retry_config=retry_config
        )
        pipeline.add_component(name="Generator", instance=generator_wrapper)

        # Add ProviderSelectableInstructGenerator
        pipeline.add_component(
            name="ProviderSelectableInstructGenerator",
            instance=ProviderSelectableInstructGenerator(),
        )

        # Connect pipeline components
        pipeline.connect(sender="PromptBuilder", receiver="Generator")
        pipeline.connect(
            sender="Generator", receiver="ProviderSelectableInstructGenerator"
        )

        logger.debug("Pipeline created successfully")
        return pipeline

    except Exception as e:
        logger.exception(f"Failed to create pipeline for model {model}")
        raise PipelineCreationError(f"Pipeline creation failed: {e}")


def create_vision_pipeline(
    model: str,
    user_prompt_template: str,
    required_variables: list[str],
    generation_kwargs: dict[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> Pipeline:
    """
    Create a vision pipeline using model name auto-detection with GeneratorFactory.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
        user_prompt_template: Template string for the vision prompt
        required_variables: List of required template variables
        generation_kwargs: Optional generation parameters
        retry_config: Optional retry configuration

    Returns:
        Configured Haystack vision pipeline

    Raises:
        PipelineCreationError: If pipeline creation fails

    Example:
        ```python
        pipeline = create_vision_pipeline(
            model="gpt-4o",
            user_prompt_template="この画像について{question}を答えてください",
            required_variables=["question"],
            generation_kwargs={"temperature": 0.7, "max_tokens": 100}
        )

        result = pipeline.run({
            "VisionPromptBuilder": {"question": "何が写っていますか？"},
            "VisionGenerator": {
                "base64_image": "base64_encoded_image_data",
                "mime_type": "image/jpeg"
            }
        })
        response = result["ProviderSelectableInstructGenerator"]["response"]
        ```
    """
    try:
        logger.debug(f"Creating vision pipeline for model {model}")

        # Create default retry config if not provided
        if retry_config is None:
            retry_config = RetryConfig()

        pipeline = Pipeline()

        # Add VisionPromptBuilder component
        pipeline.add_component(
            name="VisionPromptBuilder",
            instance=PromptBuilder(
                template=user_prompt_template, required_variables=required_variables
            ),
        )

        # Add vision generator wrapper that uses GeneratorFactory
        vision_generator_wrapper = HaystackVisionGeneratorWrapper(
            model=model, generation_kwargs=generation_kwargs, retry_config=retry_config
        )
        pipeline.add_component(
            name="VisionGenerator", instance=vision_generator_wrapper
        )

        # Add ProviderSelectableInstructGenerator
        pipeline.add_component(
            name="ProviderSelectableInstructGenerator",
            instance=ProviderSelectableInstructGenerator(),
        )

        # Connect pipeline components
        pipeline.connect(
            sender="VisionPromptBuilder", receiver="VisionGenerator.prompt"
        )
        pipeline.connect(
            sender="VisionGenerator", receiver="ProviderSelectableInstructGenerator"
        )

        logger.debug("Vision pipeline created successfully")
        return pipeline

    except Exception as e:
        logger.exception(f"Failed to create vision pipeline for model {model}")
        raise PipelineCreationError(f"Vision pipeline creation failed: {e}")
