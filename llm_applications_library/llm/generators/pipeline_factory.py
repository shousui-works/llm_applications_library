"""Pipeline factory for creating Haystack pipelines with LLM generators."""

import logging
from typing import Any

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from .factory import create_generator
from .schema import RetryConfig
from .converter import ProviderSelectableInstructGenerator


logger = logging.getLogger(__name__)


class PipelineCreationError(Exception):
    """Exception raised when pipeline creation fails."""

    pass


@component
class HaystackGeneratorWrapper:
    """Wrapper component to integrate create_generator output with Haystack pipeline."""

    def __init__(
        self,
        model: str,
        generation_kwargs: dict[str, Any] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.generator = create_generator(model=model, retry_config=retry_config)
        self.generation_kwargs = generation_kwargs or {}

    @component.output_types(replies=list[str])
    def run(self, prompt: str):
        result = self.generator.run(
            prompt=prompt, generation_kwargs=self.generation_kwargs
        )
        replies = result["replies"]
        if not isinstance(replies, list):
            replies = [str(replies)]
        return {"replies": [str(reply) for reply in replies]}


def create_pipeline(
    model: str,
    user_prompt_template: str,
    required_variables: list[str],
    generation_kwargs: dict[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> Pipeline:
    """
    Create a pipeline using model name auto-detection with create_generator.

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

        # Add generator wrapper that uses create_generator
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
