"""Tests for pipeline factory functionality."""

import pytest
from unittest.mock import patch, MagicMock

from llm_applications_library.llm.generators.pipeline_factory import (
    create_pipeline,
    create_vision_pipeline,
    PipelineCreationError,
    HaystackGeneratorWrapper,
    HaystackVisionGeneratorWrapper,
)
from llm_applications_library.llm.generators.schema import RetryConfig


class TestHaystackGeneratorWrapper:
    """Test the text generator wrapper component."""

    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.GeneratorFactory.create_text_generator"
    )
    def test_text_wrapper_initialization(self, mock_create_text):
        """Test text wrapper initialization."""
        mock_generator = MagicMock()
        mock_create_text.return_value = mock_generator

        retry_config = RetryConfig()
        wrapper = HaystackGeneratorWrapper(
            model="gpt-4o",
            generation_kwargs={"temperature": 0.7},
            retry_config=retry_config,
        )

        mock_create_text.assert_called_once_with(
            model="gpt-4o", retry_config=retry_config
        )
        assert wrapper.generator == mock_generator
        assert wrapper.generation_kwargs == {"temperature": 0.7}

    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.GeneratorFactory.create_text_generator"
    )
    def test_text_wrapper_run(self, mock_create_text):
        """Test text wrapper run method."""
        mock_generator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated response"
        mock_generator.run.return_value = mock_response
        mock_create_text.return_value = mock_generator

        wrapper = HaystackGeneratorWrapper(model="gpt-4o")
        result = wrapper.run("Test prompt")

        mock_generator.run.assert_called_once_with(
            prompt="Test prompt", generation_kwargs={}
        )
        assert result == {"replies": ["Generated response"]}

    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.GeneratorFactory.create_text_generator"
    )
    def test_text_wrapper_run_empty_content(self, mock_create_text):
        """Test text wrapper run with empty content."""
        mock_generator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = None
        mock_generator.run.return_value = mock_response
        mock_create_text.return_value = mock_generator

        wrapper = HaystackGeneratorWrapper(model="gpt-4o")
        result = wrapper.run("Test prompt")

        assert result == {"replies": []}


class TestHaystackVisionGeneratorWrapper:
    """Test the vision generator wrapper component."""

    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.GeneratorFactory.create_vision_generator"
    )
    def test_vision_wrapper_initialization(self, mock_create_vision):
        """Test vision wrapper initialization."""
        mock_generator = MagicMock()
        mock_create_vision.return_value = mock_generator

        retry_config = RetryConfig()
        wrapper = HaystackVisionGeneratorWrapper(
            model="gpt-4o",
            generation_kwargs={"temperature": 0.7},
            retry_config=retry_config,
        )

        mock_create_vision.assert_called_once_with(
            model="gpt-4o", retry_config=retry_config
        )
        assert wrapper.generator == mock_generator
        assert wrapper.generation_kwargs == {"temperature": 0.7}

    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.GeneratorFactory.create_vision_generator"
    )
    def test_vision_wrapper_run(self, mock_create_vision):
        """Test vision wrapper run method."""
        mock_generator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Image analysis result"
        mock_generator.run.return_value = mock_response
        mock_create_vision.return_value = mock_generator

        wrapper = HaystackVisionGeneratorWrapper(model="gpt-4o")
        result = wrapper.run(
            base64_image="fake_base64",
            mime_type="image/jpeg",
            prompt="Analyze this image",
        )

        mock_generator.run.assert_called_once_with(
            base64_image="fake_base64",
            mime_type="image/jpeg",
            prompt="Analyze this image",
            generation_kwargs={},
        )
        assert result == {"replies": ["Image analysis result"]}

    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.GeneratorFactory.create_vision_generator"
    )
    def test_vision_wrapper_run_default_prompt(self, mock_create_vision):
        """Test vision wrapper run with default prompt."""
        mock_generator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Analysis result"
        mock_generator.run.return_value = mock_response
        mock_create_vision.return_value = mock_generator

        wrapper = HaystackVisionGeneratorWrapper(model="gpt-4o")
        result = wrapper.run(base64_image="fake_base64", mime_type="image/jpeg")

        mock_generator.run.assert_called_once_with(
            base64_image="fake_base64",
            mime_type="image/jpeg",
            prompt="この画像を詳細に分析してください。",
            generation_kwargs={},
        )
        assert result == {"replies": ["Analysis result"]}


class TestCreatePipeline:
    """Test create_pipeline function."""

    @patch("llm_applications_library.llm.generators.pipeline_factory.Pipeline")
    @patch("llm_applications_library.llm.generators.pipeline_factory.PromptBuilder")
    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.HaystackGeneratorWrapper"
    )
    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.ProviderSelectableInstructGenerator"
    )
    def test_create_pipeline_success(
        self, mock_provider_gen, mock_wrapper, mock_prompt_builder, mock_pipeline
    ):
        """Test successful pipeline creation."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        result = create_pipeline(
            model="gpt-4o",
            user_prompt_template="Answer: {question}",
            required_variables=["question"],
            generation_kwargs={"temperature": 0.7},
        )

        # Verify pipeline components were added
        assert mock_pipeline_instance.add_component.call_count == 3
        assert mock_pipeline_instance.connect.call_count == 2
        assert result == mock_pipeline_instance

    def test_create_pipeline_error_handling(self):
        """Test pipeline creation error handling."""
        with patch(
            "llm_applications_library.llm.generators.pipeline_factory.Pipeline"
        ) as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline error")

            with pytest.raises(PipelineCreationError, match="Pipeline creation failed"):
                create_pipeline(
                    model="gpt-4o",
                    user_prompt_template="Answer: {question}",
                    required_variables=["question"],
                )


class TestCreateVisionPipeline:
    """Test create_vision_pipeline function."""

    @patch("llm_applications_library.llm.generators.pipeline_factory.Pipeline")
    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.HaystackVisionGeneratorWrapper"
    )
    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.ProviderSelectableInstructGenerator"
    )
    def test_create_vision_pipeline_success(
        self, mock_provider_gen, mock_wrapper, mock_pipeline
    ):
        """Test successful vision pipeline creation."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        result = create_vision_pipeline(
            model="gpt-4o", generation_kwargs={"temperature": 0.7}
        )

        # Verify pipeline components were added
        assert mock_pipeline_instance.add_component.call_count == 2
        assert mock_pipeline_instance.connect.call_count == 1
        assert result == mock_pipeline_instance

        # Verify wrapper was called with correct parameters
        mock_wrapper.assert_called_once()
        args, kwargs = mock_wrapper.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["generation_kwargs"] == {"temperature": 0.7}

    @patch("llm_applications_library.llm.generators.pipeline_factory.Pipeline")
    @patch(
        "llm_applications_library.llm.generators.pipeline_factory.HaystackVisionGeneratorWrapper"
    )
    def test_create_vision_pipeline_with_retry_config(
        self, mock_wrapper, mock_pipeline
    ):
        """Test vision pipeline creation with custom retry config."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        retry_config = RetryConfig(max_attempts=5)
        result = create_vision_pipeline(
            model="claude-3-haiku", retry_config=retry_config
        )

        assert result == mock_pipeline_instance

    def test_create_vision_pipeline_error_handling(self):
        """Test vision pipeline creation error handling."""
        with patch(
            "llm_applications_library.llm.generators.pipeline_factory.Pipeline"
        ) as mock_pipeline:
            mock_pipeline.side_effect = Exception("Vision pipeline error")

            with pytest.raises(
                PipelineCreationError, match="Vision pipeline creation failed"
            ):
                create_vision_pipeline(model="gpt-4o")


class TestPipelineIntegration:
    """Test pipeline integration scenarios."""

    def test_text_pipeline_components_structure(self):
        """Test that text pipeline has correct component structure."""
        pipeline = create_pipeline(
            model="gpt-4o",
            user_prompt_template="Answer: {question}",
            required_variables=["question"],
        )

        # Check components exist
        expected_components = [
            "PromptBuilder",
            "Generator",
            "ProviderSelectableInstructGenerator",
        ]
        actual_components = list(pipeline.graph.nodes())

        assert set(expected_components) == set(actual_components)

        # Check connections
        expected_connections = [
            ("PromptBuilder", "Generator"),
            ("Generator", "ProviderSelectableInstructGenerator"),
        ]
        actual_connections = list(pipeline.graph.edges())

        assert set(expected_connections) == set(actual_connections)

    def test_vision_pipeline_components_structure(self):
        """Test that vision pipeline has correct component structure."""
        pipeline = create_vision_pipeline(model="gpt-4o")

        # Check components exist
        expected_components = ["VisionGenerator", "ProviderSelectableInstructGenerator"]
        actual_components = list(pipeline.graph.nodes())

        assert set(expected_components) == set(actual_components)

        # Check connections
        expected_connections = [
            ("VisionGenerator", "ProviderSelectableInstructGenerator")
        ]
        actual_connections = list(pipeline.graph.edges())

        assert set(expected_connections) == set(actual_connections)
