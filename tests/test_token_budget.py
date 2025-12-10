from llm_applications_library.utilities.token_budget import (
    MIN_AVAILABLE_TOKENS,
    calculate_available_tokens,
)


def test_calculate_available_tokens_basic():
    available = calculate_available_tokens(
        prompt_template="Hello {name}",
        variables={"name": "world"},
        model="gpt-4o",
        generation_config={"max_output_tokens": 1000},
        context_windows={"gpt-4o": 10_000},
    )

    assert available > 0
    assert available < 10_000  # should respect context window


def test_calculate_available_tokens_enforces_minimum():
    available = calculate_available_tokens(
        prompt_template="short",
        variables={},
        model="gpt-4o",
        generation_config={"max_output_tokens": 9000},
        context_windows={"gpt-4o": 5000},
    )

    assert available == MIN_AVAILABLE_TOKENS
