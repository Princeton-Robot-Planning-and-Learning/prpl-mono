"""Tests for code.py."""

import json
import tempfile
from pathlib import Path

import pytest

from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SemanticsPythonRepromptCheck,
    SyntaxRepromptCheck,
    SynthesizedPythonFunction,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Query, Response

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_synthesized_python_function():
    """Tests for SynthesizedPythonFunction()."""

    code_str = """
from dataclasses import dataclass

@dataclass
class Dog:

    name: str
    is_cute: bool = True


def count_cute_dogs(dog_names: list[str]) -> int:
    dogs = [Dog(d) for d in dog_names]
    return sum(d.is_cute for d in dogs)
"""

    synthesized_python_fn = SynthesizedPythonFunction("count_cute_dogs", code_str)
    assert synthesized_python_fn.run(["nomsy"]) == 1
    assert synthesized_python_fn.run(["nomsy", "puddles"]) == 2


def test_synthesize_python_function_with_llm():
    """Tests for synthesize_python_function_with_llm()."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
        cache = FilePretrainedLargeModelCache(cache_path)

        function_name = "count_good_dogs"
        input_output_examples = [([["nomsy", "rover"]], 2), ([["nomsy"]], 1)]
        inputs = [i for i, _ in input_output_examples]
        output_check_fns = [lambda x, o=o: x == o for _, o in input_output_examples]
        reprompt_checks = [
            SyntaxRepromptCheck(),
            FunctionOutputRepromptCheck(
                function_name, inputs, output_check_fns, function_timeout=1.0
            ),
        ]

        query = Query(
            """Generate a Python function of the form

def count_good_dogs(dog_names: list[str]) -> int:
    # your code here
"""
        )

        ordered_responses = []

        # Syntax Error
        ordered_responses.append(
            Response(
                """```python
def count_good_dogs(dog_names: list[str) -> int:
    return len(dog_names)
```
""",
                {},
            )
        )

        # Exception Failure
        ordered_responses.append(
            Response(
                """```python
def count_good_dogs(dog_names: list[str]) -> int:
    raise ValueError("Oops!")
```
""",
                {},
            )
        )

        # Semantic Failure
        ordered_responses.append(
            Response(
                """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return 2
```
""",
                {},
            )
        )

        # Infinite Loop
        ordered_responses.append(
            Response(
                """```python
def count_good_dogs(dog_names: list[str]) -> int:
    num_good_dogs = 0
    while True:
        num_good_dogs += 1
    return num_good_dogs
```
""",
                {},
            )
        )

        # Correct Answer
        ordered_responses.append(
            Response(
                """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return len(dog_names)
```
""",
                {},
            )
        )

        llm = OrderedResponseModel(ordered_responses, cache)

        synthesized_python_fn = synthesize_python_function_with_llm(
            function_name,
            llm,
            query,
            reprompt_checks=reprompt_checks,
            max_attempts=len(ordered_responses),
        )

        for input_args, expected_output in input_output_examples:
            assert synthesized_python_fn.run(*input_args) == expected_output


@runllms
def test_function_synthesis_with_real_llm():
    """Tests for synthesize_python_function_with_llm() with a real LLM."""

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
        cache = FilePretrainedLargeModelCache(cache_path)
        llm = OpenAIModel("gpt-4o-mini", cache)

        function_name = "count_vowels"
        input_output_examples = [(["nomsy"], 2), (["boooooo"], 6)]
        inputs = [i for i, _ in input_output_examples]
        output_check_fns = [lambda x, o=o: x == o for _, o in input_output_examples]
        reprompt_checks = [
            SyntaxRepromptCheck(),
            FunctionOutputRepromptCheck(
                function_name, inputs, output_check_fns, function_timeout=1.0
            ),
        ]

        query = Query(
            """Generate a Python function of the form

def count_vowels(s: str) -> int:
    # your code here

Note that "y" should be counted.
"""
        )

        synthesized_python_fn = synthesize_python_function_with_llm(
            function_name,
            llm,
            query,
            reprompt_checks=reprompt_checks,
        )

        for input_args, expected_output in input_output_examples:
            assert synthesized_python_fn.run(*input_args) == expected_output


###Tests for the SemanticsPythonRepromptCheck class.###


def test_valid_stub():
    """Test that a valid stub does not trigger a reprompt."""
    query = Query("Initial prompt")
    response = Response(
        '{"proposal": {"python_stub": "x = 1\\nprint(x)"}}',
        {},
    )
    response_data = json.loads(response.text)  # output from llm
    python_stub = response_data["proposal"]["python_stub"]
    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response, python_stub)

    assert result is None, "Valid stub should not trigger a reprompt."


def test_syntax_error_stub():
    """Test that a stub with syntax errors triggers a reprompt."""
    query = Query("Initial prompt")
    response = Response(
        '{"proposal": {"python_stub": "x = "}}',
        {},
    )
    response_data = json.loads(response.text)
    python_stub = response_data["proposal"]["python_stub"]

    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response, python_stub)
    assert result is not None, "Syntax error stub should trigger a reprompt."
    assert "invalid Python syntax" in result.prompt


def test_execution_error_stub():
    """Test that a stub with execution errors triggers a reprompt."""
    query = Query("Initial prompt")
    response = Response(
        '{"proposal": {"python_stub": "raise ValueError(\\"Error\\")"}}',
        {},
    )
    response_data = json.loads(response.text)
    python_stub = response_data["proposal"]["python_stub"]

    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response, python_stub)
    assert result is not None, "Execution error stub should trigger a reprompt."
    assert "raised an error during execution" in result.prompt


def test_undefined_variable_stub():
    """Test that a stub with undefined variables triggers a reprompt."""
    query = Query("Initial prompt")
    response = Response(
        '{"proposal": {"python_stub": "print(y)"}}',
        {},
    )
    response_data = json.loads(response.text)
    python_stub = response_data["proposal"]["python_stub"]
    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response, python_stub)
    assert result is not None, "Undefined variable stub should trigger a reprompt."
    assert "is not defined" in result.prompt
