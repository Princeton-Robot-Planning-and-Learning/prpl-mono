"""Tests for code.py."""

import tempfile
from pathlib import Path

import pytest

from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
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
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
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

    response_with_syntax_error = Response(
        """```python
def count_good_dogs(dog_names: list[str) -> int:
    return len(dog_names)
```    
""",
        {},
    )

    response_with_exception_failure = Response(
        """```python
def count_good_dogs(dog_names: list[str]) -> int:
    raise ValueError("Oops!")
```    
""",
        {},
    )

    response_with_semantic_failure = Response(
        """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return 2
```    
""",
        {},
    )

    response_with_infinite_loop = Response(
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

    response_with_correct_answer = Response(
        """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return len(dog_names)
```    
""",
        {},
    )

    ordered_responses = [
        response_with_syntax_error,
        response_with_exception_failure,
        response_with_semantic_failure,
        response_with_infinite_loop,
        response_with_correct_answer,
    ]

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

    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
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
