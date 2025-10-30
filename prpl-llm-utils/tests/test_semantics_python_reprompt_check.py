"""Tests for the SemanticsPythonRepromptCheck class."""

from prpl_llm_utils.code import SemanticsPythonRepromptCheck


class Query:
    """Minimal Query class for testing."""

    def __init__(self, prompt: str, imgs=None, hyperparameters=None):
        self.prompt = prompt
        self.imgs = imgs or []
        self.hyperparameters = hyperparameters or {}


class Response:
    """Minimal Response class for testing."""

    def __init__(self, text: str):
        self.text = text


def test_valid_stub():
    """Test that a valid stub does not trigger a reprompt."""
    query = Query("Initial prompt")
    response = Response('{"proposal": {"semantics_py_stub": "x = 1\\nprint(x)"}}')

    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response)

    assert result is None, "Valid stub should not trigger a reprompt."


def test_syntax_error_stub():
    """Test that a stub with syntax errors triggers a reprompt."""
    query = Query("Initial prompt")
    response = Response('{"proposal": {"semantics_py_stub": "x = "}}')

    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response)
    assert result is not None, "Syntax error stub should trigger a reprompt."
    assert "invalid Python syntax" in result.prompt


def test_execution_error_stub():
    """Test that a stub with execution errors triggers a reprompt."""
    query = Query("Initial prompt")
    response = Response(
        '{"proposal": {"semantics_py_stub": "raise ValueError(\\"Error\\")"}}'
    )

    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response)
    assert result is not None, "Execution error stub should trigger a reprompt."
    assert "raised an error during execution" in result.prompt


def test_undefined_variable_stub():
    """Test that a stub with undefined variables triggers a reprompt."""
    query = Query("Initial prompt")
    response = Response('{"proposal": {"semantics_py_stub": "print(y)"}}')

    check = SemanticsPythonRepromptCheck()
    result = check.get_reprompt(query, response)
    assert result is not None, "Undefined variable stub should trigger a reprompt."
    assert "is not defined" in result.prompt
