"""Tests for the large language model interface."""

import tempfile
from pathlib import Path

import pytest

from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import CannedResponseModel, OpenAIModel
from prpl_llm_utils.structs import Query, Response

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_canned_response_model():
    """Tests for CannedResponseModel()."""
    canned_responses = {
        Query("Hello!"): Response("Hi!", {}),
        Query("Hello!", hyperparameters={"seed": 1}): Response("Hello!", {}),
        Query("What's up?"): Response("Nothing much.", {}),
    }
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
    cache = FilePretrainedLargeModelCache(cache_path)
    llm = CannedResponseModel(canned_responses, cache)
    assert llm.query("Hello!").text == "Hi!"
    assert llm.query("Hello!", hyperparameters={"seed": 1}).text == "Hello!"
    with pytest.raises(KeyError):
        llm.query("Hi!")
    llm = CannedResponseModel(canned_responses, cache, use_cache_only=True)
    assert llm.query("Hello!").text == "Hi!"
    with pytest.raises(ValueError) as e:
        llm.query("What's up?")
    assert "No cached response found for prompt." in str(e)


@runllms
def test_openai_model():
    """Tests for OpenAIModel()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
    cache = FilePretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-4o-mini", cache)
    response = llm.query("Hello!")
    assert hasattr(response, "text")
    assert hasattr(response, "metadata")
    llm = OpenAIModel("gpt-4o-mini", cache, use_cache_only=True)
    response2 = llm.query("Hello!")
    assert response.text == response2.text
    with pytest.raises(ValueError) as e:
        llm.query("What's up?")
    assert "No cached response found for prompt." in str(e)
