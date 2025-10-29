"""Tests for the large language model interface."""

import tempfile
from pathlib import Path

import PIL.Image
import pytest

from prpl_llm_utils.cache import (
    FilePretrainedLargeModelCache,
    SQLite3PretrainedLargeModelCache,
)
from prpl_llm_utils.models import CannedResponseModel, GeminiModel, OpenAIModel
from prpl_llm_utils.structs import Query, Response

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_canned_response_model():
    """Tests for CannedResponseModel()."""
    canned_responses = {
        Query("Hello!"): Response("Hi!", {}),
        Query("Hello!", hyperparameters={"seed": 1}): Response("Hello!", {}),
        Query("What's up?"): Response("Nothing much.", {}),
    }
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
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
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
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


@runllms
def test_gemini_language():
    """Tests for GeminiModel() without vision."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        vlm = GeminiModel("gemini-2.5-flash-lite", cache)
        response = vlm.query("Hello!")
        assert hasattr(response, "text")
        assert hasattr(response, "metadata")
        vlm = GeminiModel("gemini-2.5-flash-lite", cache, use_cache_only=True)
        response2 = vlm.query("Hello!")
        assert response.text == response2.text
        with pytest.raises(ValueError) as e:
            vlm.query("What's up?")
        assert "No cached response found for prompt." in str(e)


@runllms
def test_gemini_vision():
    """Tests for GeminiModel() with vision."""

    def fetch_mnist_image(name: str) -> PIL.Image.Image:
        mnist_path = Path(__file__).parent / "mnist_samples" / name
        return PIL.Image.open(mnist_path)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        vlm = GeminiModel("gemini-2.5-flash", cache)
        img2 = fetch_mnist_image("2.png")
        response = vlm.query(
            "What is the attached number? Give a 1-character answer.", [img2]
        )
        img3 = fetch_mnist_image("3.png")
        response2 = vlm.query(
            "What is the attached number? Give a 1-character answer.", [img3]
        )
        assert response.text != response2.text
        assert hasattr(response, "text")
        assert hasattr(response, "metadata")
        vlm = GeminiModel("gemini-2.5-flash", cache, use_cache_only=True)
        response3 = vlm.query(
            "What is the attached number? Give a 1-character answer.", [img2]
        )
        assert response.text == response3.text
        with pytest.raises(ValueError) as e:
            vlm.query("What's up?")
        assert "No cached response found for prompt." in str(e)
