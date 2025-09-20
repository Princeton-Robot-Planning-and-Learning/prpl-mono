"""Tests for cache implementations."""

import tempfile
from pathlib import Path

import pytest

from prpl_llm_utils.cache import (
    FilePretrainedLargeModelCache,
    ResponseNotFound,
    SQLite3PretrainedLargeModelCache,
)
from prpl_llm_utils.structs import Query, Response


def test_file_cache():
    """Tests for FilePretrainedLargeModelCache()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
    cache = FilePretrainedLargeModelCache(cache_path)

    query = Query("Hello!", hyperparameters={"seed": 1})
    response = Response("Hi there!", {"tokens": 5})

    # Test save and load.
    cache.save(query, "test-model", response)
    loaded_response = cache.try_load_response(query, "test-model")
    assert loaded_response.text == "Hi there!"
    assert loaded_response.metadata["tokens"] == 5

    # Test cache miss.
    with pytest.raises(ResponseNotFound):
        cache.try_load_response(Query("Different query"), "test-model")


def test_sqlite_cache():
    """Tests for SQLite3PretrainedLargeModelCache()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name) / "test.db"
    cache = SQLite3PretrainedLargeModelCache(cache_path)

    query = Query("Hello!", hyperparameters={"temperature": 0.7, "max_tokens": 100})
    response = Response("Hi there!", {"tokens": 5})

    # Test save and load.
    cache.save(query, "test-model", response)
    loaded_response = cache.try_load_response(query, "test-model")
    assert loaded_response.text == "Hi there!"
    assert loaded_response.metadata["tokens"] == 5

    # Test cache miss.
    with pytest.raises(ResponseNotFound):
        cache.try_load_response(Query("Different query"), "test-model")

    # Test hyperparameter consistency.
    different_query = Query("Hello!", hyperparameters={"different_key": 1})
    with pytest.raises(AssertionError):
        cache.save(different_query, "test-model", response)
