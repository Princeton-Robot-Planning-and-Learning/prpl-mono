"""Tests for reprompting.py."""

import tempfile
from pathlib import Path

import pytest

from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import CannedResponseModel
from prpl_llm_utils.reprompting import FunctionalRepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query, Response


def test_query_with_reprompts():
    """Tests for query_with_reprompts()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
    cache = FilePretrainedLargeModelCache(cache_path)

    query1 = Query("What is 1+1?")
    query2 = Query("You said 1. That's not right. What is 1+1?")
    response1 = Response("1", {})
    response2 = Response("2", {})
    canned_responses = {query1: response1, query2: response2}
    canned_llm = CannedResponseModel(canned_responses, cache)

    check_fn = lambda _, r: None if r.text == "2" else query2
    checker = FunctionalRepromptCheck(check_fn)

    # Test successful querying.
    response = query_with_reprompts(canned_llm, query1, [checker])
    assert response.text == "2"
    assert len(response.metadata["queries"]) == 2
    assert len(response.metadata["responses"]) == 2

    # Test failure due to max attempts.
    with pytest.raises(RuntimeError) as e:
        query_with_reprompts(canned_llm, query1, [checker], max_attempts=1)
    assert "Reprompting failed after 1 attempts" in str(e)
