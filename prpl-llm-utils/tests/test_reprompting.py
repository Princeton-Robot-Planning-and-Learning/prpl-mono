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
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
        cache = FilePretrainedLargeModelCache(cache_path)

        query1 = Query("What is 1+1?")
        query2 = Query("You said 1. That's not right. What is 1+1?")
        response1 = Response("1", {})
        response2 = Response("2", {})
        canned_responses = {query1: response1, query2: response2}
        canned_llm = CannedResponseModel(canned_responses, cache)

        def check_fn(_, r):
            if r.text == "2":
                return None
            return query2

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


def test_query_with_reprompts_seed():
    """Test that seed parameter works with query_with_reprompts."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
        cache = FilePretrainedLargeModelCache(cache_path)

        # Create queries with hyperparameters including seed
        query1_seed1 = Query("What is 1+1?", hyperparameters={"seed": 1})
        query1_seed2 = Query("What is 1+1?", hyperparameters={"seed": 2})
        query2_seed1 = Query(
            "You said 1. That's not right. What is 1+1?", hyperparameters={"seed": 1}
        )
        query2_seed2 = Query(
            "You said 3. That's not right. What is 1+1?", hyperparameters={"seed": 2}
        )

        response1_seed1 = Response("1", {})
        response1_seed2 = Response("3", {})
        response2_seed1 = Response("2", {})
        response2_seed2 = Response("2", {})

        canned_responses = {
            query1_seed1: response1_seed1,
            query1_seed2: response1_seed2,
            query2_seed1: response2_seed1,
            query2_seed2: response2_seed2,
        }
        canned_llm = CannedResponseModel(canned_responses, cache)

        def check_fn(_, r):
            if r.text == "2":
                return None
            if r.text == "1":
                return Query(
                    "You said 1. That's not right. What is 1+1?",
                    hyperparameters={"seed": 1},
                )
            if r.text == "3":
                return Query(
                    "You said 3. That's not right. What is 1+1?",
                    hyperparameters={"seed": 2},
                )
            return None

        checker = FunctionalRepromptCheck(check_fn)

        # Test with seed=1
        response1 = query_with_reprompts(
            canned_llm, Query("What is 1+1?"), [checker], seed=1
        )
        assert response1.text == "2"
        assert len(response1.metadata["queries"]) == 2

        # Test with seed=2 - should get different reprompt path
        response2 = query_with_reprompts(
            canned_llm, Query("What is 1+1?"), [checker], seed=2
        )
        assert response2.text == "2"
        assert len(response2.metadata["queries"]) == 2

        # Verify different seeds used different cached entries
        assert response1.metadata["queries"][0] != response2.metadata["queries"][0]
