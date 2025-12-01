"""Tests for the large language model interface."""

import tempfile
from pathlib import Path

import PIL.Image
import pytest

from prpl_llm_utils.cache import (
    FilePretrainedLargeModelCache,
    SQLite3PretrainedLargeModelCache,
)
from prpl_llm_utils.models import (
    CannedResponseModel,
    GeminiModel,
    OpenAIModel,
    OrderedResponseModel,
)
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
        assert "Missing cached responses" in str(e)


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
        assert "Missing cached responses" in str(e)


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
        assert "Missing cached responses" in str(e)


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
        assert "Missing cached responses" in str(e)


def test_multi_response_basic():
    """Test basic multi-response functionality with OrderedResponseModel."""
    responses_data = [
        Response("Response 0", {"index": 0}),
        Response("Response 1", {"index": 1}),
        Response("Response 2", {"index": 2}),
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Request 3 responses
        query = Query("Test query")
        responses = llm.run_query_multi_response(query, num_responses=3)

        assert len(responses) == 3
        assert responses[0].text == "Response 0"
        assert responses[1].text == "Response 1"
        assert responses[2].text == "Response 2"


def test_multi_response_caching():
    """Test that multi-response caching works correctly."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(5)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # First request - should query the model
        query = Query("Test query")
        responses1 = llm.run_query_multi_response(query, num_responses=3)
        assert len(responses1) == 3
        assert all(r.text == f"Response {i}" for i, r in enumerate(responses1))

        # Second request - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        responses2 = llm2.run_query_multi_response(query, num_responses=3)
        assert len(responses2) == 3
        assert all(r.text == f"Response {i}" for i, r in enumerate(responses2))

        # Verify responses are identical
        for r1, r2 in zip(responses1, responses2):
            assert r1.text == r2.text


def test_multi_response_n_greater_than_m():
    """Test requesting fewer responses after requesting more (N > M)."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(5)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # First request 5 responses
        query = Query("Test query")
        responses1 = llm.run_query_multi_response(query, num_responses=5)
        assert len(responses1) == 5

        # Now request only 3 responses - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        responses2 = llm2.run_query_multi_response(query, num_responses=3)
        assert len(responses2) == 3

        # Verify first 3 responses match
        for i in range(3):
            assert responses1[i].text == responses2[i].text


def test_multi_response_n_less_than_m():
    """Test requesting more responses after requesting fewer (N < M)."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(10)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # First request 3 responses
        query = Query("Test query")
        responses1 = llm.run_query_multi_response(query, num_responses=3)
        assert len(responses1) == 3

        # Now request 5 responses - should use cache for first 3, query for last 2
        responses2 = llm.run_query_multi_response(query, num_responses=5)
        assert len(responses2) == 5

        # Verify first 3 responses match cache
        for i in range(3):
            assert responses1[i].text == responses2[i].text

        # Verify last 2 responses are new
        assert responses2[3].text == "Response 3"
        assert responses2[4].text == "Response 4"


def test_multi_response_bypass_cache():
    """Test that bypass_cache works with multi-response."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(6)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # First request with caching
        query = Query("Test query")
        responses1 = llm.run_query_multi_response(query, num_responses=3)
        assert len(responses1) == 3

        # Second request with bypass_cache - should get new responses
        responses2 = llm.run_query_multi_response(
            query, num_responses=3, bypass_cache=True
        )
        assert len(responses2) == 3

        # The responses should be different because we're getting the next 3
        # from the ordered list
        assert responses2[0].text == "Response 3"
        assert responses2[1].text == "Response 4"
        assert responses2[2].text == "Response 5"


def test_multi_response_file_cache():
    """Test multi-response with FilePretrainedLargeModelCache."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(5)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
        cache = FilePretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Request 3 responses
        query = Query("Test query")
        responses1 = llm.run_query_multi_response(query, num_responses=3)
        assert len(responses1) == 3

        # Request again from cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        responses2 = llm2.run_query_multi_response(query, num_responses=3)
        assert len(responses2) == 3

        # Verify responses match
        for r1, r2 in zip(responses1, responses2):
            assert r1.text == r2.text


def test_multi_response_with_hyperparameters():
    """Test multi-response with hyperparameters."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(5)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Request with hyperparameters
        query = Query("Test query", hyperparameters={"temperature": 0.5})
        responses = llm.run_query_multi_response(query, num_responses=3)
        assert len(responses) == 3

        # Request again with same hyperparameters - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        responses2 = llm2.run_query_multi_response(query, num_responses=3)
        assert len(responses2) == 3

        # Verify responses match
        for r1, r2 in zip(responses, responses2):
            assert r1.text == r2.text


def test_multi_response_canned_model():
    """Test multi-response with CannedResponseModel."""
    canned_responses = {
        Query("Hello!"): Response("Hi!", {}),
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = CannedResponseModel(canned_responses, cache)

        # Request multiple responses
        query = Query("Hello!")
        responses = llm.run_query_multi_response(query, num_responses=3)
        assert len(responses) == 3

        # All responses should have the same text (since it's canned)
        # but different metadata
        assert all(r.text == "Hi!" for r in responses)
        assert responses[0].metadata["response_index"] == 0
        assert responses[1].metadata["response_index"] == 1
        assert responses[2].metadata["response_index"] == 2


def test_multi_response_use_cache_only_missing():
    """Test that use_cache_only raises error when responses are missing."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(5)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Cache only 2 responses
        query = Query("Test query")
        llm.run_query_multi_response(query, num_responses=2)

        # Try to request 5 responses with use_cache_only
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        with pytest.raises(ValueError) as e:
            llm2.run_query_multi_response(query, num_responses=5)
        assert "Missing cached responses at indices" in str(e)
        assert "[2, 3, 4]" in str(e)


def test_seed_parameter_single_response():
    """Test that different seeds produce different cached responses."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(10)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Query with seed=1
        response1 = llm.query("Test prompt", seed=1)
        assert response1.text == "Response 0"

        # Query with seed=2 - should get different response
        response2 = llm.query("Test prompt", seed=2)
        assert response2.text == "Response 1"

        # Query again with seed=1 - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        response3 = llm2.query("Test prompt", seed=1)
        assert response3.text == "Response 0"
        assert response3.text == response1.text

        # Query again with seed=2 - should use cache
        response4 = llm2.query("Test prompt", seed=2)
        assert response4.text == "Response 1"
        assert response4.text == response2.text


def test_seed_parameter_multi_response():
    """Test that seed works with multi-response queries."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(20)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Query 3 responses with seed=1
        responses1 = llm.query_multi_response("Test prompt", num_responses=3, seed=1)
        assert len(responses1) == 3
        assert [r.text for r in responses1] == [
            "Response 0",
            "Response 1",
            "Response 2",
        ]

        # Query 3 responses with seed=2 - should get different responses
        responses2 = llm.query_multi_response("Test prompt", num_responses=3, seed=2)
        assert len(responses2) == 3
        assert [r.text for r in responses2] == [
            "Response 3",
            "Response 4",
            "Response 5",
        ]

        # Query again with seed=1 - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        responses3 = llm2.query_multi_response("Test prompt", num_responses=3, seed=1)
        assert len(responses3) == 3
        assert [r.text for r in responses3] == [
            "Response 0",
            "Response 1",
            "Response 2",
        ]

        # Query again with seed=2 - should use cache
        responses4 = llm2.query_multi_response("Test prompt", num_responses=3, seed=2)
        assert len(responses4) == 3
        assert [r.text for r in responses4] == [
            "Response 3",
            "Response 4",
            "Response 5",
        ]


def test_seed_none_vs_no_seed():
    """Test that seed=None behaves the same as not providing seed."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(5)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Query without seed
        response1 = llm.query("Test prompt")
        assert response1.text == "Response 0"

        # Query with seed=None - should use same cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        response2 = llm2.query("Test prompt", seed=None)
        assert response2.text == "Response 0"


def test_seed_with_other_hyperparameters():
    """Test that seed works alongside other hyperparameters."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(10)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir) / "cache.db"
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Query with temperature and seed
        response1 = llm.query(
            "Test prompt", hyperparameters={"temperature": 0.5}, seed=1
        )
        assert response1.text == "Response 0"

        # Query with different seed but same temperature
        response2 = llm.query(
            "Test prompt", hyperparameters={"temperature": 0.5}, seed=2
        )
        assert response2.text == "Response 1"

        # Query with same seed and temperature - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        response3 = llm2.query(
            "Test prompt", hyperparameters={"temperature": 0.5}, seed=1
        )
        assert response3.text == "Response 0"


def test_seed_with_file_cache():
    """Test that seed works properly with FilePretrainedLargeModelCache."""
    responses_data = [Response(f"Response {i}", {"index": i}) for i in range(10)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        cache_path = Path(cache_dir)
        cache = FilePretrainedLargeModelCache(cache_path)
        llm = OrderedResponseModel(responses_data, cache)

        # Query with seed=1
        response1 = llm.query("Test prompt", seed=1)
        assert response1.text == "Response 0"

        # Query with seed=2 - should create different cache entry
        response2 = llm.query("Test prompt", seed=2)
        assert response2.text == "Response 1"

        # Query again with seed=1 - should use cache
        llm2 = OrderedResponseModel(responses_data, cache, use_cache_only=True)
        response3 = llm2.query("Test prompt", seed=1)
        assert response3.text == "Response 0"

        # Query again with seed=2 - should use cache
        response4 = llm2.query("Test prompt", seed=2)
        assert response4.text == "Response 1"

        # Verify different cache directories were created
        cache_dirs = list(cache_path.iterdir())
        assert len(cache_dirs) == 2  # Two different cache entries for different seeds
