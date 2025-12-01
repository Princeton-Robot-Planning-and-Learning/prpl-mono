"""Interfaces for large language models."""

import abc
import base64
import io
import logging
import os
from typing import Hashable

import openai
import PIL.Image
from google import genai
from google.genai import types

from prpl_llm_utils.cache import PretrainedLargeModelCache
from prpl_llm_utils.structs import Query, Response


class PretrainedLargeModel(abc.ABC):
    """A pretrained large vision or language model."""

    def __init__(
        self, cache: PretrainedLargeModelCache, use_cache_only: bool = False
    ) -> None:
        self._cache = cache
        self._use_cache_only = use_cache_only

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this model.

        This identifier should include sufficient information so that
        querying the same model with the same query and same identifier
        should yield the same result.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _run_query(self, query: Query) -> Response:
        """This is the main method that subclasses must implement.

        This helper method is called by query(), which caches the
        queries and responses to disk.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _run_query_multi_response(
        self, query: Query, num_responses: int
    ) -> list[Response]:
        """Generate multiple responses for the same query.

        This helper method is called by run_query_multi_response(),
        which handles caching. Subclasses must implement this to support
        generating multiple diverse responses.
        """
        raise NotImplementedError("Override me!")

    def run_query(self, query: Query, bypass_cache: bool = False) -> Response:
        """Run a built query."""
        # Use run_query_multi_response with num_responses=1 as a special case
        responses = self.run_query_multi_response(
            query, num_responses=1, bypass_cache=bypass_cache
        )
        return responses[0]

    def query(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None = None,
        hyperparameters: dict[str, Hashable] | None = None,
        bypass_cache: bool = False,
        seed: int | None = None,
    ) -> Response:
        """Build and run a query.

        Args:
            prompt: The text prompt
            imgs: Optional list of images
            hyperparameters: Optional model-specific hyperparameters
            bypass_cache: If True, always query the model
            seed: Optional seed for reproducibility. Different seeds with the
                  same prompt will be cached separately.
        """
        # Add seed to hyperparameters if provided
        if seed is not None:
            hyperparameters = hyperparameters or {}
            hyperparameters = {**hyperparameters, "seed": seed}

        query = Query(prompt, imgs=imgs, hyperparameters=hyperparameters)
        return self.run_query(query, bypass_cache)

    def query_multi_response(
        self,
        prompt: str,
        num_responses: int,
        imgs: list[PIL.Image.Image] | None = None,
        hyperparameters: dict[str, Hashable] | None = None,
        bypass_cache: bool = False,
        seed: int | None = None,
    ) -> list[Response]:
        """Build and run a query that returns multiple responses.

        Args:
            prompt: The text prompt
            num_responses: Number of responses to generate
            imgs: Optional list of images
            hyperparameters: Optional model-specific hyperparameters
            bypass_cache: If True, always query the model
            seed: Optional seed for reproducibility. Different seeds with the
                  same prompt will be cached separately.
        """
        # Add seed to hyperparameters if provided
        if seed is not None:
            hyperparameters = hyperparameters or {}
            hyperparameters = {**hyperparameters, "seed": seed}

        query = Query(prompt, imgs=imgs, hyperparameters=hyperparameters)
        return self.run_query_multi_response(query, num_responses, bypass_cache)

    def run_query_multi_response(
        self, query: Query, num_responses: int, bypass_cache: bool = False
    ) -> list[Response]:
        """Run a query and return multiple responses.

        This method supports partial caching - if some responses are cached
        and others are not, only the missing responses will be queried from
        the model.

        Args:
            query: The query to run
            num_responses: Number of responses to generate
            bypass_cache: If True, always query the model even if cached

        Returns:
            A list of num_responses Response objects
        """
        model_id = self.get_id()

        # Try to load cached responses
        cached_responses: list[Response | None]
        if bypass_cache:
            cached_responses = [None] * num_responses
        else:
            cached_responses = self._cache.try_load_responses(
                query, model_id, num_responses
            )

        # Identify which responses need to be queried
        missing_indices = [i for i, resp in enumerate(cached_responses) if resp is None]

        if missing_indices:
            if self._use_cache_only:
                raise ValueError(
                    f"Missing cached responses at indices {missing_indices}."
                )

            # Query for all missing responses at once
            logging.debug(
                f"Querying model {self.get_id()} for {len(missing_indices)} "
                f"new responses (indices {missing_indices})."
            )
            new_responses = self._run_query_multi_response(query, len(missing_indices))

            # Save and insert new responses at their correct indices
            for idx, new_response in zip(missing_indices, new_responses):
                self._cache.save_response_at_index(query, model_id, new_response, idx)
                cached_responses[idx] = new_response

        # At this point, all responses should be non-None
        assert all(r is not None for r in cached_responses)
        return cached_responses  # type: ignore


class OpenAIModel(PretrainedLargeModel):
    """Common interface with methods for all OpenAI-based models."""

    def __init__(
        self,
        model_name: str,
        cache: PretrainedLargeModelCache,
        use_cache_only: bool = False,
    ) -> None:
        self._model_name = model_name
        assert "OPENAI_API_KEY" in os.environ, "Need to set OPENAI_API_KEY"
        super().__init__(cache, use_cache_only)
        self._client = openai.OpenAI()

    def get_id(self) -> str:
        return self._model_name

    def _run_query(self, query: Query) -> Response:
        kwargs = query.hyperparameters or {}

        # Build message content
        if query.imgs:
            # For vision models, use content array with text and images
            content: list[dict] = [{"type": "text", "text": query.prompt}]

            for img in query.imgs:
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                )

            messages = [{"role": "user", "content": content}]
        else:
            # For text-only, use simple string content
            messages = [{"role": "user", "content": query.prompt}]

        completion = self._client.chat.completions.create(  # type: ignore[call-overload]
            messages=messages,  # type: ignore
            model=self._model_name,
            **kwargs,  # type: ignore
        )
        assert len(completion.choices) == 1
        text = completion.choices[0].message.content
        metadata = completion.usage.to_dict() if completion.usage else {}
        return Response(text, metadata)

    def _run_query_multi_response(
        self, query: Query, num_responses: int
    ) -> list[Response]:
        kwargs = query.hyperparameters or {}

        # Build message content
        if query.imgs:
            # For vision models, use content array with text and images
            content: list[dict] = [{"type": "text", "text": query.prompt}]

            for img in query.imgs:
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                )

            messages = [{"role": "user", "content": content}]
        else:
            # For text-only, use simple string content
            messages = [{"role": "user", "content": query.prompt}]

        # Use the 'n' parameter to generate multiple responses
        completion = self._client.chat.completions.create(  # type: ignore[call-overload]
            messages=messages,  # type: ignore
            model=self._model_name,
            n=num_responses,
            **kwargs,  # type: ignore
        )

        assert len(completion.choices) == num_responses
        responses = []
        base_metadata = completion.usage.to_dict() if completion.usage else {}

        for i, choice in enumerate(completion.choices):
            text = choice.message.content
            # Add choice-specific metadata
            metadata = {**base_metadata, "choice_index": i}
            responses.append(Response(text, metadata))

        return responses


class GeminiModel(PretrainedLargeModel):
    """Common interface with methods for all Gemini-based models."""

    def __init__(
        self,
        model_name: str,
        cache: PretrainedLargeModelCache,
        use_cache_only: bool = False,
        thinking_budget: int = 0,
    ) -> None:
        self._model_name = model_name
        assert "GEMINI_API_KEY" in os.environ, "Need to set GEMINI_API_KEY"
        self._client = genai.Client()
        self._config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        super().__init__(cache, use_cache_only)

    def get_id(self) -> str:
        return self._model_name

    def _run_query(self, query: Query) -> Response:
        prompt = query.prompt
        imgs = query.imgs
        hp = query.hyperparameters

        if hp is not None:
            temperature = hp.get("temperature", 1.0)
            assert isinstance(temperature, float), "temperature must be float"
            self._config.temperature = temperature

        if imgs is None:
            imgs = []

        contents = [prompt] + imgs

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,  # type: ignore
            config=self._config,
        )

        assert response.text is not None

        return Response(response.text, metadata={"model": self._model_name})

    def _run_query_multi_response(
        self, query: Query, num_responses: int
    ) -> list[Response]:
        # Gemini API doesn't support generating multiple responses in one call,
        # so we make multiple independent calls
        responses = []
        for i in range(num_responses):
            response = self._run_query(query)
            # Add index to metadata to distinguish responses
            response.metadata["response_index"] = i
            responses.append(response)
        return responses


class CannedResponseModel(PretrainedLargeModel):
    """A model that returns responses from a dictionary and raises an error if
    no matching query is found.

    This is useful for development and testing.
    """

    def __init__(
        self,
        query_to_response: dict[Query, Response],
        cache: PretrainedLargeModelCache,
        use_cache_only: bool = False,
    ) -> None:
        self._query_to_response = query_to_response
        super().__init__(cache, use_cache_only)

    def get_id(self) -> str:
        return "canned"

    def _run_query(self, query: Query) -> Response:
        return self._query_to_response[query]

    def _run_query_multi_response(
        self, query: Query, num_responses: int
    ) -> list[Response]:
        # For testing, just return the same response multiple times
        # In real usage, tests can populate different responses if needed
        base_response = self._query_to_response[query]
        responses = []
        for i in range(num_responses):
            # Create a copy with modified metadata
            metadata = {**base_response.metadata, "response_index": i}
            responses.append(Response(base_response.text, metadata))
        return responses


class OrderedResponseModel(PretrainedLargeModel):
    """A model that returns responses from a list and raises an error if the
    index is exceeded.

    This is useful for development and testing.
    """

    def __init__(
        self,
        responses: list[Response],
        cache: PretrainedLargeModelCache,
        use_cache_only: bool = False,
    ) -> None:
        # Track the next response index to return
        self._next_response_idx = 0
        self._responses = responses
        super().__init__(cache, use_cache_only)

    def get_id(self) -> str:
        return "ordered"

    def _run_query(self, query: Query) -> Response:
        idx = self._next_response_idx
        self._next_response_idx += 1
        if idx >= len(self._responses):
            raise IndexError(
                f"Requested response at index {idx} but only "
                f"{len(self._responses)} responses available"
            )
        return self._responses[idx]

    def _run_query_multi_response(
        self, query: Query, num_responses: int
    ) -> list[Response]:
        # Return the next num_responses from the list
        responses = []
        for _ in range(num_responses):
            idx = self._next_response_idx
            self._next_response_idx += 1
            if idx >= len(self._responses):
                raise IndexError(
                    f"Requested response at index {idx} but only "
                    f"{len(self._responses)} responses available"
                )
            responses.append(self._responses[idx])
        return responses
