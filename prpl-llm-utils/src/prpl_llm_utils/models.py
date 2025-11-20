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

from prpl_llm_utils.cache import PretrainedLargeModelCache, ResponseNotFound
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

    def run_query(self, query: Query, bypass_cache: bool = False) -> Response:
        """Run a built query."""
        # Try to load from the cache.
        model_id = self.get_id()
        try:
            if bypass_cache:
                raise ResponseNotFound()
            response = self._cache.try_load_response(query, model_id)
            logging.debug("Loaded model response from cache.")
        except ResponseNotFound as e:
            # No response found, so we need to query.
            if self._use_cache_only:
                raise ValueError("No cached response found for prompt.") from e
            logging.debug(f"Querying model {self.get_id()} with new prompt.")
            response = self._run_query(query)
            # Save the response to cache.
            self._cache.save(query, model_id, response)
        return response

    def query(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None = None,
        hyperparameters: dict[str, Hashable] | None = None,
        bypass_cache: bool = False,
    ) -> Response:
        """Build and run a query."""
        query = Query(prompt, imgs=imgs, hyperparameters=hyperparameters)
        return self.run_query(query, bypass_cache)


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
        # To avoid possible issues with caching, assume that each query is asked
        # once, and we always want the same response for that query.
        self._seen_queries: set[Query] = set()
        self._responses = responses
        super().__init__(cache, use_cache_only)

    def get_id(self) -> str:
        return "ordered"

    def _run_query(self, query: Query) -> Response:
        self._seen_queries.add(query)
        idx = len(self._seen_queries) - 1
        return self._responses[idx]
