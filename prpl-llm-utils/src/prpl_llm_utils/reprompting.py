"""Data structures and methods for checking responses and reprompting."""

import abc
from typing import Callable

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.structs import Query, Response


class RepromptCheck(abc.ABC):
    """Check if reprompting is necessary and create a reprompt if so.

    Note that the reprompt needs to include any relevant history; there
    is no conversation carried between queries.
    """

    @abc.abstractmethod
    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        """Return a new query or None if the response was a success."""


class FunctionalRepromptCheck(RepromptCheck):
    """A reprompt check defined by a given function."""

    def __init__(self, fn: Callable[[Query, Response], Query | None]) -> None:
        self._fn = fn

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        return self._fn(query, response)


def create_reprompt_from_error_message(
    query: Query, response: Response, error_msg: str
) -> Query:
    """Append new text to the original query."""
    addendum = f"\nPreviously, you responded:\n{response.text}"
    addendum += f"\nYour previous response had the following error:\n{error_msg}"
    new_prompt = query.prompt + addendum
    return Query(new_prompt, query.imgs, query.hyperparameters)


def query_with_reprompts(
    model: PretrainedLargeModel,
    query: Query,
    reprompt_checks: list[RepromptCheck],
    max_attempts: int = 5,
) -> Response:
    """Query the model until all reprompt checks pass.

    Raise a RuntimeError if still failing after max_attempts.
    """
    all_queries: list[Query] = [query]
    all_responses: list[Response] = []
    for _ in range(max_attempts):
        # Query the model.
        response = model.run_query(query)
        all_responses.append(response)
        # Run all checks in order.
        new_query: Query | None = None
        for reprompt_check in reprompt_checks:
            new_query = reprompt_check.get_reprompt(query, response)
            if new_query is not None:
                break
        # If all checks passed, we're done.
        if new_query is None:
            # Finalize the metadata.
            metadata = {"queries": all_queries, "responses": all_responses}
            # Create the final response.
            response = Response(response.text, metadata)
            return response
        # Otherwise, need to re-query.
        query = new_query
        all_queries.append(query)
    raise RuntimeError(f"Reprompting failed after {max_attempts} attempts")
