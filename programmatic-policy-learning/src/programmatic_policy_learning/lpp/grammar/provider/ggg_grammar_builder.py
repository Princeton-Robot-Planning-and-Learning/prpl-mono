"""GGG provider grammar builder."""

from typing import Any

from programmatic_policy_learning.lpp.grammar.base_grammar_builder import (
    BaseGrammarBuilder,
)
from programmatic_policy_learning.lpp.grammar.constants import (
    CONDITION,
    DIRECTION,
    LOCAL_PROGRAM,
    NEGATIVE_NUM,
    POSITIVE_NUM,
    START,
    VALUE,
)


class GGGGrammarBuilder(BaseGrammarBuilder):
    """Grammar builder for GGG (Grid Game Generator) environments."""

    def get_provider_name(self) -> str:
        """Return the name of the provider this grammar builder supports."""
        return "ggg"

    def get_supported_environments(self) -> list[str]:
        """Return list of specific GGG environment names this grammar
        supports."""
        return ["TwoPileNim", "GridWorld"]  # Add more GGG environments as needed

    def create_grammar(
        self, object_types: tuple[str, ...]
    ) -> dict[int, tuple[Any, list[float]]]:
        """Create a grammar definition for generating LPP programs for GGG
        environments.

        Args:
            object_types: Available object types/values in the GGG environment.

        Returns:
            Grammar dictionary with production rules and probabilities.
        """
        grammar = {
            START: (
                [
                    ["at_cell_with_value(", VALUE, ",", LOCAL_PROGRAM, ", s)"],
                    ["at_action_cell(", LOCAL_PROGRAM, ", a, s)"],
                ],
                [0.5, 0.5],
            ),
            LOCAL_PROGRAM: (
                [
                    [CONDITION],
                    [
                        "lambda cell,o : shifted(",
                        DIRECTION,
                        ",",
                        CONDITION,
                        ", cell, o)",
                    ],
                ],
                [0.5, 0.5],
            ),
            CONDITION: (
                [
                    ["lambda cell,o : cell_is_value(", VALUE, ", cell, o)"],
                    [
                        "lambda cell,o : scanning(",
                        DIRECTION,
                        ",",
                        LOCAL_PROGRAM,
                        ",",
                        LOCAL_PROGRAM,
                        ", cell, o)",
                    ],
                ],
                [0.5, 0.5],
            ),
            DIRECTION: (
                [
                    ["(", POSITIVE_NUM, ", 0)"],
                    ["(0,", POSITIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ", 0)"],
                    ["(0,", NEGATIVE_NUM, ")"],
                    ["(", POSITIVE_NUM, ",", POSITIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ",", POSITIVE_NUM, ")"],
                    ["(", POSITIVE_NUM, ",", NEGATIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ",", NEGATIVE_NUM, ")"],
                ],
                [1.0 / 8] * 8,
            ),
            POSITIVE_NUM: ([["1"], [POSITIVE_NUM, "+1"]], [0.99, 0.01]),
            NEGATIVE_NUM: ([["-1"], [NEGATIVE_NUM, "-1"]], [0.99, 0.01]),
            VALUE: (object_types, [1.0 / len(object_types) for _ in object_types]),
        }
        return grammar
