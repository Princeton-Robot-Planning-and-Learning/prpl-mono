"""Core LPP program representation that wraps logical program strings."""

from typing import Any, Callable, Generic, TypeVar

from programmatic_policy_learning.lpp.dsl import primitives

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class StateActionProgram(Generic[ObsType, ActType]):
    """A logical program that operates on state-action pairs.

    Takes a program string (like "cell_is_value(1, a, s)") and makes it
    executable by converting it to a Python function when called.

    This class is generic and can work with any observation and action
    types.
    """

    def __init__(
        self, program_string: str, eval_context: dict[str, Any] | None = None
    ) -> None:
        """Initialize with a program string and optional evaluation context.

        If evaluation context isNone, uses default grid-based primitives
        for backwards compatibility.
        """

        self.program = program_string
        self.eval_context = (
            self._get_default_eval_context() if eval_context is None else eval_context
        )
        self.compiled_func: Callable[[ObsType, ActType], bool] | None = None

    def _get_default_eval_context(self) -> dict[str, Any]:
        """Get default evaluation context with grid-based primitives."""

        # Automatically import all public functions from primitives module
        return {
            name: getattr(primitives, name)
            for name in dir(primitives)
            if not name.startswith("_") and callable(getattr(primitives, name))
        }

    def __call__(self, s: ObsType, a: ActType) -> bool:
        """Execute the program on a state-action pair.

        Args:
            s: Observation/state of any type
            a: Action of any type

        Returns:
            Boolean result of program evaluation
        """
        if self.compiled_func is None:
            # Convert string to executable function using provided evaluation context
            self.compiled_func = eval(f"lambda s, a: {self.program}", self.eval_context)

        return self.compiled_func(s, a)

    def __str__(self) -> str:
        """String representation of the program."""
        return self.program

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"StateActionProgram('{self.program}')"
