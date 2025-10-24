"""Base interface."""

import abc

import spatialmath


class BaseInterface(abc.ABC):
    """Base interface."""

    @abc.abstractmethod
    def get_base_state(self) -> spatialmath.SE2:
        """Get the current base state."""


class FakeBaseInterface(BaseInterface):
    """Fake base interface."""

    def __init__(self):
        self.base_state = spatialmath.SE2(x=0, y=0, theta=0)

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_state


class RealBaseInterface(BaseInterface):
    """Real base interface.

    Coming soon!
    """

    def __init__(self):
        self.base_state = spatialmath.SE2(x=0, y=0, theta=0)

    def get_base_state(self) -> spatialmath.SE2:
        raise NotImplementedError("Real base interface not implemented yet.")
