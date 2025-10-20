"""Gym spaces that use relational data structures."""

from __future__ import annotations

from typing import Any, Collection, List, Sequence, Set
from typing import Type as TypingType
from typing import cast

import numpy as np
from gymnasium.spaces import Box, Space

from relational_structs.common import Array
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object, Type


class ObjectCentricStateSpace(Space):
    """A space for where members are State instances.

    Different members may have different objects, but the types of the
    objects are fixed.
    """

    def __init__(
        self,
        types: Collection[Type],
        state_cls: TypingType[ObjectCentricState] = ObjectCentricState,
    ) -> None:
        super().__init__()
        self._types = types
        self._state_cls = state_cls

    @property
    def types(self) -> Set[Type]:
        """Expose the types in this space."""
        return set(self._types)

    def contains(self, x: Any) -> bool:
        assert isinstance(x, ObjectCentricState)
        return all(any(o.is_instance(t) for t in self._types) for o in x)

    @property
    def is_np_flattenable(self) -> bool:
        # Not flattenable because number of objects can change.
        return False

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> ObjectCentricState:
        raise NotImplementedError("Sampling is not well-defined.")

    def to_box(
        self,
        constant_objects: list[Object],
        type_features: dict[Type, list[str]],
        dtype: Any = np.float32,
    ) -> ObjectCentricBoxSpace:
        """Create an ObjectCentricState given a fixed object list."""
        return ObjectCentricBoxSpace(
            constant_objects, type_features, state_cls=self._state_cls,
            dtype=dtype
        )


class ObjectCentricBoxSpace(Box):
    """A box space where elements are vectors, but the entries represent
    flattened features for a constant number of objects."""

    def __init__(
        self,
        constant_objects: list[Object],
        type_features: dict[Type, list[str]],
        state_cls: TypingType[ObjectCentricState] = ObjectCentricState,
        dtype: Any = np.float32,
    ) -> None:
        self.constant_objects = constant_objects
        self.type_features = type_features
        self.state_cls = state_cls
        num_dims = sum(len(type_features[o.type]) for o in constant_objects)
        shape = (num_dims,)
        low = np.full(shape, -np.inf, dtype=dtype)
        high = np.full(shape, np.inf, dtype=dtype)
        super().__init__(low, high, shape, dtype=dtype)

    def vectorize(self, object_centric_state: ObjectCentricState) -> Array:
        """Create a vector in this space for the given object-centric state."""
        return object_centric_state.vec(self.constant_objects, dtype=self.dtype)

    def devectorize(self, vec: Array) -> ObjectCentricState:
        """Create an object-centric state from a vector that is in this
        space."""
        assert self.contains(vec)
        return self.state_cls.from_vec(vec, self.constant_objects, self.type_features)

    def create_markdown_description(self) -> str:
        """Create a markdown-format description of this space."""
        md_table_str = "| **Index** | **Object** | **Feature** |"
        md_table_str += "\n| --- | --- | --- |"
        idx = 0
        for obj in self.constant_objects:
            for feat in self.type_features[obj.type]:
                md_table_str += f"\n| {idx} | {obj.name} | {feat} |"
                idx += 1
        return f"The entries of an array in this Box space correspond to the following object features:\n{md_table_str}\n"  # pylint: disable=line-too-long


class ObjectSequenceSpace(Space):
    """A space for where members are sequences of objects corresponding to a
    given sequence of types.

    For example, if space = ObjectSequenceSpace([dog, cat, dog]), then
    one member might be [nomsy_the_dog, alfred_the_cat, goldie_the_dog].
    """

    def __init__(self, types: Sequence[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self) -> List[Type]:
        """Expose the types in this space."""
        return list(self._types)

    def contains(self, x: Any) -> bool:
        x = cast(Sequence[Object], x)
        if len(x) != len(self._types):
            return False
        return all(o.is_instance(t) for o, t in zip(x, self._types))

    @property
    def is_np_flattenable(self) -> bool:
        # Is flattenable because the length of members is fixed.
        return True

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> ObjectCentricState:
        raise NotImplementedError("Sampling is not well-defined.")


class ObjectSequenceBoxSpace(Space):
    """A wrapper around an ObjectSequenceSpace and a Box space. This is a very
    common kind of space, e.g., for defining parameterized options that have
    both object and continuous parameters.

    Elements are tuples where the first element is an object sequence and the
    second element is a vector in the Box space.

    NOTE: the Box input spec is much more general, but we restrict it here for
    consistency and ease of use.
    """

    def __init__(
        self, types: Sequence[Type], low: Sequence[float], high: Sequence[float]
    ) -> None:
        super().__init__()
        self._object_sequence_space = ObjectSequenceSpace(types)
        low_arr = np.array(low, dtype=np.float32)
        high_arr = np.array(high, dtype=np.float32)
        self._box_space = Box(low=low_arr, high=high_arr, dtype=np.float32)

    def contains(self, x: Any) -> bool:
        assert isinstance(x, tuple)
        object_sequence, box_params = x
        return self._object_sequence_space.contains(
            object_sequence
        ) and self._box_space.contains(box_params)

    @property
    def is_np_flattenable(self) -> bool:
        return (
            self._object_sequence_space.is_np_flattenable
            and self._box_space.is_np_flattenable
        )

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> ObjectCentricState:
        raise NotImplementedError("Sampling is not well-defined.")
