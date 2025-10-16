"""Data structures for types, objects, and variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Optional, Set
)

from prpl_utils.utils import consistent_hash


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type."""

    name: str
    parent: Optional[Type] = field(default=None, repr=False)

    def __call__(self, name: str) -> TypedEntity:
        """Convenience method for generating _TypedEntities."""
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self) -> int:
        return consistent_hash(self.name)

    def get_ancestors(self) -> Set[Type]:
        """Get the set of all types that are ancestors (i.e. parents,
        grandparents, great-grandparents, etc.) of the current type."""
        curr_type: Optional[Type] = self
        ancestors_set = set()
        while curr_type is not None:
            ancestors_set.add(curr_type)
            curr_type = curr_type.parent
        return ancestors_set

@dataclass(frozen=True, order=True, repr=False)
class TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).

    Should not be instantiated externally.
    """

    name: str
    type: Type

    @cached_property
    def _str(self) -> str:
        return f"{self.name}:{self.type.name}"

    @cached_property
    def _hash(self) -> int:
        return consistent_hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def is_instance(self, t: Type) -> bool:
        """Return whether this entity is an instance of the given type, taking
        hierarchical typing into account."""
        cur_type: Optional[Type] = self.type
        while cur_type is not None:
            if cur_type == t:
                return True
            cur_type = cur_type.parent
        return False


@dataclass(frozen=True, order=True, repr=False)
class Object(TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name does
    not start with "?"."""

    def __post_init__(self) -> None:
        assert not self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Variable(TypedEntity):
    """Struct defining a Variable, which is just a _TypedEntity whose name
    starts with "?"."""

    def __post_init__(self) -> None:
        assert self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash
